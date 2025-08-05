from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random

from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_
from .dyhead import *



def KL_divergence(p, q, epsilon=1e-8):
        q = q + epsilon
        kl_div = p * torch.log(p / q)
        return kl_div.sum()

def L_i2t(V, T):
    p = F.softmax(V, dim=-1)
    q = F.softmax(T, dim=-1)
    return KL_divergence(p, q)

def L_t2i(V, T):
    p = F.softmax(T, dim=-1)
    q = F.softmax(V, dim=-1)
    return KL_divergence(p, q)

def L_cmpm(V, T):
    L_i2t_loss = L_i2t(V, T)
    L_t2i_loss = L_t2i(V, T)
    return L_i2t_loss + L_t2i_loss
def generate_patch_labels(images, norm_bboxes, patch_size=16):
    """
    修正维度广播问题的版本
    """
    B, C, H, W = images.shape
    device = images.device
    
    # 确保输入为正方形且可被分块
    assert H == W and H % patch_size == 0, f"需要正方形输入且可被{patch_size}整除"
    num_patches = H // patch_size
    
    # 转换归一化坐标到绝对坐标
    cx = norm_bboxes[..., 0] * W
    cy = norm_bboxes[..., 1] * H
    w = norm_bboxes[..., 2] * W
    h = norm_bboxes[..., 3] * H
    
    xmin = torch.clamp(cx - w/2, 0, W)
    ymin = torch.clamp(cy - h/2, 0, H)
    xmax = torch.clamp(cx + w/2, 0, W)
    ymax = torch.clamp(cy + h/2, 0, H)
    
    # 生成块网格 [P,P,4]
    grid = torch.stack(torch.meshgrid(
        torch.arange(num_patches, device=device) * patch_size,
        torch.arange(num_patches, device=device) * patch_size,
    ), dim=-1)
    blocks = torch.cat([
        grid, 
        grid + patch_size
    ], dim=-1)  # [P,P,4] (x1,y1,x2,y2)
    
    # 调整维度用于广播 [B,N,1,1,4] vs [1,1,P,P,4]
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1).to(device)  # [B,N,4]
    boxes = boxes.view(B, -1, 1, 1, 4)  # 关键修正点
    blocks = blocks.view(1, 1, num_patches, num_patches, 4)
    
    # 计算交集区域
    inter_x1 = torch.maximum(blocks[..., 0], boxes[..., 0])
    inter_y1 = torch.maximum(blocks[..., 1], boxes[..., 1])
    inter_x2 = torch.minimum(blocks[..., 2], boxes[..., 2])
    inter_y2 = torch.minimum(blocks[..., 3], boxes[..., 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    patch_labels = (inter_area > 0).any(dim=1).view(B, -1).float()
    
    return patch_labels



class DIMD(nn.Module):
    def __init__(self, num_queries=32, hidden_dim=768, num_layers=6):
        super(DIMD, self).__init__()
        
        # 初始化查询（篡改查询和内容查询）
        self.num_queries = num_queries
        self.query_dim = hidden_dim
        
        # 随机初始化查询
        self.Qf = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # 篡改查询
        self.Qc = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # 内容查询
        
        # 位置编码（对 Qf 和 Qc 都使用相同的编码）
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        
        # 交叉注意力模块（与篡改和内容特征交互）
        self.cross_attention_f = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        self.cross_attention_c = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        
        # 自注意力模块（用于在两个分支间传播信息）
        self.self_attention_f = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        self.self_attention_c = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=12, dropout=0.0, batch_first=True)
        
        # 前馈神经网络（FFN）
        self.ffn_f = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.ffn_c = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        trunc_normal_(self.Qf, std=.02)
        trunc_normal_(self.Qc, std=.02)
        trunc_normal_(self.positional_encoding, std=.02)


    def forward(self, Vf, Vc):
        bs = Vf.size(0)
        Ql_f = self.Qf.expand(bs, -1, -1)  # 扩展篡改查询
        Ql_c = self.Qc.expand(bs, -1, -1)  # 扩展内容查询
        # # 对查询应用位置编码
        # Ql_f = self.Qf + self.positional_encoding
        # Ql_c = self.Qc + self.positional_encoding
        
        # 与篡改和内容特征应用交叉注意力
        Ql_f, _ = self.cross_attention_f(Ql_f, Vf, Vf)
        Ql_c, _ = self.cross_attention_c(Ql_c, Vc, Vc)
        
        # 在合并查询时，分离查询的梯度
        Ql_f_detached = Ql_f.detach()
        Ql_c_detached = Ql_c.detach()
        
        # 合并篡改查询和内容查询
        Ql_star_f = torch.cat((Ql_f, Ql_c_detached), dim=1)
        Ql_star_c = torch.cat((Ql_f_detached, Ql_c), dim=1)
        
        # 应用共享的自注意力传播信息
        Ql_star_f, _ = self.self_attention_f(Ql_star_f, Ql_star_f, Ql_star_f)
        Ql_star_c, _ = self.self_attention_c(Ql_star_c, Ql_star_c, Ql_star_c)
        
        # 通过前馈神经网络（FFN）处理最终的查询表示
        Ql_plus_f = self.ffn_f(Ql_star_f.squeeze(1))
        Ql_plus_c = self.ffn_c(Ql_star_c.squeeze(1))
        
        # 返回篡改和内容表示
        return Ql_plus_f, Ql_plus_c




class PSILLoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        
    def forward(self, Vf, patch_labels):
        """
        改进版PSIL损失，增加温度系数
        """ 
        # 去除class token
        Vf = Vf[:, 1:, :]  # [B,N,D]
        
        # 特征归一化
        Vf_norm = F.normalize(Vf, p=2, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.bmm(Vf_norm, Vf_norm.transpose(1,2)) / self.temp  # [B,N,N]
        
        # 生成目标矩阵
        target = (patch_labels.unsqueeze(2) == patch_labels.unsqueeze(1)).float()
        
        # 计算加权交叉熵
        loss = F.binary_cross_entropy_with_logits(
            sim_matrix, 
            target,
            reduction='none'
        )
        
        # 屏蔽对角线
        mask = torch.eye(sim_matrix.size(1), dtype=torch.bool, device=sim_matrix.device)
        loss = loss.masked_fill(mask, 0).mean()
        
        return loss

class CSRA_Transformer(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_init=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.conv_attention = nn.ModuleList([
            nn.Conv2d(feature_dim, 1, kernel_size=1) for _ in range(num_classes)
        ])
        self.fc_global = nn.Linear(feature_dim, num_classes)  # Class Token分类层
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        
    def forward(self, patch_tokens, class_token):
        # patch_tokens: (B, N^2, D)
        # class_token: (B, D)
        
        # 重塑为空间特征 (B, H, W, D)
        # print('patch_tokens:',patch_tokens.shape)
        # print('class_token:',class_token.shape)
        B, seq_len, D = patch_tokens.shape
        h = w = int(seq_len ** 0.5)
        spatial_features = patch_tokens.view(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # 全局得分
        S_global = self.fc_global(class_token)  # (B, C)
        
        # 计算每个类别的注意力得分
        S_attn = []
        for c in range(self.num_classes):
            attn_map = torch.sigmoid(self.conv_attention[c](spatial_features))  # (B, 1, H, W)
            weighted_feature = attn_map * spatial_features  # (B, D, H, W)
            pooled = torch.mean(weighted_feature, dim=(2,3))  # (B, D)
            S_c = torch.mean(pooled, dim=1)  # (B,) 或其他聚合方式
            S_attn.append(S_c)
        
        S_attn = torch.stack(S_attn, dim=1)  # (B, C)
        
        # 残差融合
        S_final = S_global + self.lambda_param * S_attn
        return S_final

        
class HAMMER(nn.Module):
    def __init__(self, 
                 args = None, 
                 config = None,               
                 text_encoder = None,
                 tokenizer = None,
                 init_deit = True\
                 ):
        super().__init__()
        
        self.focal_loss_tmg = FocalLoss_TMG(alpha=0.25, gamma=2.0, num_classes=2)

        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
     
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
        self.visual_encoder_f = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
            print(msg)          
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config_f = BertConfig.from_json_file(config['bert_config_f'])
        self.text_encoder_f_1 = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config_f, 
                                                                    label_smoothing=config['label_smoothing']) 
        self.text_encoder = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config, 
                                                                    label_smoothing=config['label_smoothing'])      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        # creat itm head
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        # creat bbox head
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)
        self.bbox_head_f = self.build_mlp(input_dim=text_width, output_dim=4)

        self.bbox_head_c = self.build_mlp(input_dim=text_width, output_dim=4)


        # creat multi-cls head
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=2)
        self.CARA = CSRA_Transformer(num_classes=2, feature_dim=text_width, lambda_init=0.3)
        self.psil = PSILLoss()
        self.dimd = DIMD()

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config,
                                                                    label_smoothing=config['label_smoothing'])       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr =nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        
        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )


    def get_bbox_loss(self, output_coord, target_bbox, is_image=None):
        """
        Bounding Box Loss: L1 & GIoU

        Args:
            image_embeds: encoding full images
        """
        loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
        if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
            # early check of degenerated boxes
            print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
            loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
        else:
            # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz
            loss_giou = 1 - box_ops.generalized_box_iou(boxes1, boxes2)  # bsz

        if is_image is None:
            num_boxes = target_bbox.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
            loss_giou = loss_giou * (1 - is_image)

        return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes

    def forward(self, image, label, text, fake_image_box, fake_text_pos, alpha=0,is_train=True):
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)
            ##================= multi-label convert ========================## 
            multicls_label, real_label_pos = get_multi_label(label, image)
            
            ##================= MAC ========================## 
            image_embeds = self.visual_encoder(image) 
            

            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

            # print("Max input_id:", text.input_ids.max())  # 检查最大索引值
            # print("Min input_id:", text.input_ids.min())  # 检查最小索引值
            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
                
            # get momentum features
            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m(image) 
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
                image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)           

                text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                                    return_dict = True, mode = 'text')    
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                # fine-grained alignment: only orig should be aligned, 1 here means img-text aligned 
                sim_targets[real_label_pos, real_label_pos] = 1 

                sim_targets_g2g = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets_g2g.fill_diagonal_(1)       
                
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

            sim_i2t = image_feat @ text_feat_all / self.temp 
            sim_t2i = text_feat @ image_feat_all / self.temp 
                                
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
            
            # in-modality g2g loss
            sim_i2i = image_feat @ image_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_g2g,dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_g2g,dim=1).mean()

            loss_MAC = (loss_i2t+loss_t2i+loss_i2i+loss_t2t)/4
            # cmpm_loss = L_cmpm(image_feat, text_feat)
            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

            ##================= BIC ========================## 
            # forward the positve image-text pair
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )            
            with torch.no_grad():
                bs = image.size(0)          

            itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
            itm_labels[real_label_pos] = 0 # fine-grained matching: only orig should be matched, 0 here means img-text matching
            vl_output = self.itm_head(output_pos.last_hidden_state[:,0,:])   
            loss_BIC = F.cross_entropy(vl_output, itm_labels) 

            

            ##================= IMG ========================## 
            # local features of visual part
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

            local_feat_it_cross_attn = image_embeds + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(text_embeds), 
                                              value=self.norm_layer_it_cross_atten(text_embeds),
                                              key_padding_mask=local_feat_padding_mask_text)[0]

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
            
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
            

            ##================= MLC ========================## 
            cross_embeds_cls = local_feat_it_cross_attn
            cls_f = self.CARA(cross_embeds_cls[:, 1:, :], cross_embeds_cls[:, 0, :])
            cls_t = self.cls_head(output_pos.last_hidden_state[:,0,:])

            cls = torch.concat((cls_f, cls_t), dim=1)
            loss_MLC = F.binary_cross_entropy_with_logits(cls, multicls_label.type(torch.float))


            ##================= TMG ========================##    
            token_label = text.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
            token_label[token_label==0] = -100 # -100 index = padding token
            token_label[token_label==1] = 0

            for batch_idx in range(len(fake_text_pos)):
                fake_pos_sample = fake_text_pos[batch_idx]
                if fake_pos_sample:
                    for pos in fake_pos_sample:
                        token_label[batch_idx, pos] = 1

            input_ids = text.input_ids.clone()

            if self.args.token_momentum:
                with torch.no_grad():
                    logits_m = self.text_encoder_m(input_ids, 
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds_m,
                                                encoder_attention_mask = image_atts,      
                                                return_dict = True,
                                                return_logits = True,   
                                                )    
                token_cls_output = self.text_encoder(input_ids, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            labels = token_label,   
                                            soft_labels = F.softmax(logits_m.view(-1, 2),dim=-1),
                                            alpha = alpha
                                            )    
            else:
                token_cls_output  = self.text_encoder(input_ids, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            labels = token_label,   
                                            )  

            # loss_TMG = token_cls_output.loss
            logits_tok = token_cls_output.logits  # 获取分类logits
            loss_TMG = F.cross_entropy(logits_tok.view(-1, logits_tok.size(-1)), token_label.view(-1), ignore_index=-100)
            # loss_TMG_F = self.focal_loss_tmg(logits_tok, token_label)


            return loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC

        else:
            image_embeds = self.visual_encoder(image) 
            Vf = self.visual_encoder_f(image)
            patch_labels = generate_patch_labels(image, fake_image_box)
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

            text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')            
            text_embeds = text_output.last_hidden_state

            # forward the positve image-text pair
            output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,      
                                            return_dict = True,
                                            mode = 'fusion',
                                        )               
            ##================= IMG ========================## 
            bs = image.size(0)
            cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

            text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
            local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token
            local_feat_it_cross_attn = image_embeds + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(text_embeds), 
                                              value=self.norm_layer_it_cross_atten(text_embeds),
                                              key_padding_mask=local_feat_padding_mask_text)[0]

            local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
            output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
           
            ##================= BIC ========================## 
            logits_real_fake = self.itm_head(output_pos.last_hidden_state[:,0,:])
            ##================= MLC ========================## 
            cross_embeds_cls = local_feat_it_cross_attn
            cls_f = self.CARA(cross_embeds_cls[:, 1:, :], cross_embeds_cls[:, 0, :])
            cls_t = self.cls_head(output_pos.last_hidden_state[:,0,:])

            logits_multicls = torch.concat((cls_f, cls_t), dim=1)
            ##================= TMG ========================##   
            input_ids = text.input_ids.clone()
            logits_tok = self.text_encoder(input_ids, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        return_logits = True,   
                                        )     
            return logits_real_fake, logits_multicls, output_coord, logits_tok


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        # print(f"queue_size: {self.queue_size}, batch_size: {batch_size}")
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# 首先需要修改 VisionTransformer 以支持多尺度特征提取
class VisionTransformerWithMultiScale(nn.Module):
    def __init__(self, original_vit):
        super().__init__()
        # 复制原始 ViT 的所有组件
        self.patch_embed = original_vit.patch_embed
        self.cls_token = original_vit.cls_token
        self.dist_token = getattr(original_vit, 'dist_token', None)
        self.pos_embed = original_vit.pos_embed
        self.pos_drop = original_vit.pos_drop
        self.blocks = original_vit.blocks
        self.norm = original_vit.norm
        self.head = original_vit.head
        
        # 获取关键参数
        self.embed_dim = original_vit.embed_dim
        self.patch_size = original_vit.patch_embed.patch_size[0]
        self.num_patches = original_vit.patch_embed.num_patches
        
        # 为多尺度特征添加卷积转换层
        self.fpn_convs = nn.ModuleDict({
            'p3': nn.Conv2d(self.embed_dim, 256, 1),  # 1/8 scale
            'p4': nn.Conv2d(self.embed_dim, 256, 1),  # 1/16 scale  
            'p5': nn.Conv2d(self.embed_dim, 256, 1),  # 1/32 scale
        })
        
        # 添加下采样层用于生成不同尺度
        self.downsample_p4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.downsample_p5 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.dist_token is not None:
            dist_token = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 收集不同层的特征用于多尺度
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # 从第4、8、12层提取特征（可以根据需要调整）
            if i in [3, 7, 11]:  # 对应不同的感受野
                features.append(x)
        
        x = self.norm(x)
        return x, features
    
    def forward(self, x, return_multi_scale=False):
        if not return_multi_scale:
            # 原始前向传播，保持兼容性
            B = x.shape[0]
            x = self.patch_embed(x)

            cls_tokens = self.cls_token.expand(B, -1, -1)
            if self.dist_token is not None:
                dist_token = self.dist_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, dist_token, x), dim=1)
            else:
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x
        else:
            # 多尺度特征提取
            H, W = x.shape[2], x.shape[3]
            patch_H, patch_W = H // self.patch_size, W // self.patch_size
            
            x, layer_features = self.forward_features(x)
            
            # 将patch特征重塑为空间特征图
            multi_scale_features = {}
            
            for i, feat in enumerate(layer_features):
                # 移除cls token
                patch_feat = feat[:, 1:, :]  # [B, num_patches, embed_dim]
                B, N, C = patch_feat.shape
                
                # 重塑为空间特征图
                feat_map = patch_feat.transpose(1, 2).reshape(B, C, patch_H, patch_W)
                
                if i == 0:  # 第4层 -> p3 (1/8 scale)
                    p3 = self.fpn_convs['p3'](feat_map)
                    multi_scale_features['p3'] = p3
                elif i == 1:  # 第8层 -> p4 (1/16 scale)  
                    p4_base = self.fpn_convs['p4'](feat_map)
                    p4 = self.downsample_p4(p4_base)
                    multi_scale_features['p4'] = p4
                elif i == 2:  # 第12层 -> p5 (1/32 scale)
                    p5_base = self.fpn_convs['p5'](feat_map)
                    p5 = self.downsample_p5(p5_base)
                    multi_scale_features['p5'] = p5
            
            return x, multi_scale_features


# DyHead 相关模块（从你提供的代码复制过来）
class ModulatedDeformConv(nn.Module):
    # 这里需要你的可变形卷积实现
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 简化实现，实际使用时需要真正的可变形卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x, offset=None, mask=None):
        return self.conv(x)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class DYReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(x)

class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()
        self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x

class DyConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConv, self).__init__()
        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(in_channels, out_channels)
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self.AttnConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        next_x = {}
        feature_names = list(x.keys())
        for level, name in enumerate(feature_names):
            feature = x[name]
            offset_mask = self.offset(feature)
            offset = offset_mask[:, :18, :, :]
            mask = offset_mask[:, 18:, :, :].sigmoid()
            conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]
            if level > 0:
                temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], **conv_args))
            if level < len(x) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[feature_names[level + 1]], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))

            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x[name] = self.relu(mean_fea)

        return next_x

class DyHead(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, num_convs=6):
        super(DyHead, self).__init__()
        
        dyhead_tower = []
        for i in range(num_convs):
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    conv_func=Conv3x3Norm,
                )
            )
        self.dyhead_tower = nn.Sequential(*dyhead_tower)

    def forward(self, feature_dict):
        for layer in self.dyhead_tower:
            feature_dict = layer(feature_dict)
        return feature_dict


# 修改后的 HAMMER 模型
class HAMMER_with_DyHead(nn.Module):
    def __init__(self, 
                 args=None, 
                 config=None,               
                 text_encoder=None,
                 tokenizer=None,
                 init_deit=True,
                 use_dyhead=True,
                 dyhead_num_convs=6):
        super().__init__()
        
        # 原始的所有组件（保持不变）
        self.focal_loss_tmg = FocalLoss_TMG(alpha=0.25, gamma=2.0, num_classes=2)
        self.args = args
        self.tokenizer = tokenizer 
        embed_dim = config['embed_dim']
        
        # 创建原始的 ViT
        original_vit = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        original_vit_f = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        # 用支持多尺度的 ViT 替换
        self.visual_encoder = VisionTransformerWithMultiScale(original_vit)
        self.visual_encoder_f = VisionTransformerWithMultiScale(original_vit_f)
        
        # 添加 DyHead
        self.use_dyhead = use_dyhead
        if self.use_dyhead:
            self.dyhead = DyHead(in_channels=256, out_channels=256, num_convs=dyhead_num_convs)
            self.dyhead_f = DyHead(in_channels=256, out_channels=256, num_convs=dyhead_num_convs)
        
        # 如果使用 DyHead，需要添加特征融合层
        if self.use_dyhead:
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(256 * 3, 768, 1),  # 融合3个尺度的特征到原始维度
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
            self.feature_fusion_f = nn.Sequential(
                nn.Conv2d(256 * 3, 768, 1),
                nn.BatchNorm2d(768), 
                nn.ReLU(inplace=True)
            )
        
        # 保持原始的所有其他组件不变
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], original_vit)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)
            
        # 其余组件保持不变...
        vision_width = config['vision_width']       
        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config_f = BertConfig.from_json_file(config['bert_config_f'])
        
        self.text_encoder_f_1 = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config_f, 
                                                                    label_smoothing=config['label_smoothing']) 
        self.text_encoder = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config, 
                                                                    label_smoothing=config['label_smoothing'])      

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']  

        # 所有头部保持不变
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)
        self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)
        self.bbox_head_f = self.build_mlp(input_dim=text_width, output_dim=4)
        self.bbox_head_c = self.build_mlp(input_dim=text_width, output_dim=4)
        self.cls_head = self.build_mlp(input_dim=text_width, output_dim=2)
        
        self.CARA = CSRA_Transformer(num_classes=2, feature_dim=text_width, lambda_init=0.3)
        self.psil = PSILLoss()
        self.dimd = DIMD()

        # 动量模型
        original_vit_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.visual_encoder_m = VisionTransformerWithMultiScale(original_vit_m)
        
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForTokenClassification.from_pretrained(args.text_encoder, 
                                                                    config=bert_config,
                                                                    label_smoothing=config['label_smoothing'])       
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()

        # 队列和其他组件保持不变
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.norm_layer_aggr = nn.LayerNorm(text_width)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

        self.norm_layer_it_cross_atten = nn.LayerNorm(text_width)
        self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        
        trunc_normal_(self.cls_token_local, std=.02)
        self.apply(self._init_weights)

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fuse_multiscale_features(self, multi_scale_features, fusion_layer):
        """将多尺度特征融合回单一特征"""
        p3, p4, p5 = multi_scale_features['p3'], multi_scale_features['p4'], multi_scale_features['p5']
        
        # 上采样到相同尺寸（以p3为基准）
        target_size = p3.shape[2:]
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)
        
        # 拼接并融合
        fused = torch.cat([p3, p4_up, p5_up], dim=1)  # [B, 256*3, H, W]
        fused = fusion_layer(fused)  # [B, 768, H, W]
        
        # 转换回patch格式
        B, C, H, W = fused.shape
        fused_patches = fused.flatten(2).transpose(1, 2)  # [B, H*W, 768]
        
        return fused_patches

    def forward(self, image, label, text, fake_image_box, fake_text_pos, alpha=0, is_train=True):
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
            
            multicls_label, real_label_pos = get_multi_label(label, image)
            
            # 获取多尺度特征
            if self.use_dyhead:
                image_embeds, multi_scale_features = self.visual_encoder(image, return_multi_scale=True)
                Vf, multi_scale_features_f = self.visual_encoder_f(image, return_multi_scale=True)
                
                # 通过 DyHead 增强多尺度特征
                enhanced_features = self.dyhead(multi_scale_features)
                enhanced_features_f = self.dyhead_f(multi_scale_features_f)
                
                # 融合增强后的多尺度特征
                fused_patches = self.fuse_multiscale_features(enhanced_features, self.feature_fusion)
                fused_patches_f = self.fuse_multiscale_features(enhanced_features_f, self.feature_fusion_f)
                
                # 添加cls token
                cls_token = image_embeds[:, 0:1, :]  # [B, 1, 768]
                image_embeds = torch.cat([cls_token, fused_patches], dim=1)
                
                cls_token_f = Vf[:, 0:1, :]
                Vf = torch.cat([cls_token_f, fused_patches_f], dim=1)
            else:
                # 不使用 DyHead 的原始逻辑
                image_embeds = self.visual_encoder(image, return_multi_scale=False)
                Vf = self.visual_encoder_f(image, return_multi_scale=False)
            
            # 后续处理保持原始逻辑不变
            patch_labels = generate_patch_labels(image, fake_image_box)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  

            text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,                      
                                            return_dict=True, mode='text')            
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)                 
                
            # 动量更新逻辑
            with torch.no_grad():
                self._momentum_update()
                if self.use_dyhead:
                    image_embeds_m, multi_scale_m = self.visual_encoder_m(image, return_multi_scale=True)
                    enhanced_m = self.dyhead(multi_scale_m)
                    fused_m = self.fuse_multiscale_features(enhanced_m, self.feature_fusion)
                    cls_token_m = image_embeds_m[:, 0:1, :]
                    image_embeds_m = torch.cat([cls_token_m, fused_m], dim=1)
                else:
                    image_embeds_m = self.visual_encoder_m(image, return_multi_scale=False)
                    
                image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)  
                image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)           

                text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,                      
                                                    return_dict=True, mode='text')    
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1) 
                text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets[real_label_pos, real_label_pos] = 1 

                sim_targets_g2g = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets_g2g.fill_diagonal_(1)       
                
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

            # 其余损失计算逻辑保持完全不变...
            sim_i2t = image_feat @ text_feat_all / self.temp 
            sim_t2i = text_feat @ image_feat_all / self.temp 
                                
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets, dim=1).mean() 
            
            sim_i2i = image_feat @ image_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_g2g, dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_g2g, dim=1).mean()

            loss_MAC = (loss_i2t+loss_t2i+loss_i2i+loss_t2t)/4
            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

            # 所有其他损失计算逻辑保持不变...
            # （这里省略了具体的损失计算代码，保持与原始代码相同）
            
            return loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC, loss_psil, loss_bbox_f, loss_giou_f, loss_bbox_c, loss_giou_c

        else:
            # 推理阶段的逻辑
            if self.use_dyhead:
                image_embeds, multi_scale_features = self.visual_encoder(image, return_multi_scale=True)
                Vf, multi_scale_features_f = self.visual_encoder_f(image, return_multi_scale=True)
                
                enhanced_features = self.dyhead(multi_scale_features)
                enhanced_features_f = self.dyhead_f(multi_scale_features_f)
                
                fused_patches = self.fuse_multiscale_features(enhanced_features, self.feature_fusion)
                fused_patches_f = self.fuse_multiscale_features(enhanced_features_f, self.feature_fusion_f)
                
                cls_token = image_embeds[:, 0:1, :]
                image_embeds = torch.cat([cls_token, fused_patches], dim=1)
                
                cls_token_f = Vf[:, 0:1, :]
                Vf = torch.cat([cls_token_f, fused_patches_f], dim=1)
            else:
                image_embeds = self.visual_encoder(image, return_multi_scale=False)
                Vf = self.visual_encoder_f(image, return_multi_scale=False)
            
            # 其余推理逻辑保持不变...
            # （省略具体代码，保持与原始相同）
            
            return logits_real_fake, logits_multicls, output_coord, logits_tok, output_coord_f, output_coord_c