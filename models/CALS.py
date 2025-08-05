# from functools import partial
# from models.vit import CAFormer, interpolate_pos_embed
# from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification
# import os
# import torch
# import torch.nn.functional as F
# from torch import nn
# import json
# import numpy as np
# import random

# from models import box_ops
# from tools.multilabel_metrics import get_multi_label
# from timm.models.layers import trunc_normal_



# def KL_divergence(p, q, epsilon=1e-8):
#         q = q + epsilon
#         kl_div = p * torch.log(p / q)
#         return kl_div.sum()

# def L_i2t(V, T):
#     p = F.softmax(V, dim=-1)
#     q = F.softmax(T, dim=-1)
#     return KL_divergence(p, q)

# def L_t2i(V, T):
#     p = F.softmax(T, dim=-1)
#     q = F.softmax(V, dim=-1)
#     return KL_divergence(p, q)

# def L(V, T):
#     L_i2t_loss = L_i2t(V, T)
#     L_t2i_loss = L_t2i(V, T)
#     return L_i2t_loss + L_t2i_loss



# def generate_fakemap(cx, cy, w, h, image_size=256):
   
#     if isinstance(image_size, int):
#         H, W = image_size, image_size
#     else:
#         H, W = image_size  
#     device = cx.device
#     original_size=(256, 256)
#     W_orig, H_orig = original_size  
    
   
#     cx_pixel = cx * W_orig  
#     cy_pixel = cy * H_orig  
#     w_pixel = w * W_orig    
#     h_pixel = h * H_orig   
    
  
#     sigma = torch.sqrt((h_pixel/2)**2 + (w_pixel/2)**2) 

#     x = torch.arange(image_size[0], device=device).float() 
#     y = torch.arange(image_size[1], device=device).float()
#     yy, xx = torch.meshgrid(y, x, indexing='ij')  
    
  
#     xx = xx.unsqueeze(0)  
#     yy = yy.unsqueeze(0)
#     cx_pixel = cx_pixel.view(-1, 1, 1) 
#     cy_pixel = cy_pixel.view(-1, 1, 1)
#     sigma = sigma.view(-1, 1, 1)
    
    
#     distance_sq = (xx - cx_pixel)**2 + (yy - cy_pixel)**2  
#     fake_map = torch.exp(-distance_sq / (2 * (sigma**2) + 1e-8))

   
#     fake_map = torch.clamp(fake_map, 0, 1)
#     return fake_map


# class face_cls_head(nn.Module):
#     def __init__(self, num_classes, feature_dim, lambda_init=1.0):
#         super().__init__()
#         self.num_classes = num_classes
#         self.conv_attention = nn.ModuleList([
#             nn.Conv2d(feature_dim, 1, kernel_size=1) for _ in range(num_classes)
#         ])
#         self.fc_global = nn.Linear(feature_dim, num_classes)  
#         self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        
#     def forward(self, patch_tokens, class_token):
       
#         B, seq_len, D = patch_tokens.shape
#         h = w = int(seq_len ** 0.5)
#         spatial_features = patch_tokens.view(B, h, w, D).permute(0, 3, 1, 2)  
        
       
#         S_global = self.fc_global(class_token) 
        
      
#         S_attn = []
#         for c in range(self.num_classes):
#             attn_map = torch.sigmoid(self.conv_attention[c](spatial_features)) 
#             weighted_feature = attn_map * spatial_features  
#             pooled = torch.mean(weighted_feature, dim=(2,3))  
#             S_c = torch.mean(pooled, dim=1)  
#             S_attn.append(S_c)
        
#         S_attn = torch.stack(S_attn, dim=1) 
        
     
#         S_final = S_global + self.lambda_param * S_attn
#         return S_final

# class LIG(nn.Module):
#     def __init__(self, input_size=256, output_dim=768):
#         super().__init__()
       
#         self.conv_net = nn.Sequential(
           
#             nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
            
           
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
            
          
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
            
            
#             nn.AdaptiveAvgPool2d((16,16)),
            
           
#             nn.Conv2d(256, 768, kernel_size=1),
#         )
        
#     def forward(self, fake_map):
  
#         x = self.conv_net(fake_map) 
#         x = x.flatten(2).permute(0,2,1) 
#         return x

 

# class CALS(nn.Module):
#     def __init__(self, 
#                  args = None, 
#                  config = None,               
#                  text_encoder = None,
#                  tokenizer = None,
#                  init_deit = True\
#                  ):
#         super().__init__()
        
#         self.batch_count = 0
#         self.args = args
#         self.tokenizer = tokenizer 
#         embed_dim = config['embed_dim']
     
#         self.visual_encoder = CAFormer(
#             img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
#             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))   
        
#         if init_deit:
#             checkpoint = torch.hub.load_state_dict_from_url(
#                 url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
#                 map_location="cpu", check_hash=True)
#             state_dict = checkpoint["model"]
#             pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
#             state_dict['pos_embed'] = pos_embed_reshaped
#             msg = self.visual_encoder.load_state_dict(state_dict,strict=False)
#             print(msg)          
#         vision_width = config['vision_width']       
#         bert_config = BertConfig.from_json_file(config['bert_config'])
     
#         self.text_encoder = BertForTokenClassification.from_pretrained(args.text_encoder, 
#                                                                     config=bert_config, 
#                                                                     label_smoothing=config['label_smoothing'])      

#         text_width = self.text_encoder.config.hidden_size
#         self.vision_proj = nn.Linear(vision_width, embed_dim)
#         self.text_proj = nn.Linear(text_width, embed_dim)         

#         self.temp = nn.Parameter(torch.ones([]) * config['temp'])   
#         self.queue_size = config['queue_size']
#         self.momentum = config['momentum']  

#         # creat itm head
#         self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

#         # creat bbox head
#         self.bbox_head = self.build_mlp(input_dim=text_width, output_dim=4)

#         # creat multi-cls head
#         self.cls_head = self.build_mlp(input_dim=text_width, output_dim=2)
#         self.fch = face_cls_head(num_classes=2, feature_dim=text_width, lambda_init=0.3)
#         self.LIG = LIG(input_size=(256, 256), output_dim=embed_dim)  # fake map encoder

#         # create momentum models
#         self.visual_encoder_m = CAFormer(
#             img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
#             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
#         self.vision_proj_m = nn.Linear(vision_width, embed_dim)
#         self.text_encoder_m = BertForTokenClassification.from_pretrained(args.text_encoder, 
#                                                                     config=bert_config,
#                                                                     label_smoothing=config['label_smoothing'])       
#         self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
#         self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
#                             [self.vision_proj,self.vision_proj_m],
#                             [self.text_encoder,self.text_encoder_m],
#                             [self.text_proj,self.text_proj_m],
#                            ]
        
#         self.copy_params()

#         # create the queue
#         self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
#         self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
#         self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
#         self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

#         self.norm_layer_aggr =nn.LayerNorm(text_width)
#         self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
#         self.cls_token_local_e = nn.Parameter(torch.zeros(1, 1, text_width))

#         self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)

#         self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
#         self.it_cross_attn = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
#         trunc_normal_(self.cls_token_local, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def build_mlp(self, input_dim, output_dim):
#         return nn.Sequential(
#             nn.Linear(input_dim, input_dim * 2),
#             nn.LayerNorm(input_dim * 2),
#             nn.GELU(),
#             nn.Linear(input_dim* 2, input_dim * 2),
#             nn.LayerNorm(input_dim * 2),
#             nn.GELU(),
#             nn.Linear(input_dim * 2, output_dim)
#         )


#     def get_bbox_loss(self, output_coord, target_bbox_ex, is_image=None, target_bbox_map_ids=None):
#         """
#         Bounding Box Loss: L1 & GIoU

#         Args:
#             image_embeds: encoding full images
#         """
#         n_objs = output_coord.size(0)
#         n_bbox = target_bbox_ex.size(0)

#         assert n_objs == n_bbox
#         target_bbox = target_bbox_ex

#         loss_bbox = F.l1_loss(output_coord, target_bbox, reduction='none')  # bsz, 4
#         boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
#         boxes2 = box_ops.box_cxcywh_to_xyxy(target_bbox)
#         if (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any():
#             # early check of degenerated boxes
#             print("### (boxes1[:, 2:] < boxes1[:, :2]).any() or (boxes2[:, 2:] < boxes2[:, :2]).any()")
#             loss_giou = torch.zeros(output_coord.size(0), device=output_coord.device)
#         else:
#             loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(boxes1, boxes2))  # bsz

#         if is_image is None:
#             num_boxes = target_bbox.size(0)
#         else:
#             num_boxes = torch.sum(1 - is_image)
#             loss_bbox = loss_bbox * (1 - is_image.view(-1, 1))
#             loss_giou = loss_giou * (1 - is_image)

#         return loss_bbox.sum() / num_boxes, loss_giou.sum() / num_boxes
    
#     def forward(self, image, label, text, fake_image_box, fake_text_pos, epoch,  alpha=0,is_train=True):
        
#             if is_train:
#                 with torch.no_grad():
#                     self.temp.clamp_(0.001,0.5)
#                 ##================= multi-label convert ========================## 
#                 multicls_label, real_label_pos = get_multi_label(label, image)
#                 with torch.no_grad():
#                     bs = image.size(0)   
#                 ##================= MAC ========================## 
#                 image_embeds, image_embeds_c = self.visual_encoder(image) 
#                 image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

#                 image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  

            
#                 text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
#                                                 return_dict = True, mode = 'text')            
#                 text_embeds = text_output.last_hidden_state
#                 text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
                    
#                 # get momentum features
#                 with torch.no_grad():
#                     self._momentum_update()
#                     image_embeds_m, _ = self.visual_encoder_m(image) 
#                     image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
#                     image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)           

#                     text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask = text.attention_mask,                      
#                                                         return_dict = True, mode = 'text')    
#                     text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 
#                     text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

#                     sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
#                     sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

#                     sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
              
#                     sim_targets[real_label_pos, real_label_pos] = 1 

#                     sim_targets_g2g = torch.zeros(sim_i2t_m.size()).to(image.device)
#                     sim_targets_g2g.fill_diagonal_(1)       
                    
#                     sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
#                     sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

#                 sim_i2t = image_feat @ text_feat_all / self.temp 
#                 sim_t2i = text_feat @ image_feat_all / self.temp 
                                    
#                 loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
#                 loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
                
             
#                 sim_i2i = image_feat @ image_feat_all / self.temp
#                 sim_t2t = text_feat @ text_feat_all / self.temp

#                 loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_g2g,dim=1).mean()
#                 loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_g2g,dim=1).mean()

#                 loss_MAC = (loss_i2t+loss_t2i+loss_i2i+loss_t2t)/4
#                 loss_other = L(image_feat, text_feat)
#                 loss_CSA = loss_MAC + loss_other
#                 self._dequeue_and_enqueue(image_feat_m, text_feat_m)

                
                

#                 ##================= IMG ========================## 
#                 cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

#                 text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
#                 local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token

#                 local_feat_it_cross_attn = image_embeds_c + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds_c), 
#                                                 key=self.norm_layer_it_cross_atten(text_embeds), 
#                                                 value=self.norm_layer_it_cross_atten(text_embeds),
#                                                 key_padding_mask=local_feat_padding_mask_text)[0]

#                 local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
#                                                 key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
#                                                 value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
#                 output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()

#                 cx_p, cy_p, w_p, h_p = output_coord[:,0], output_coord[:,1], output_coord[:,2], output_coord[:,3]
#                 fake_map = generate_fakemap(cx_p, cy_p, w_p, h_p,image_size=(256,256))
#                 fake_map = torch.tensor(fake_map).unsqueeze(1).to(image.device)  
#                 fg = self.LIG(fake_map)
#                 cls_token = image_embeds_c[:, 0, :]  
#                 patch_features = image_embeds_c[:, 1:, :]  
             
#                 enhanced_patches = patch_features + fg 

#                 enhanced_image_embeds = torch.cat([cls_token, enhanced_patches], dim=1) 
                
#                 cls_tokens_local_e = self.cls_token_local_e.expand(bs, -1, -1)

#                 text_attention_mask_clone_e = text.attention_mask.clone() # [:,1:] for ingoring class token
#                 local_feat_padding_mask_text_e = text_attention_mask_clone_e==0 # 0 = pad token

#                 local_feat_it_cross_attn_e = enhanced_image_embeds + self.it_cross_attn(query=self.norm_layer_it_cross_atten(enhanced_image_embeds), 
#                                                 key=self.norm_layer_it_cross_atten(text_embeds), 
#                                                 value=self.norm_layer_it_cross_atten(text_embeds),
#                                                 key_padding_mask=local_feat_padding_mask_text_e)[0]

#                 local_feat_aggr_e = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local_e), 
#                                                 key=self.norm_layer_aggr(local_feat_it_cross_attn_e[:,1:,:]), 
#                                                 value=self.norm_layer_aggr(local_feat_it_cross_attn_e[:,1:,:]))[0]
#                 enhanced_output_coord = self.bbox_head(local_feat_aggr_e.squeeze(1)).sigmoid()
#                 loss_bbox, loss_giou = self.get_bbox_loss(output_coord, fake_image_box)
#                 loss_bbox_e, loss_giou_e = self.get_bbox_loss(enhanced_output_coord, fake_image_box)
#                 total_loss_bbox = loss_bbox + loss_bbox_e
#                 total_loss_giou = loss_giou + loss_giou_e
#                 ##================= BIC ========================## 
#                 # forward the positve image-text pair
#                 output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
#                                                 attention_mask = text.attention_mask,
#                                                 encoder_hidden_states = image_embeds,
#                                                 encoder_attention_mask = image_atts,      
#                                                 return_dict = True,
#                                                 mode = 'fusion',
#                                             )            
                        

#                 itm_labels = torch.ones(bs, dtype=torch.long).to(image.device)
#                 itm_labels[real_label_pos] = 0 # fine-grained matching: only orig should be matched, 0 here means img-text matching
#                 vl_output = self.itm_head(output_pos.last_hidden_state[:,0,:])   
#                 loss_BIC = F.cross_entropy(vl_output, itm_labels) 
             
#                 ##================= MLC ========================## 
#                 cross_embeds_cls = local_feat_it_cross_attn
#                 cls_f = self.fch(cross_embeds_cls[:, 1:, :], cross_embeds_cls[:, 0, :])
#                 cls_t = self.cls_head(output_pos.last_hidden_state[:,0,:])

#                 cls = torch.concat((cls_f, cls_t), dim=1)
#                 loss_MLC = F.binary_cross_entropy_with_logits(cls, multicls_label.type(torch.float))


#                 ##================= TMG ========================##    
#                 token_label = text.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
#                 token_label[token_label==0] = -100 # -100 index = padding token
#                 token_label[token_label==1] = 0

#                 for batch_idx in range(len(fake_text_pos)):
#                     fake_pos_sample = fake_text_pos[batch_idx]
#                     if fake_pos_sample:
#                         for pos in fake_pos_sample:
#                             token_label[batch_idx, pos] = 1

#                 input_ids = text.input_ids.clone()

#                 if self.args.token_momentum:
#                     with torch.no_grad():
#                         logits_m = self.text_encoder_m(input_ids, 
#                                                     attention_mask = text.attention_mask,
#                                                     encoder_hidden_states = image_embeds_m,
#                                                     encoder_attention_mask = image_atts,      
#                                                     return_dict = True,
#                                                     return_logits = True,   
#                                                     )    
#                     token_cls_output = self.text_encoder(input_ids, 
#                                                 attention_mask = text.attention_mask,
#                                                 encoder_hidden_states = image_embeds,
#                                                 encoder_attention_mask = image_atts,      
#                                                 return_dict = True,
#                                                 labels = token_label,   
#                                                 soft_labels = F.softmax(logits_m.view(-1, 2),dim=-1),
#                                                 alpha = alpha
#                                                 )    
#                 else:
#                     token_cls_output  = self.text_encoder(input_ids, 
#                                                 attention_mask = text.attention_mask,
#                                                 encoder_hidden_states = image_embeds,
#                                                 encoder_attention_mask = image_atts,      
#                                                 return_dict = True,
#                                                 labels = token_label,   
#                                                 )  

#                 logits_tok = token_cls_output.logits 
#                 loss_TMG = F.cross_entropy(logits_tok.view(-1, logits_tok.size(-1)), token_label.view(-1), ignore_index=-100)

#                 return loss_CSA, loss_BIC, total_loss_bbox, total_loss_giou, loss_TMG, loss_MLC


#             else:
#                 image_embeds, image_embeds_c = self.visual_encoder(image) 
            
#                 image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

#                 text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
#                                                 return_dict = True, mode = 'text')            
#                 text_embeds = text_output.last_hidden_state

                
#                 ##================= IMG ========================## 
#                 bs = image.size(0)
#                 cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)

#                 text_attention_mask_clone = text.attention_mask.clone() # [:,1:] for ingoring class token
#                 local_feat_padding_mask_text = text_attention_mask_clone==0 # 0 = pad token
#                 local_feat_it_cross_attn = image_embeds_c + self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds_c), 
#                                                 key=self.norm_layer_it_cross_atten(text_embeds), 
#                                                 value=self.norm_layer_it_cross_atten(text_embeds),
#                                                 key_padding_mask=local_feat_padding_mask_text)[0]

#                 local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
#                                                 key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
#                                                 value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
#                 output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
             

#                 cx_p, cy_p, w_p, h_p = output_coord[:,0], output_coord[:,1], output_coord[:,2], output_coord[:,3]

#                 fake_map = generate_fakemap(cx_p, cy_p, w_p, h_p,image_size=(256,256))

#                 fake_map = torch.tensor(fake_map).unsqueeze(1).to(image.device) 

#                 fg = self.LIG(fake_map)
#                 cls_token = image_embeds_c[:, 0, :]  
#                 patch_features = image_embeds_c[:, 1:, :]  
              
#                 enhanced_patches = patch_features + fg

#                 enhanced_image_embeds = torch.cat([cls_token, enhanced_patches], dim=1)  

#                 cls_tokens_local_e = self.cls_token_local_e.expand(bs, -1, -1)

#                 text_attention_mask_clone_e = text.attention_mask.clone() # [:,1:] for ingoring class token
#                 local_feat_padding_mask_text_e = text_attention_mask_clone_e==0 # 0 = pad token

#                 local_feat_it_cross_attn_e = enhanced_image_embeds + self.it_cross_attn(query=self.norm_layer_it_cross_atten(enhanced_image_embeds), 
#                                                 key=self.norm_layer_it_cross_atten(text_embeds), 
#                                                 value=self.norm_layer_it_cross_atten(text_embeds),
#                                                 key_padding_mask=local_feat_padding_mask_text_e)[0]

#                 local_feat_aggr_e = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local_e), 
#                                                 key=self.norm_layer_aggr(local_feat_it_cross_attn_e[:,1:,:]), 
#                                                 value=self.norm_layer_aggr(local_feat_it_cross_attn_e[:,1:,:]))[0]
#                 enhanced_output_coord = self.bbox_head(local_feat_aggr_e.squeeze(1)).sigmoid()
#                 # forward the positve image-text pair
#                 output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
#                                                 attention_mask = text.attention_mask,
#                                                 encoder_hidden_states = image_embeds,
#                                                 encoder_attention_mask = image_atts,      
#                                                 return_dict = True,
#                                                 mode = 'fusion',
#                                             )               
#                 ##================= BIC ========================## 
#                 logits_real_fake = self.itm_head(output_pos.last_hidden_state[:,0,:])
#                 ##================= IMG_e ========================## 
               
                
#                 # ##================= EMLC ========================## 
#                 cross_embeds_cls = local_feat_it_cross_attn
#                 cls_f = self.fch(cross_embeds_cls[:, 1:, :], cross_embeds_cls[:, 0, :])
#                 cls_t = self.cls_head(output_pos.last_hidden_state[:,0,:])

#                 logits_multicls = torch.concat((cls_f, cls_t), dim=1)
#                 ##================= TMG ========================##   
#                 input_ids = text.input_ids.clone()
#                 logits_tok = self.text_encoder(input_ids, 
#                                             attention_mask = text.attention_mask,
#                                             encoder_hidden_states = image_embeds,
#                                             encoder_attention_mask = image_atts,      
#                                             return_dict = True,
#                                             return_logits = True,   
#                                             )     
#                 return logits_real_fake, logits_multicls, enhanced_output_coord, logits_tok
        
  

#     @torch.no_grad()    
#     def copy_params(self):
#         for model_pair in self.model_pairs:           
#             for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
#                 param_m.data.copy_(param.data)  # initialize
#                 param_m.requires_grad = False  # not update by gradient    

            
#     @torch.no_grad()        
#     def _momentum_update(self):
#         for model_pair in self.model_pairs:           
#             for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
#                 param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
            
            
#     @torch.no_grad()
#     def _dequeue_and_enqueue(self, image_feat, text_feat):
#         # gather keys before updating queue
#         image_feats = concat_all_gather(image_feat)
#         text_feats = concat_all_gather(text_feat)

#         batch_size = image_feats.shape[0]

#         ptr = int(self.queue_ptr)
#         # print(f"queue_size: {self.queue_size}, batch_size: {batch_size}")
#         # assert self.queue_size % batch_size == 0  # for simplicity

#         # replace the keys at ptr (dequeue and enqueue)
#         self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
#         self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
#         ptr = (ptr + batch_size) % self.queue_size  # move pointer

#         self.queue_ptr[0] = ptr 
        
        
# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output

"""
CALS: Contrastive Alignment and Localization System
A multimodal deep learning framework for image-text understanding with enhanced localization capabilities.

This module implements the CALS architecture which combines:
- Multi-modal contrastive learning (MAC)
- Image-text matching with binary classification (BIC) 
- Token-level matching guidance (TMG)
- Multi-label classification (MLC)
- Iterative multi-modal grounding (IMG)

"""

from functools import partial
from typing import Optional, Tuple, Dict, Any, List
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.vit import CAFormer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertForTokenClassification
from models import box_ops
from tools.multilabel_metrics import get_multi_label
from timm.models.layers import trunc_normal_


class ContrastiveLoss:
    """Advanced contrastive loss functions for multimodal learning."""
    
    @staticmethod
    def kl_divergence(p: Tensor, q: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Compute KL divergence between two probability distributions.
        
        Args:
            p: Source probability distribution
            q: Target probability distribution  
            epsilon: Small constant to prevent numerical instability
            
        Returns:
            KL divergence value
        """
        q = q + epsilon
        kl_div = p * torch.log(p / q)
        return kl_div.sum()
    
    @staticmethod
    def image_to_text_loss(visual_features: Tensor, text_features: Tensor) -> Tensor:
        """Compute image-to-text contrastive loss."""
        p = F.softmax(visual_features, dim=-1)
        q = F.softmax(text_features, dim=-1)
        return ContrastiveLoss.kl_divergence(p, q)
    
    @staticmethod
    def text_to_image_loss(visual_features: Tensor, text_features: Tensor) -> Tensor:
        """Compute text-to-image contrastive loss."""
        p = F.softmax(text_features, dim=-1)
        q = F.softmax(visual_features, dim=-1)
        return ContrastiveLoss.kl_divergence(p, q)
    
    @staticmethod
    def bidirectional_loss(visual_features: Tensor, text_features: Tensor) -> Tensor:
        """Compute bidirectional contrastive loss."""
        i2t_loss = ContrastiveLoss.image_to_text_loss(visual_features, text_features)
        t2i_loss = ContrastiveLoss.text_to_image_loss(visual_features, text_features)
        return i2t_loss + t2i_loss


def generate_spatial_attention_map(
    center_x: Tensor, 
    center_y: Tensor, 
    width: Tensor, 
    height: Tensor, 
    image_size: Tuple[int, int] = (256, 256)
) -> Tensor:
    """
    Generate Gaussian-based spatial attention maps for localization.
    
    Args:
        center_x: Normalized center x coordinates [0, 1]
        center_y: Normalized center y coordinates [0, 1] 
        width: Normalized box widths [0, 1]
        height: Normalized box heights [0, 1]
        image_size: Target image dimensions (H, W)
        
    Returns:
        Spatial attention maps of shape (batch_size, 1, H, W)
    """
    if isinstance(image_size, int):
        H, W = image_size, image_size
    else:
        H, W = image_size
        
    device = center_x.device
    original_size = (256, 256)
    W_orig, H_orig = original_size
    
    # Convert normalized coordinates to pixel coordinates
    cx_pixel = center_x * W_orig
    cy_pixel = center_y * H_orig
    w_pixel = width * W_orig
    h_pixel = height * H_orig
    
    # Compute Gaussian standard deviation based on box dimensions
    sigma = torch.sqrt((h_pixel / 2) ** 2 + (w_pixel / 2) ** 2)
    
    # Create coordinate grids
    x = torch.arange(image_size[0], device=device, dtype=torch.float32)
    y = torch.arange(image_size[1], device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Expand dimensions for batch processing
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    cx_pixel = cx_pixel.view(-1, 1, 1)
    cy_pixel = cy_pixel.view(-1, 1, 1)
    sigma = sigma.view(-1, 1, 1)
    
    # Generate Gaussian attention maps
    distance_squared = (xx - cx_pixel) ** 2 + (yy - cy_pixel) ** 2
    attention_map = torch.exp(-distance_squared / (2 * (sigma ** 2) + 1e-8))
    
    return torch.clamp(attention_map, 0, 1)


class MultiClassAttentionHead(nn.Module):
    """
    Multi-class attention head with spatial and global feature fusion.
    
    This module combines global classification with spatial attention mechanisms
    to improve fine-grained classification performance.
    """
    
    def __init__(self, num_classes: int, feature_dim: int, lambda_init: float = 1.0):
        """
        Initialize the multi-class attention head.
        
        Args:
            num_classes: Number of classification classes
            feature_dim: Dimension of input features
            lambda_init: Initial weight for attention fusion
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Spatial attention convolutions for each class
        self.spatial_attention = nn.ModuleList([
            nn.Conv2d(feature_dim, 1, kernel_size=1) 
            for _ in range(num_classes)
        ])
        
        # Global classification head
        self.global_classifier = nn.Linear(feature_dim, num_classes)
        
        # Learnable fusion weight
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
    
    def forward(self, patch_tokens: Tensor, class_token: Tensor) -> Tensor:
        """
        Forward pass of the attention head.
        
        Args:
            patch_tokens: Spatial feature tokens (B, seq_len, D)
            class_token: Global class token (B, D)
            
        Returns:
            Combined classification scores (B, num_classes)
        """
        batch_size, seq_len, feature_dim = patch_tokens.shape
        spatial_size = int(seq_len ** 0.5)
        
        # Reshape to spatial format
        spatial_features = patch_tokens.view(
            batch_size, spatial_size, spatial_size, feature_dim
        ).permute(0, 3, 1, 2)
        
        # Global classification
        global_scores = self.global_classifier(class_token)
        
        # Spatial attention classification
        attention_scores = []
        for class_idx in range(self.num_classes):
            # Generate attention map
            attention_map = torch.sigmoid(
                self.spatial_attention[class_idx](spatial_features)
            )
            
            # Apply attention weighting
            weighted_features = attention_map * spatial_features
            
            # Global average pooling
            pooled_features = torch.mean(weighted_features, dim=(2, 3))
            class_score = torch.mean(pooled_features, dim=1)
            attention_scores.append(class_score)
        
        attention_scores = torch.stack(attention_scores, dim=1)
        
        # Combine global and attention-based scores
        final_scores = global_scores + self.lambda_param * attention_scores
        return final_scores


class LocalizationGuidanceNetwork(nn.Module):
    """
    Localization-based Image Guidance (LIG) Network.
    
    Transforms spatial attention maps into feature representations
    for enhanced image understanding.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (256, 256), output_dim: int = 768):
        """
        Initialize the LIG network.
        
        Args:
            input_size: Input spatial dimensions
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Second conv block  
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Adaptive pooling
            nn.AdaptiveAvgPool2d((16, 16)),
            
            # Final projection
            nn.Conv2d(256, output_dim, kernel_size=1),
        )
    
    def forward(self, attention_map: Tensor) -> Tensor:
        """
        Forward pass of the LIG network.
        
        Args:
            attention_map: Spatial attention map (B, 1, H, W)
            
        Returns:
            Feature representation (B, seq_len, output_dim)
        """
        features = self.feature_extractor(attention_map)
        # Reshape to sequence format
        features = features.flatten(2).permute(0, 2, 1)
        return features


class CALS(nn.Module):
    """
    Contrastive Alignment and Localization System (CALS).
    
    A comprehensive multimodal framework that integrates contrastive learning,
    localization, and fine-grained understanding for image-text tasks.
    
    Key Components:
    - Multi-modal Alignment via Contrastive learning (MAC)
    - Binary Image-text Correspondence (BIC) 
    - Token-level Matching Guidance (TMG)
    - Multi-Label Classification (MLC)
    - Iterative Multi-modal Grounding (IMG)
    """
    
    def __init__(
        self,
        args: Any = None,
        config: Dict[str, Any] = None,
        text_encoder: Optional[str] = None,
        tokenizer: Any = None,
        init_deit: bool = True
    ):
        """
        Initialize the CALS model.
        
        Args:
            args: Training arguments
            config: Model configuration dictionary
            text_encoder: Pre-trained text encoder path
            tokenizer: Text tokenizer
            init_deit: Whether to initialize with DeiT weights
        """
        super().__init__()
        
        self.args = args
        self.tokenizer = tokenizer
        self.batch_count = 0
        
        # Model configuration
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        
        # Initialize visual encoder
        self._init_visual_encoder(config, init_deit)
        
        # Initialize text encoder
        self._init_text_encoder(config, text_encoder)
        
        # Feature projection layers
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        
        # Contrastive learning parameters
        self.temperature = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        
        # Task-specific heads
        self._init_task_heads(text_width, embed_dim)
        
        # Momentum models for contrastive learning
        self._init_momentum_models(config, text_encoder)
        
        # Cross-attention and aggregation modules
        self._init_attention_modules(text_width)
        
        # Initialize memory queue
        self._init_memory_queue(embed_dim)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_visual_encoder(self, config: Dict[str, Any], init_deit: bool):
        """Initialize the visual encoder with optional DeiT pretraining."""
        self.visual_encoder = CAFormer(
            img_size=config['image_res'],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        if init_deit:
            self._load_deit_weights()
    
    def _load_deit_weights(self):
        """Load pre-trained DeiT weights."""
        try:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu",
                check_hash=True
            )
            state_dict = checkpoint["model"]
            
            # Interpolate positional embeddings
            pos_embed_reshaped = interpolate_pos_embed(
                state_dict['pos_embed'], self.visual_encoder
            )
            state_dict['pos_embed'] = pos_embed_reshaped
            
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(f"DeiT weights loaded: {msg}")
        except Exception as e:
            print(f"Warning: Failed to load DeiT weights: {e}")
    
    def _init_text_encoder(self, config: Dict[str, Any], text_encoder: str):
        """Initialize the text encoder."""
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForTokenClassification.from_pretrained(
            text_encoder,
            config=bert_config,
            label_smoothing=config['label_smoothing']
        )
    
    def _init_task_heads(self, text_width: int, embed_dim: int):
        """Initialize task-specific prediction heads."""
        # Image-text matching head
        self.itm_head = self._build_mlp(text_width, 2)
        
        # Bounding box regression head
        self.bbox_head = self._build_mlp(text_width, 4)
        
        # Classification heads
        self.cls_head = self._build_mlp(text_width, 2)
        self.face_cls_head = MultiClassAttentionHead(
            num_classes=2,
            feature_dim=text_width,
            lambda_init=0.3
        )
        
        # Localization guidance network
        self.localization_guidance = LocalizationGuidanceNetwork(
            input_size=(256, 256),
            output_dim=embed_dim
        )
    
    def _init_momentum_models(self, config: Dict[str, Any], text_encoder: str):
        """Initialize momentum models for contrastive learning."""
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        
        # Momentum visual encoder
        self.visual_encoder_m = CAFormer(
            img_size=config['image_res'],
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        # Momentum projections
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        
        # Momentum text encoder
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder_m = BertForTokenClassification.from_pretrained(
            text_encoder,
            config=bert_config,
            label_smoothing=config['label_smoothing']
        )
        
        text_width = self.text_encoder_m.config.hidden_size
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        # Model pairs for momentum update
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        
        self.copy_params()
    
    def _init_attention_modules(self, text_width: int):
        """Initialize cross-attention and aggregation modules."""
        # Normalization layers
        self.norm_layer_aggr = nn.LayerNorm(text_width)
        self.norm_layer_it_cross_atten = nn.LayerNorm(text_width)
        
        # Learnable tokens
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.cls_token_local_e = nn.Parameter(torch.zeros(1, 1, text_width))
        
        # Multi-head attention modules
        self.aggregator = nn.MultiheadAttention(
            text_width, num_heads=12, dropout=0.0, batch_first=True
        )
        self.it_cross_attn = nn.MultiheadAttention(
            text_width, num_heads=12, dropout=0.0, batch_first=True
        )
        
        # Initialize learnable tokens
        trunc_normal_(self.cls_token_local, std=0.02)
        trunc_normal_(self.cls_token_local_e, std=0.02)
    
    def _init_memory_queue(self, embed_dim: int):
        """Initialize memory queue for contrastive learning."""
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Normalize queue
        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
    
    def _build_mlp(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build a multi-layer perceptron with normalization and activation."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )
    
    def _init_weights(self, module: nn.Module):
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def compute_bbox_loss(
        self,
        predicted_coords: Tensor,
        target_coords: Tensor,
        is_image: Optional[Tensor] = None,
        target_bbox_map_ids: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute bounding box regression loss using L1 and GIoU.
        
        Args:
            predicted_coords: Predicted bounding box coordinates
            target_coords: Ground truth coordinates
            is_image: Mask for image samples
            target_bbox_map_ids: Mapping indices for targets
            
        Returns:
            Tuple of (L1 loss, GIoU loss)
        """
        n_predictions = predicted_coords.size(0)
        n_targets = target_coords.size(0)
        
        assert n_predictions == n_targets, "Prediction and target batch sizes must match"
        
        # L1 loss
        l1_loss = F.l1_loss(predicted_coords, target_coords, reduction='none')
        
        # Convert to xyxy format for GIoU
        pred_boxes = box_ops.box_cxcywh_to_xyxy(predicted_coords)
        target_boxes = box_ops.box_cxcywh_to_xyxy(target_coords)
        
        # Check for degenerate boxes
        pred_degenerate = (pred_boxes[:, 2:] < pred_boxes[:, :2]).any()
        target_degenerate = (target_boxes[:, 2:] < target_boxes[:, :2]).any()
        
        if pred_degenerate or target_degenerate:
            print("Warning: Degenerate boxes detected")
            giou_loss = torch.zeros(predicted_coords.size(0), device=predicted_coords.device)
        else:
            giou_loss = 1 - torch.diag(box_ops.generalized_box_iou(pred_boxes, target_boxes))
        
        # Apply masking if provided
        if is_image is None:
            num_boxes = target_coords.size(0)
        else:
            num_boxes = torch.sum(1 - is_image)
            l1_loss = l1_loss * (1 - is_image.view(-1, 1))
            giou_loss = giou_loss * (1 - is_image)
        
        return l1_loss.sum() / num_boxes, giou_loss.sum() / num_boxes
    
    def forward(
        self,
        image: Tensor,
        label: Tensor,
        text: Any,
        fake_image_box: Tensor,
        fake_text_pos: List[List[int]],
        epoch: int,
        alpha: float = 0,
        is_train: bool = True
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass of the CALS model.
        
        Args:
            image: Input images
            label: Ground truth labels
            text: Tokenized text input
            fake_image_box: Fake bounding box coordinates
            fake_text_pos: Fake text token positions
            epoch: Current training epoch
            alpha: Interpolation factor for momentum updates
            is_train: Training mode flag
            
        Returns:
            Loss values during training, predictions during inference
        """
        if is_train:
            return self._forward_train(
                image, label, text, fake_image_box, fake_text_pos, epoch, alpha
            )
        else:
            return self._forward_inference(image, text)
    
    def _forward_train(
        self,
        image: Tensor,
        label: Tensor,
        text: Any,
        fake_image_box: Tensor,
        fake_text_pos: List[List[int]],
        epoch: int,
        alpha: float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Training forward pass."""
        with torch.no_grad():
            self.temperature.clamp_(0.001, 0.5)
        
        # Convert to multi-label format
        multicls_label, real_label_pos = get_multi_label(label, image)
        batch_size = image.size(0)
        
        # Extract visual features
        image_embeds, image_embeds_c = self.visual_encoder(image)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image.device
        )
        
        # Extract text features
        text_output = self.text_encoder.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text'
        )
        text_embeds = text_output.last_hidden_state
        
        # Project features for contrastive learning
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        
        # Compute contrastive losses
        loss_csa = self._compute_contrastive_loss(
            image_feat, text_feat, real_label_pos, alpha
        )
        
        # Compute localization and classification losses
        loss_bic, total_loss_bbox, total_loss_giou, loss_mlc = self._compute_grounding_losses(
            image_embeds, image_embeds_c, text_embeds, text, image_atts,
            fake_image_box, multicls_label, batch_size
        )
        
        # Compute token matching loss
        loss_tmg = self._compute_token_matching_loss(
            text, fake_text_pos, image_embeds, image_atts, alpha
        )
        
        return loss_csa, loss_bic, total_loss_bbox, total_loss_giou, loss_tmg, loss_mlc
    
    def _forward_inference(self, image: Tensor, text: Any) -> Tuple[Tensor, ...]:
        """Inference forward pass."""
        batch_size = image.size(0)
        
        # Extract features
        image_embeds, image_embeds_c = self.visual_encoder(image)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image.device
        )
        
        text_output = self.text_encoder.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text'
        )
        text_embeds = text_output.last_hidden_state
        
        # Perform localization and generate enhanced features
        output_coord, enhanced_output_coord = self._perform_localization(
            image_embeds_c, text_embeds, text.attention_mask, batch_size
        )
        
        # Generate predictions
        logits_real_fake, logits_multicls, logits_tok = self._generate_predictions(
            image_embeds, text_embeds, text, image_atts
        )
        
        return logits_real_fake, logits_multicls, enhanced_output_coord, logits_tok
    
    def _compute_contrastive_loss(
        self,
        image_feat: Tensor,
        text_feat: Tensor,
        real_label_pos: Tensor,
        alpha: float
    ) -> Tensor:
        """Compute multi-modal contrastive alignment loss."""
        # Get momentum features
        with torch.no_grad():
            self._momentum_update()
            
            image_embeds_m, _ = self.visual_encoder_m(image_feat.unsqueeze(0))
            image_feat_m = F.normalize(
                self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1
            )
            
            # Combine with queue
            image_feat_all = torch.cat([
                image_feat_m.t(), self.image_queue.clone().detach()
            ], dim=1)
            
            text_feat_all = torch.cat([
                text_feat.t(), self.text_queue.clone().detach()
            ], dim=1)
            
            # Compute similarity matrices
            sim_i2t_m = image_feat_m @ text_feat_all / self.temperature
            sim_t2i_m = text_feat @ image_feat_all / self.temperature
            
            # Create target distributions
            sim_targets = torch.zeros_like(sim_i2t_m)
            sim_targets[real_label_pos, real_label_pos] = 1
            
            sim_targets_diag = torch.zeros_like(sim_i2t_m)
            sim_targets_diag.fill_diagonal_(1)
            
            # Soft targets with momentum
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
        
        # Compute current similarities
        sim_i2t = image_feat @ text_feat_all / self.temperature
        sim_t2i = text_feat @ image_feat_all / self.temperature
        sim_i2i = image_feat @ image_feat_all / self.temperature
        sim_t2t = text_feat @ text_feat_all / self.temperature
        
        # Compute losses
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_targets_diag, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_targets_diag, dim=1).mean()
        
        loss_mac = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
        loss_other = ContrastiveLoss.bidirectional_loss(image_feat, text_feat)
        
        # Update queue
        self._dequeue_and_enqueue(image_feat, text_feat)
        
        return loss_mac + loss_other
    
    @torch.no_grad()
    def copy_params(self):
        """Copy parameters to momentum models."""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum model parameters."""
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
    
    def _compute_grounding_losses(
        self,
        image_embeds: Tensor,
        image_embeds_c: Tensor,
        text_embeds: Tensor,
        text: Any,
        image_atts: Tensor,
        fake_image_box: Tensor,
        multicls_label: Tensor,
        batch_size: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute localization and classification losses."""
        
        # Perform iterative multi-modal grounding
        output_coord, enhanced_output_coord = self._perform_localization(
            image_embeds_c, text_embeds, text.attention_mask, batch_size
        )
        
        # Compute bounding box losses
        loss_bbox, loss_giou = self.compute_bbox_loss(output_coord, fake_image_box)
        loss_bbox_e, loss_giou_e = self.compute_bbox_loss(enhanced_output_coord, fake_image_box)
        
        total_loss_bbox = loss_bbox + loss_bbox_e
        total_loss_giou = loss_giou + loss_giou_e
        
        # Binary image-text correspondence
        output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion'
        )
        
        # Create labels (assuming all positive pairs in training)
        itm_labels = torch.ones(batch_size, dtype=torch.long, device=image_embeds.device)
        vl_output = self.itm_head(output_pos.last_hidden_state[:, 0, :])
        loss_bic = F.cross_entropy(vl_output, itm_labels)
        
        # Multi-label classification
        loss_mlc = self._compute_multilabel_loss(
            image_embeds_c, text_embeds, output_pos, multicls_label
        )
        
        return loss_bic, total_loss_bbox, total_loss_giou, loss_mlc
    
    def _perform_localization(
        self,
        image_embeds_c: Tensor,
        text_embeds: Tensor,
        attention_mask: Tensor,
        batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        """Perform iterative multi-modal grounding."""
        
        # Initial localization
        cls_tokens_local = self.cls_token_local.expand(batch_size, -1, -1)
        
        # Cross-attention between image and text
        local_feat_it_cross_attn = self._apply_cross_attention(
            image_embeds_c, text_embeds, attention_mask
        )
        
        # Aggregate features for localization
        local_feat_aggr = self.aggregator(
            query=self.norm_layer_aggr(cls_tokens_local),
            key=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :]),
            value=self.norm_layer_aggr(local_feat_it_cross_attn[:, 1:, :])
        )[0]
        
        # Predict bounding box coordinates
        output_coord = self.bbox_head(local_feat_aggr.squeeze(1)).sigmoid()
        
        # Generate spatial attention map
        cx_p, cy_p, w_p, h_p = output_coord[:, 0], output_coord[:, 1], output_coord[:, 2], output_coord[:, 3]
        attention_map = generate_spatial_attention_map(cx_p, cy_p, w_p, h_p, image_size=(256, 256))
        attention_map = attention_map.unsqueeze(1).to(image_embeds_c.device)
        
        # Apply localization guidance
        localization_features = self.localization_guidance(attention_map)
        
        # Enhance image features
        cls_token = image_embeds_c[:, 0, :]
        patch_features = image_embeds_c[:, 1:, :]
        enhanced_patches = patch_features + localization_features
        enhanced_image_embeds = torch.cat([cls_token.unsqueeze(1), enhanced_patches], dim=1)
        
        # Enhanced localization
        cls_tokens_local_e = self.cls_token_local_e.expand(batch_size, -1, -1)
        
        local_feat_it_cross_attn_e = self._apply_cross_attention(
            enhanced_image_embeds, text_embeds, attention_mask
        )
        
        local_feat_aggr_e = self.aggregator(
            query=self.norm_layer_aggr(cls_tokens_local_e),
            key=self.norm_layer_aggr(local_feat_it_cross_attn_e[:, 1:, :]),
            value=self.norm_layer_aggr(local_feat_it_cross_attn_e[:, 1:, :])
        )[0]
        
        enhanced_output_coord = self.bbox_head(local_feat_aggr_e.squeeze(1)).sigmoid()
        
        return output_coord, enhanced_output_coord
    
    def _apply_cross_attention(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """Apply cross-attention between image and text features."""
        attention_mask_clone = attention_mask.clone()
        padding_mask = attention_mask_clone == 0
        
        cross_attn_output = self.it_cross_attn(
            query=self.norm_layer_it_cross_atten(image_embeds),
            key=self.norm_layer_it_cross_atten(text_embeds),
            value=self.norm_layer_it_cross_atten(text_embeds),
            key_padding_mask=padding_mask
        )[0]
        
        return image_embeds + cross_attn_output
    
    def _compute_multilabel_loss(
        self,
        image_embeds_c: Tensor,
        text_embeds: Tensor,
        output_pos: Any,
        multicls_label: Tensor
    ) -> Tensor:
        """Compute multi-label classification loss."""
        
        # Face classification using attention head
        cls_f = self.face_cls_head(
            image_embeds_c[:, 1:, :],  # patch tokens
            image_embeds_c[:, 0, :]    # class token
        )
        
        # Text-based classification
        cls_t = self.cls_head(output_pos.last_hidden_state[:, 0, :])
        
        # Combine predictions
        combined_cls = torch.cat([cls_f, cls_t], dim=1)
        
        # Binary cross-entropy loss
        loss_mlc = F.binary_cross_entropy_with_logits(
            combined_cls, multicls_label.float()
        )
        
        return loss_mlc
    
    def _compute_token_matching_loss(
        self,
        text: Any,
        fake_text_pos: List[List[int]],
        image_embeds: Tensor,
        image_atts: Tensor,
        alpha: float
    ) -> Tensor:
        """Compute token-level matching guidance loss."""
        
        # Prepare token labels
        token_label = text.attention_mask[:, 1:].clone()  # Ignore class token
        token_label[token_label == 0] = -100  # Padding tokens
        token_label[token_label == 1] = 0     # Normal tokens
        
        # Mark fake positions
        for batch_idx, fake_positions in enumerate(fake_text_pos):
            if fake_positions:
                for pos in fake_positions:
                    if pos < token_label.size(1):
                        token_label[batch_idx, pos] = 1
        
        input_ids = text.input_ids.clone()
        
        # Use momentum-based soft labels if enabled
        if hasattr(self.args, 'token_momentum') and self.args.token_momentum:
            with torch.no_grad():
                logits_m = self.text_encoder_m(
                    input_ids,
                    attention_mask=text.attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    return_logits=True
                )
            
            token_cls_output = self.text_encoder(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                labels=token_label,
                soft_labels=F.softmax(logits_m.view(-1, 2), dim=-1),
                alpha=alpha
            )
        else:
            token_cls_output = self.text_encoder(
                input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                labels=token_label
            )
        
        logits_tok = token_cls_output.logits
        loss_tmg = F.cross_entropy(
            logits_tok.view(-1, logits_tok.size(-1)),
            token_label.view(-1),
            ignore_index=-100
        )
        
        return loss_tmg
    
    def _generate_predictions(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
        text: Any,
        image_atts: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate predictions during inference."""
        
        # Image-text matching prediction
        output_pos = self.text_encoder.bert(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode='fusion'
        )
        
        logits_real_fake = self.itm_head(output_pos.last_hidden_state[:, 0, :])
        
        # Multi-label classification
        cross_embeds_cls = image_embeds  # Simplified for inference
        cls_f = self.face_cls_head(
            cross_embeds_cls[:, 1:, :],
            cross_embeds_cls[:, 0, :]
        )
        cls_t = self.cls_head(output_pos.last_hidden_state[:, 0, :])
        logits_multicls = torch.cat([cls_f, cls_t], dim=1)
        
        # Token-level prediction
        input_ids = text.input_ids.clone()
        logits_tok = self.text_encoder(
            input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            return_logits=True
        )
        
        return logits_real_fake, logits_multicls, logits_tok
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat: Tensor, text_feat: Tensor):
        """Update memory queue with new features."""
        # Gather features from all processes (for distributed training)
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        
        # Update queue
        if ptr + batch_size <= self.queue_size:
            self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        else:
            # Handle wraparound
            remaining = self.queue_size - ptr
            self.image_queue[:, ptr:] = image_feats[:remaining].T
            self.text_queue[:, ptr:] = text_feats[:remaining].T
            
            if batch_size > remaining:
                overflow = batch_size - remaining
                self.image_queue[:, :overflow] = image_feats[remaining:].T
                self.text_queue[:, :overflow] = text_feats[remaining:].T
        
        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor: Tensor) -> Tensor:
    """
    Perform all_gather operation on the provided tensors for distributed training.
    
    Args:
        tensor: Input tensor to gather
        
    Returns:
        Concatenated tensor from all processes
        
    Warning:
        torch.distributed.all_gather has no gradient.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather, dim=0)
    return output


# Model factory functions
def create_cals_model(
    config_path: str,
    text_encoder_path: str,
    tokenizer: Any,
    **kwargs
) -> CALS:
    """
    Factory function to create a CALS model with specified configuration.
    
    Args:
        config_path: Path to model configuration file
        text_encoder_path: Path to pre-trained text encoder
        tokenizer: Text tokenizer
        **kwargs: Additional arguments
        
    Returns:
        Initialized CALS model
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = CALS(
        config=config,
        text_encoder=text_encoder_path,
        tokenizer=tokenizer,
        **kwargs
    )
    
    return model


def load_pretrained_cals(
    checkpoint_path: str,
    config_path: str,
    text_encoder_path: str,
    tokenizer: Any,
    **kwargs
) -> CALS:
    """
    Load a pre-trained CALS model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model configuration
        text_encoder_path: Path to text encoder
        tokenizer: Text tokenizer
        **kwargs: Additional arguments
        
    Returns:
        Loaded CALS model
    """
    model = create_cals_model(config_path, text_encoder_path, tokenizer, **kwargs)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print(f"Loaded CALS model from {checkpoint_path}")
    return model
    