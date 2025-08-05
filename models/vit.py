import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath





def spatial_apply(x, module, batch_size, keep_cls=True):
    """通用维度转换函数"""
    if keep_cls:
        cls_token = x[:, :1]  # 保存CLS Token
        patches = x[:, 1:]
    else:
        patches = x
        
    # 转换为2D特征图
    h = w = int(patches.shape[1]**0.5)
    feat_map = patches.permute(0,2,1).view(batch_size, -1, h, w)
    
    # 应用USAM
    feat_map = module(feat_map)
    
    # 转换回序列
    patches = feat_map.flatten(2).permute(0,2,1)
    return torch.cat([cls_token, patches], dim=1) if keep_cls else patches

class USAMAdapter(nn.Module):
    """适配器模块（处理维度转换）"""
    def __init__(self, dim):
        super().__init__()
        self.usam = USAM(kernel_size=3)
        
    def forward(self, x, batch_size):
        return spatial_apply(x, self.usam, batch_size)


# 修改后的USAM类
class USAM(nn.Module):
    def __init__(self, kernel_size=3, padding=1, polish=True):
        super(USAM, self).__init__()
        kernel = torch.ones((kernel_size, kernel_size))
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
        kernel2 = torch.ones((1, 1)) * (kernel_size * kernel_size)
        kernel2 = kernel2.unsqueeze(0).unsqueeze(0)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

        self.polish = polish
        self.pad = padding
        self.bn = nn.BatchNorm2d(1)

    def __call__(self, x):
        fmap = x.sum(1, keepdim=True)      
        x1 = F.conv2d(fmap, self.weight, padding=self.pad)
        x2 = F.conv2d(fmap, self.weight2, padding=0) 
        
        att = x2 - x1
        att = self.bn(att)
        att = F.relu(att)

        if self.polish:
            # 创建边界mask
            B, C, H, W = att.shape
            mask = torch.ones((B, C, H, W), device=att.device)
            mask[:, :, :, 0] = 0      # 左边界
            mask[:, :, :, -1] = 0     # 右边界
            mask[:, :, 0, :] = 0      # 上边界
            mask[:, :, -1, :] = 0     # 下边界
            att = att * mask

        output = x + att * x
        return output
    
class MLP(nn.Module):
    def __init__(self, input_dim=258*258, hidden_dim=512, output_dim=768):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)  # 输入形状 [B, 257, 258*258]，输出 [B, 257, 768]
class CoordConv(nn.Module):
    """
    CoordConv Layer as described in the paper.
    Adds two coordinate channels to the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(CoordConv, self).__init__()
        self.coord_conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor after applying CoordConv
        """
        B, C, H, W = x.size()

        # Create coordinate tensors
        device = x.device
        dtype = x.dtype

        # Create normalized coordinates ranging from -1 to 1
        y_coords = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)

        # Concatenate coordinate channels with input
        x = torch.cat([x, x_coords, y_coords], dim=1)  # Shape: (B, C+2, H, W)

        # Apply convolution
        out = self.coord_conv(x)
        return out
class ConvBlock(nn.Module):
    """
    Conv Block containing:
    - 3x3 CoordConv layer
    - 5x5 Standard Conv layer
    Each convolution is followed by BatchNorm and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        # 5x5 Standard Conv Layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor after Conv Block
        """     
        out = self.conv(x)
        out = self.bn2(out)
        out = self.relu2(out)
        
        return out


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionBlock, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.mha(x, x, x)
        x = self.layer_norm1(x + self.dropout1(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout2(ffn_output))
        return x



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
class CAFormer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.CoordConv = CoordConv(in_chans,embed_dim)
        # self.Conv = ConvBlock(embed_dim,embed_dim)
        self.Conv = ConvBlock(in_chans,embed_dim)
        self.usam_pre = USAM(kernel_size=3, polish=True)

        self.MHA1 = MultiHeadAttentionBlock(embed_dim,num_heads=8)
        self.MHA2 = MultiHeadAttentionBlock(embed_dim,num_heads=8)
        self.MHA3 = MultiHeadAttentionBlock(embed_dim,num_heads=8)

        self.fl1_mlp = MLP(input_dim=258*258, hidden_dim=512, output_dim=768)
        self.fl2_mlp = MLP(input_dim=258*258, hidden_dim=512, output_dim=768)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, register_blk=-1):
        B = x.shape[0]
        image = x
        x = self.patch_embed(x)#[8,256,768]
        # x = spatial_apply(x, self.usam_pre, B, keep_cls=False)

        # print('x.shape:',x.shape)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # print('x.shape:',x.shape)
        
        x = x + self.pos_embed[:,:x.size(1),:]
        # print('x.shape:',x.shape)

        x = self.pos_drop(x)
        # print('x.shape:',x.shape)

        for blk in self.blocks[0:8]:
            x = blk(x)
      

        Fg2 = self.MHA2(x)
        Fl2 = self.Conv(image)
        # Fg2 = self.MHA2(Fu1)
        # Fl2 = self.Conv(Fl1)
        Fu2 = sum_feature(Fg2,Fl2,B)
        for blk in self.blocks[8:12]:#加一下深度
            Fu2 = blk(Fu2)
            x = blk(x)
        # x = self.MHA3(Fu2)
        fu = self.norm(Fu2)
        x = self.norm(x)
        
        return fu, x

    # def forward(self, x, register_blk=-1):
    #     B = x.shape[0]
    #     image = x
    #     x = self.patch_embed(x)#[8,256,768]
    #     # x = spatial_apply(x, self.usam_pre, B, keep_cls=False)

    #     # print('x.shape:',x.shape)
    #     cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     # print('x.shape:',x.shape)
        
    #     x = x + self.pos_embed[:,:x.size(1),:]
    #     # print('x.shape:',x.shape)

    #     x = self.pos_drop(x)
    #     # print('x.shape:',x.shape)

    #     for blk in self.blocks:
    #         x = blk(x)
      
    #     x = self.norm(x)
        
    #     return x

def sum_feature(trans,cnn,B):
    cls_token = trans[:,0,:]
    x = trans[:,1:,:]
    emb_reshape = x.permute(0,2,1).contiguous().view(B,768,16,16)
    emb_upsample =  F.interpolate(emb_reshape, size=(256, 256), mode='bilinear', align_corners=False)
    fused_features = emb_upsample + cnn
    fuse_down =  F.interpolate(fused_features, size=(16, 16), mode='bilinear', align_corners=False)
    fuse_features = fuse_down.view(B,768,256).permute(0,2,1).contiguous()
    out = torch.cat((cls_token.unsqueeze(1),fuse_features),dim=1)
    return out

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint