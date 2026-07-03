import torch
import torch.nn as nn
import torch.nn.functional as F

  
class AdaptivePyramidConvUnit(nn.Module):
    """
    Advanced Pyramid Convolution Unit
    - Depthwise Pyramid Convolution
    - Dynamic Scale Selection
    - Channel Attention Fusion
    """

    def __init__(
        self,
        embed_dim,
        dilation=(1,2,3),
        reduction=4
    ):
        super().__init__()

        self.num_scales = len(dilation)

        # ===== multi-scale depthwise conv =====
        self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        embed_dim,
                        embed_dim,
                        kernel_size=3,
                        padding=k,
                        dilation=k,
                        groups=embed_dim  # depthwise
                    ),
                    nn.BatchNorm2d(embed_dim),
                    nn.GELU(),                    
                    nn.Conv2d(embed_dim, embed_dim, 1)  # pointwise
                )
            for k in dilation
        ])

        # ===== scale attention =====
        hidden = max(embed_dim // reduction, 16)

        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim * self.num_scales, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_scales, 1),
            nn.Softmax(dim=1)
        )
        
        # ===== fusion layer =====
        self.fuse = nn.Conv2d(embed_dim, embed_dim, 1)

        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):

        B, C, H, W = x.shape

        # multi-scale features
        feats = [conv(x) for conv in self.branches]

        concat_feat = torch.cat(feats, dim=1)

        # scale attention weights
        attn = self.scale_attn(concat_feat)
        #print(attn.shape)

        attn = attn.view(B, self.num_scales, 1, 1, 1)

        feats = torch.stack(feats, dim=1)

        # weighted fusion
        out = (attn * feats).sum(dim=1)

        out = self.fuse(out)

        out = self.norm(out)

        return out
    

class Segmenter(nn.Module):
    def __init__(self, encoder, decoder, n_cls, embed_dim=384):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder

        self.decoder = decoder
        self.pyramid_conv = AdaptivePyramidConvUnit(embed_dim)
        
        
    def forward(self, im, use_momentum=False, return_features=False):

        H, W = im.size(2), im.size(3)
        
        # ===== 编码器输出 =====
        x = self.encoder(im, return_features=True)
        
        num_extra_tokens = 1 + self.encoder.distilled_num
        patch_token = x[:, num_extra_tokens:]  # [B, N, C]
        
        
        # ===== Token → Feature Map =====
        B, N, C = patch_token.shape
        H_p, W_p = H // self.patch_size, W // self.patch_size
        patch_feat = patch_token.transpose(1, 2).reshape(B, C, H_p, W_p)
        
        # ===== Stacked Pyramid Encoding =====
        patch_feat = self.pyramid_conv(patch_feat)
        

        # ===== Feature Map → Token =====
        patch_token = patch_feat.flatten(2).transpose(1, 2)

        fin_patch_masks, _, _ = self.decoder(patch_token, (H, W), distilled=False) 
        

        fin_pix_masks = F.interpolate(fin_patch_masks, size=(H, W), mode="bilinear")

        
        return {"fin_patch_masks": fin_patch_masks, 
                "fin_pix_masks": fin_pix_masks}
