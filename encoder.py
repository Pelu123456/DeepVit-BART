import torch.nn as nn
from deepvit import DeepVisionTransformer  # Đảm bảo bạn đã cài đặt deepvit

class CustomDVitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dvit_encoder = DeepVisionTransformer(
            img_size=384,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            group=False,
            re_atten=True,
            cos_reg=False,
            use_cnn_embed=False,
            apply_transform=None,
            transform_scale=False,
            scale_adjustment=1.0
        )

    def forward(self, images):
        encoded_image = self.dvit_encoder(images)
        return encoded_image
