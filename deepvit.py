import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_
from deepvit import DeepVisionTransformer

# Tạo lớp AAR Decoder
class AARDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, encoded_image):
        embedded_input = self.embed_layer(input_ids)
        output = embedded_input
        for layer in self.layers:
            output, _ = layer(output, encoded_image, encoded_image)  # AAR Attention
        output_logits = self.linear(output)
        return output_logits

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, group=False, re_atten=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AARDecoder(dim, num_heads, attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x, attn):
        x = x + self.attn(self.norm1(x), attn) 
        x = x + self.mlp(self.norm2(x))
        return x, attn

class DeepVisionTransformer(nn.Module):
    def forward_features(self, x):
        if self.cos_reg:
            atten_list = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn = None
        for blk in self.blocks:
            x, attn = blk(x, attn)
            if self.cos_reg:
                atten_list.append(attn)

        x = self.norm(x)
        if self.cos_reg and self.training:
            return x[:, 0], atten_list
        else:
            return x[:, 0]

    def forward(self, x):
        if self.cos_reg and self.training:
            x, atten = self.forward_features(x)
            x = self.head(x)
            return x, atten
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x

# Tạo mô hình DeepViT với AAR
model = DeepVisionTransformer(
    img_size=224,
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
    cos_reg=True,  # Kích hoạt regularization cosine
    use_cnn_embed=False,
    apply_transform=None,
    transform_scale=False,
    scale_adjustment=1.0
)

# Tối ưu hóa và hàm loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Huấn luyện mô hình với attention autoregressive (AAR)
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
