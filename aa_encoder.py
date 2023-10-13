import torch
import torch.nn as nn

class CustomAARDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            CustomMultiheadAttention(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_text, target_text):
        embedded_input = self.embed_layer(input_text)
        output = embedded_input
        for layer in self.layers:
            output, _ = layer(output, target_text, target_text)  # Custom AAR Attention
        output_logits = self.linear(output)
        return output_logits

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        return self.custom_forward(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)

    def custom_forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        q, k, v = self.in_proj_qkv(query, key, value)
        q = q * self.scale
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_output_weights += attn_mask
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.float().masked_fill(
                key_padding_mask,
                float('-inf'),
            ).type_as(attn_output_weights)
        attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_output_weights

    def in_proj_qkv(self, query, key, value):
        return (torch.matmul(query, self.in_proj_weight[:query.size(-1), :].t()),
                torch.matmul(key, self.in_proj_weight[query.size(-1): query.size(-1) * 2, :].t()),
                torch.matmul(value, self.in_proj_weight[query.size(-1) * 2:, :].t()))
