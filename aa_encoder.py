import torch
import torch.nn as nn

class AARDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, target_ids):
        embedded_input = self.embed_layer(input_ids)
        output = embedded_input
        for layer in self.layers:
            output, _ = layer(output, target_ids, target_ids)  # AAR Attention
        output_logits = self.linear(output)
        return output_logits
