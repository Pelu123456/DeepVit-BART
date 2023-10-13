import torch
import torch.nn as nn
import timm  # Thêm import timm

class AARDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_text, target_text):
        embedded_input = self.embed_layer(input_text)
        output = embedded_input
        for layer in self.layers:
            output, _ = layer(output, target_text, target_text)  # AAR Attention
        output_logits = self.linear(output)
        return output_logits

class CustomDVitEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, input_text):
        encoded_text = self.encoder(input_text)
        return encoded_text

class DeepVisionTransformer(nn.Module):
    def __init__(self, model_name, vocab_size, embed_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([
            AARDecoder(vocab_size, embed_dim, num_layers, num_heads, dropout) for _ in range(len(depth))
        ])
        self.head = nn.Linear(embed_dim, vocab_size)
        self.encoder = CustomDVitEncoder(model_name, pretrained=True)

    def forward(self, input_text):
        encoded_image = self.encoder(input_text)
        for block in self.blocks:
            input_text = block(input_text, encoded_image)
        output_logits = self.head(input_text)
        return output_logits
