import timm

class CustomDVitEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, input_text):
        encoded_text = self.encoder(input_text)
        return encoded_text
