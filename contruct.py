from transformers import BartForConditionalGeneration, BartTokenizer, EncoderDecoderModel
from . import aa_encoder
from .aa_encoder import CustomAARDecoder  # Sử dụng phiên bản tùy chỉnh của AARDecoder
from . import encoder

dvit_encoder = encoder.CustomDVitEncoder()

aar_decoder = CustomAARDecoder(vocab_size, embed_dim, num_layers, num_heads, dropout)  # Sử dụng phiên bản tùy chỉnh của AARDecoder

model = EncoderDecoderModel(encoder=dvit_encoder, decoder=aar_decoder)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
