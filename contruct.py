from transformers import BartForConditionalGeneration, BartTokenizer, EncoderDecoderModel
from . import aa_encoder
from .aa_encoder import AARDecoder
from . import encoder

dvit_encoder = encoder.CustomDVitEncoder()

aar_decoder = AARDecoder(vocab_size, embed_dim, num_layers, num_heads, hidden_dim, dropout)

model = EncoderDecoderModel(encoder=dvit_encoder, decoder=aar_decoder)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
