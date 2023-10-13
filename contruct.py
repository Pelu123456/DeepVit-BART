from transformers import BartForConditionalGeneration, BartTokenizer, EncoderDecoderModel
from . import aa_encoder
from .aa_encoder import CustomAARDecoder  
from . import encoder

dvit_encoder = encoder.CustomDVitEncoder()

aar_decoder = CustomAARDecoder(vocab_size, embed_dim, num_layers, num_heads, dropout) 

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
