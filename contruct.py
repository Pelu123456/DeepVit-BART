from transformers import BartForConditionalGeneration, BartTokenizer, EncoderDecoderModel
from . import aa_encoder
from .aa_encoder import CustomAARDecoder
from . import encoder

# Khởi tạo DVit encoder
dvit_encoder = encoder.CustomDVitEncoder("dvit_bart", pretrained=True)

# Khởi tạo AAR decoder
vocab_size = 30000  # Số lượng từ trong từ điển
embed_dim = 512
num_layers = 6
num_heads = 8
dropout = 0.1
aar_decoder = CustomAARDecoder(vocab_size, embed_dim, num_layers, num_heads, dropout)

# Khởi tạo tokenizer từ pre-trained BART
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Tạo mô hình hoàn chỉnh với encoder và decoder
model = EncoderDecoderModel(encoder=dvit_encoder, decoder=aar_decoder)

# Sử dụng mô hình để đào tạo và dự đoán
