import torch
from datasets import load_dataset
from transformers import BartTokenizer
from rouge import Rouge

# Initialize the tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Import your model here if it's in a separate file
from aa_encoder import CustomAARDecoder

# Initialize your model
vocab_size = tokenizer.vocab_size
embed_dim = 64
num_layers = 6
num_heads = 8
dropout = 0.2

model = CustomAARDecoder(vocab_size, embed_dim, num_layers, num_heads, dropout)

dataset = load_dataset("cnn_dailymail", "3.0.0")
test_dataset = dataset["test"]

actual_summaries = []
predicted_summaries = []

with torch.no_grad():  # Sửa thành with torch.no_grad() để tính toán không tính toán gradient
    for example in test_dataset:
        input_text = example["article"]
        target_text = example["highlights"]

        actual = target_text  # Đây là tóm tắt thực tế
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length")
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Đây là tóm tắt dự đoán

        actual_summaries.append(actual)
        predicted_summaries.append(predicted)

# Tính các thước đo ROUGE

rouge = Rouge()
scores = rouge.get_scores(predicted_summaries, actual_summaries, avg=True)

# In kết quả ROUGE
print("ROUGE Scores:", scores)
