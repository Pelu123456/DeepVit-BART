import torch
from datasets import load_dataset
from transformers import BartTokenizer
from rouge import Rouge
from aa_encoder import CustomAARDecoder

# Initialize the tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

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

with torch.no_grad():
    for example in test_dataset:
        input_text = example["article"]
        target_text = example["highlights"]

        actual = target_text  # This is the actual summary
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length")
        outputs = model(input_text=inputs["input_ids"], target_text=inputs["input_ids"])
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)  # This is the predicted summary

        actual_summaries.append(actual)
        predicted_summaries.append(predicted)

# Calculate ROUGE scores
rouge = Rouge()
scores = rouge.get_scores(predicted_summaries, actual_summaries, avg=True)

# Print ROUGE results
print("ROUGE Scores:", scores)
