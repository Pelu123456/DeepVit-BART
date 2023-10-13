from datasets import load_dataset
import os

# Tải tập dữ liệu
dataset = load_dataset("cnn_dailymail", "3.0.0")

output_dir = "preprocessed_data"
os.makedirs(output_dir, exist_ok=True)

num_examples = 100

input_texts = dataset["train"]["article"][:num_examples]
target_texts = dataset["train"]["highlights"][:num_examples]

input_texts = [str(text) for text in input_texts]
target_texts = [str(text) for text in target_texts]

with open(os.path.join(output_dir, "input.txt"), "w", encoding="utf-8") as input_file:
    input_file.write("\n".join(input_texts))

with open(os.path.join(output_dir, "target.txt"), "w", encoding="utf-8") as target_file:
    target_file.write("\n".join(target_texts))
