from datasets import load_dataset
import os
from torch.utils.data import Dataset
from transformers import BartTokenizer

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


class CustomDataset(Dataset):
    def __init__(self, input_file, target_file, tokenizer, max_length=1024):
        self.input_texts = []
        self.target_texts = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(input_file, "r", encoding="utf-8") as f:
            self.input_texts = f.read().splitlines()

        with open(target_file, "r", encoding="utf-8") as f:
            self.target_texts = f.read().splitlines()

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        inputs = self.tokenizer(input_text, max_length=self.max_length, return_tensors="pt", truncation=True, padding="max_length")
        targets = self.tokenizer(target_text, max_length=self.max_length, return_tensors="pt", truncation=True, padding="max_length")

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": targets["input_ids"].flatten(),
        }
