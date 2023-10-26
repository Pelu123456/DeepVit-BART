import torch
from aa_encoder import CustomAARDecoder
from dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from rouge import Rouge

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

vocab_size = tokenizer.vocab_size
embed_dim = 64
num_layers = 6
num_heads = 8
dropout = 0.2

# Load the trained model from the saved checkpoint
model = CustomAARDecoder(vocab_size, embed_dim, num_layers, num_heads, dropout)
model.load_state_dict(torch.load("test_model.pth"))
model.eval()

# Create a DataLoader for the test dataset
test_input_file_path = "preprocessed_data/input.txt"
test_target_file_path = "preprocessed_data/target.txt"
test_dataset = CustomDataset(test_input_file_path, test_target_file_path, tokenizer, max_length=64)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize lists to store actual and predicted summaries
actual_summaries = []
predicted_summaries = []

# Make predictions on the test dataset
for batch in test_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    with torch.no_grad():
        # Forward pass
        outputs = model(input_text=input_ids, target_text=labels)

    # Extract and store actual and predicted summaries
    actual = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Use join() to combine predicted summaries and filter out None values
    predicted_summaries = [summary for summary in tokenizer.batch_decode(outputs, skip_special_tokens=True) if summary is not None]
    predicted = "".join(predicted_summaries)
    actual_summaries.extend(actual)
    predicted_summaries.append(predicted)

# Calculate ROUGE scores for the test dataset
rouge = Rouge()
scores = rouge.get_scores(predicted_summaries, actual_summaries, avg=True)
print("ROUGE Scores:", scores)

