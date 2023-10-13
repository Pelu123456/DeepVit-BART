from aa_encoder import CustomAARDecoder
from dataset import CustomDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer

# Define hyperparameters
num_epochs = 4
batch_size = 1
learning_rate = 1e-4

# Define your model hyperparameters
vocab_size = 1
embed_dim = 64
num_layers = 6
num_heads = 8
dropout = 0.2

# Create the DataLoader
train_dataset = CustomDataset("preprocessed_data/input.txt", "preprocessed_data/target.txt", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize your model
model = CustomAARDecoder(vocab_size, embed_dim, num_layers, num_heads, dropout)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)  # Forward pass
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimization step
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), "test_model.pth")
