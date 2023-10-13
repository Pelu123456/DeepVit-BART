
from aa_encoder import AARDecoder
import dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm


num_epochs = 4
batch_size = 1
learning_rate = 1e-4

train_dataset = dataset.train_dataset  
# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


model = timm.create_model('vit_base_patch16_224', pretrained=True)


model.head = nn.Linear(in_features=model.head.in_features, out_features=vocab_size)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_text, target_text = batch 
        optimizer.zero_grad()
        outputs = model(input_text)  
        loss = criterion(outputs, target_text)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "test_model.pth")
