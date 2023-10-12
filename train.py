    # Import các module và class từ các tệp khác
    from aa_encoder import AARDecoder
    from encoder import CustomDVitEncoder
    import dataset
    from deepvit import DeepVisionTransformer
    from construct import construct_deepvit_bart

    # Các thông số và siêu tham số huấn luyện
    num_epochs = 4
    batch_size = 1
    learning_rate = 1e-4

    # Tạo dataset từ tệp dataset.py
    train_dataset = dataset.train_dataset  # Thay thế bằng cách tạo dataset của bạn

    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Xây dựng mô hình DeepViT-BART từ các module trong deepvit.py và construct.py
    model = construct_deepvit_bart()

    # Xác định hàm loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Vòng lặp huấn luyện
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_text, target_text = batch  # Đảm bảo phân chia dữ liệu thành cặp (đầu vào, đầu ra) đúng cách
            optimizer.zero_grad()
            output = model(input_text=input_text, target_text=target_text)
            loss = criterion(output, target_text)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

    # Lưu trạng thái của mô hình sau quá trình đào tạo
    torch.save(model.state_dict(), "deepvit_bart_model.pth")
