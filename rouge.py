from rouge import Rouge

# Tạo một đối tượng Rouge
rouge = Rouge()

# Tạo danh sách kết quả thực tế và kết quả dự đoán từ dữ liệu kiểm tra
actual_summaries = []  # Danh sách các tóm tắt thực tế
predicted_summaries = []  # Danh sách các tóm tắt dự đoán

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = model(input_ids, attention_mask)
        
        actual = "actual"  
        predicted = "predicted"  
        
        actual_summaries.append(actual)
        predicted_summaries.append(predicted)

# Tính các thước đo ROUGE
scores = rouge.get_scores(predicted_summaries, actual_summaries, avg=True)

# In kết quả ROUGE
print("ROUGE Scores:", scores)
