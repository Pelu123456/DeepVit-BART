from datasets import load_dataset
from transformers import BartTokenizer
from rouge import Rouge

dataset = load_dataset("cnn_dailymail", "3.0.0")
test_dataset = dataset["test"]  


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


rouge = Rouge()


actual_summaries = []  
predicted_summaries = []  


with torch.no_grad():
    for example in test_dataset:
        input_text = example["article"]
        target_text = example["highlights"]


        actual = target_text  # Đây là tóm tắt thực tế
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length")
        outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Đây là tóm tắt dự đoán

        actual_summaries.append(actual)
        predicted_summaries.append(predicted)

# Tính các thước đo ROUGE
scores = rouge.get_scores(predicted_summaries, actual_summaries, avg=True)

# In kết quả ROUGE
print("ROUGE Scores:", scores)
