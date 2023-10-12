# dataset.py
from datasets import load_dataset

# Tải tập dữ liệu CNN/Daily Mail
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Chuẩn bị dữ liệu huấn luyện
train_dataset = dataset["train"]

# Chọn một số ví dụ để mô phỏng huấn luyện
num_examples = 100

# Lấy dữ liệu gốc
input_texts = train_dataset["article"][:num_examples]
target_texts = train_dataset["highlights"][:num_examples]

# Chuyển đổi dữ liệu văn bản thành chuỗi
input_texts = [str(text) for text in input_texts]
target_texts = [str(text) for text in target_texts]

# Lưu dữ liệu đã tiền xử lý vào một thư mục hoặc file để sử dụng trong quá trình huấn luyện
with open("preprocessed_data/input.txt", "w", encoding="utf-8") as input_file:
    input_file.write("\n".join(input_texts))

with open("preprocessed_data/target.txt", "w", encoding="utf-8") as target_file:
    target_file.write("\n".join(target_texts))
