import argparse
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

def main(input_text, model_path):
    # Tạo một tokenizer và mô hình tương ứng
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained(model_path)

    # Mã hóa chuỗi và tạo tóm tắt
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Giải mã tóm tắt và in ra
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("Tóm tắt:", summary_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="running:")
    parser.add_argument("--input", type=str, help="output:")
    parser.add_argument("--model_path", type=str, help="trained model: ")

    args = parser.parse_args()

    if args.input is None or args.model_path is None:
        print("Vui lòng cung cấp đầu vào và đường dẫn đến mô hình.")
    else:
        main(args.input, args.model_path)
