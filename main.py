# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import torch
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load model và processor
# model = CLIPModel.from_pretrained(r"E:\aHieu_LLMs\Drowsy-Driving-Detection-LLM\clip-vit-large-patch14", local_files_only=True)
# processor = CLIPProcessor.from_pretrained(r"E:\aHieu_LLMs\Drowsy-Driving-Detection-LLM\clip-vit-large-patch14", local_files_only=True)
# import requests
# # Load hình ảnh
# url = "https://raw.githubusercontent.com/TaosLezz/Drowsy-Driving-Detection-and-Alert-with-LLMs/refs/heads/main/images/ngap_ngu.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# # Câu mô tả trạng thái
# # texts = ["A face is awake", "A face is drowsy or sleeping"]
# texts = [
#     "A driver with open eyes and an alert expression",  # Tỉnh táo
#     "A drowsy or sleeping driver with closed or half-closed eyes"  # Buồn ngủ / Ngủ
# ]
# import time
# start_time = time.time()
# # Xử lý dữ liệu
# inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# probs = outputs.logits_per_image.softmax(dim=1)

# # Kết quả
# labels = ["Awake", "Drowsy or Sleeping"]
# print(f"Prediction: {labels[probs.argmax().item()]}") 
# print(probs)
# end_time = time.time()
# total = end_time - start_time
# print(total)
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import requests
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model và processor một lần
model_path = r"E:\aHieu_LLMs\Drowsy-Driving-Detection-LLM\clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)

def predict(image, texts):
    """ Dự đoán trạng thái dựa trên hình ảnh và danh sách mô tả """
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    
    start_time = time.time()
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    end_time = time.time()
    
    labels = ["Awake", "Drowsy or Sleeping"]
    print(f"Prediction: {labels[probs.argmax().item()]}")
    print(probs)
    print(f"Processing Time: {end_time - start_time:.4f} seconds")

# Load hình ảnh từ URL
url = "https://raw.githubusercontent.com/TaosLezz/Drowsy-Driving-Detection-and-Alert-with-LLMs/refs/heads/main/images/ngap_ngu.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Câu mô tả trạng thái
texts = [
    "A driver with open eyes and an alert expression",  # Tỉnh táo
    "A drowsy or sleeping driver with closed or half-closed eyes"  # Buồn ngủ / Ngủ
]

# Gọi dự đoán
predict(image, texts)
