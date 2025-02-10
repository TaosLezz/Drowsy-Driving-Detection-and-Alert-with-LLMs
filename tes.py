"""Chỉ xử lý ko hiển thị video"""
# import cv2
# import torch
# import time
# import numpy as np
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import collections

# # Cấu hình thiết bị
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load model và processor một lần
# model_path = r"E:\aHieu_LLMs\Drowsy-Driving-Detection-LLM\clip-vit-large-patch14"
# model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
# processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)

# # Câu mô tả trạng thái
# texts = [
#     "A driver with open eyes and an alert expression",  # Tỉnh táo
#     "A drowsy or sleeping driver with closed or half-closed eyes"  # Buồn ngủ / Ngủ
# ]
# labels = ["Awake", "Drowsy or Sleeping"]

# # Hàm dự đoán trạng thái từ frame ảnh
# def predict(image):
#     inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
#     outputs = model(**inputs)
#     probs = outputs.logits_per_image.softmax(dim=1)
#     return labels[probs.argmax().item()], probs

# # Đọc video
# video_path = r"E:\aHieu_LLMs\Drowsy-Driving-Detection-LLM\video\test.mp4"  
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))  
# frame_interval = fps // 2 

# consecutive_drowsy = 0  
# alert_threshold = 3 
# frame_counter = 0

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if frame_counter % frame_interval == 0:
#         image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
#         start_time = time.time()
#         prediction, probs = predict(image)
#         end_time = time.time()
        
#         print(f"Frame {frame_counter}: {prediction} | Time: {end_time - start_time:.4f} s")

#         if prediction == "Drowsy or Sleeping":
#             consecutive_drowsy += 1
#         else:
#             consecutive_drowsy = 0
        
#         if consecutive_drowsy >= alert_threshold:
#             print("🚨 CẢNH BÁO: Tài xế đang buồn ngủ! 🚨")
#             consecutive_drowsy = 0 

#     frame_counter += 1

# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch
import time
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Cấu hình thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model và processor một lần
model_path = r"E:\aHieu_LLMs\Drowsy-Driving-Detection-LLM\clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)

# Câu mô tả trạng thái
texts = [
    "A driver with open eyes and an alert expression",  # Tỉnh táo
    "A drowsy or sleeping driver with closed or half-closed eyes"  # Buồn ngủ / Ngủ
]
labels = ["Awake", "Drowsy or Sleeping"]

# Hàm dự đoán trạng thái từ frame ảnh
def predict(image):
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return labels[probs.argmax().item()], probs

# Đọc video
video_path = r"E:\aHieu_LLMs\Drowsy-Driving-Detection-LLM\video\test.mp4"  # Thay bằng đường dẫn video của bạn
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Lấy FPS của video
frame_interval = fps // 2  # Lấy 2 frame mỗi giây

frame_width = int(cap.get(3))  # Chiều rộng video
frame_height = int(cap.get(4))  # Chiều cao video

# Ghi video kết quả
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

consecutive_drowsy = 0  # Số lần liên tiếp buồn ngủ
alert_threshold = 3  # Ngưỡng cảnh báo
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_counter % frame_interval == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        start_time = time.time()
        prediction, probs = predict(image)
        end_time = time.time()
        
        print(f"Frame {frame_counter}: {prediction} | Time: {end_time - start_time:.4f} s")

        if prediction == "Drowsy or Sleeping":
            consecutive_drowsy += 1
        else:
            consecutive_drowsy = 0 
        
        if consecutive_drowsy >= alert_threshold:
            cv2.putText(frame, "ALERT! Driver is Drowsy!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 0, 255), 3, cv2.LINE_AA)
            consecutive_drowsy = 0  
        else:
            cv2.putText(frame, f"State: {prediction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Drowsiness Detection", frame)
    out.write(frame) 

    frame_counter += 1

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
