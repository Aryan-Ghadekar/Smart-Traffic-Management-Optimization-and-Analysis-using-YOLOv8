import os
import torch
from ultralytics import YOLO

# 🔹 1️⃣ for MobaXterm
DATA_CONFIG = "/home/user93/ASEP_ambulance/data.yaml"

# 🔹 2️⃣ Check if GPU is Available (Force usage of CUDA for MobaXterm GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# 🔹 3️⃣ Check Dataset Paths & Warn if Missing
dataset_root = os.path.dirname(DATA_CONFIG)
data_paths = [
    "train/images", "train/labels",
    "valid/images", "valid/labels"
]

for path in data_paths:
    full_path = os.path.join(dataset_root, path)
    if not os.path.exists(full_path):
        print(f"⚠️ Warning: Missing folder: {full_path}")
    elif not os.listdir(full_path):
        print(f"⚠️ Warning: Folder is empty: {full_path}")

# 🔹 4️⃣ Load Pretrained YOLO Model
model = YOLO("yolov8n.pt")

# 🔹 5️⃣ Train Model (Ensuring GPU usage)
model.train(
    data=DATA_CONFIG,
    epochs=50,
    imgsz=640,
    batch=8,
    workers=0,  # For Windows Subsystem, setting workers to 0 avoids multiprocessing issues
    device=DEVICE,  # Ensures CUDA GPU usage
    cache=True,  # Cache images for faster training
    pretrained=True
)

# 🔹 6️⃣ Save the Model in ONNX Format
model.export(format="onnx")
