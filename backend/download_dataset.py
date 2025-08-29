# backend/download_dataset.py
from roboflow import Roboflow
import os
import shutil

# 1️⃣ Add your Roboflow API key
rf = Roboflow(api_key="HuBjsApkFg53Pzhr0yEK")  # keep inside quotes

# 2️⃣ Select your workspace & project
project = rf.workspace("varixscan").project("varicose-disease-dp898")

# 3️⃣ Download dataset (YOLOv8 format recommended)
dataset = project.version(1).download("yolov8")

# 4️⃣ Organize into your custom folder
target_images = "../dataset/images"
target_labels = "../dataset/annotations"

os.makedirs(target_images, exist_ok=True)
os.makedirs(target_labels, exist_ok=True)

# dataset.location points to the downloaded folder
download_path = dataset.location

# Copy images and labels to your structure
for root, dirs, files in os.walk(download_path):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(os.path.join(root, file), target_images)
        elif file.endswith(".txt"):  # YOLO labels
            shutil.copy(os.path.join(root, file), target_labels)

print("✅ Dataset downloaded and structured successfully!")
