import os
import torch
import numpy as np
from PIL import Image
import joblib
from transformers import CLIPProcessor, CLIPModel

# --- Load CLIP Model and Processor ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Load Trained SVM Model ---
svm_model = joblib.load("svm_phone_view_model.joblib")

# --- Label Mapping ---
label_map = {0: "front", 1: "back", 2: "side"}

# --- Function to Extract CLIP Embedding ---
def extract_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().numpy()

# --- Prediction Function ---
def predict_view(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    emb = extract_clip_embedding(image_path)
    pred = svm_model.predict([emb])[0]
    print(f"\n Image: {image_path}\n Predicted view: {label_map[pred].upper()}")

# --- Example usage ---
if __name__ == "__main__":
    # 👇 Change this path to the image you want to test
    test_image_path = "images.png"  # Replace with your image path
    predict_view(test_image_path)
