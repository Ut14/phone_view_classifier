import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Paths
dataset_path = "augmented_phone_view"
label_map = {"front": 0, "back": 1, "side": 2}
reverse_map = {v: k for k, v in label_map.items()}

# Storage
features = []
labels = []

# Feature extraction function
def extract_clip_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze().numpy()

# Loop through all images
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue
    label = label_map[class_name]

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        try:
            emb = extract_clip_embedding(img_path)
            features.append(emb)
            labels.append(label)
            print(f"✅ Processed {img_path}")
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")

# Save features and labels
features = np.array(features)
labels = np.array(labels)
np.save("clip_features.npy", features)
np.save("clip_labels.npy", labels)

print("\n✅ Saved clip_features.npy and clip_labels.npy")
