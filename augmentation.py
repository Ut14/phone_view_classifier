import os
from PIL import Image
import torchvision.transforms as T
import random

# Source and destination folders
input_root = "phone_view"
output_root = "augmented_phone_view"
os.makedirs(output_root, exist_ok=True)

# Define augmentations
augment = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
])

# How many augmented versions per image
AUG_PER_IMAGE = 10

# Loop through each class folder
for label in os.listdir(input_root):
    class_input_path = os.path.join(input_root, label)
    class_output_path = os.path.join(output_root, label)
    os.makedirs(class_output_path, exist_ok=True)

    for img_file in os.listdir(class_input_path):
        img_path = os.path.join(class_input_path, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            base_name = os.path.splitext(img_file)[0]

            # Save original resized version
            img.resize((224, 224)).save(os.path.join(class_output_path, f"{base_name}_orig.jpg"))

            # Generate augmentations
            for i in range(AUG_PER_IMAGE):
                aug_img = augment(img)
                aug_img.save(os.path.join(class_output_path, f"{base_name}_aug{i}.jpg"))

            print(f"✅ Augmented: {img_file} in '{label}'")
        except Exception as e:
            print(f"⚠️ Error with {img_file}: {e}")
