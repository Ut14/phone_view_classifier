from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import numpy as np
import joblib
from transformers import CLIPProcessor, CLIPModel
import io
import os
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Init FastAPI app and template dir ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Load CLIP and SVM model once ---
clip_model = CLIPModel.from_pretrained(
    "Ut14/clip-phone-view", subfolder="clip_model", token=HF_TOKEN
)
clip_processor = CLIPProcessor.from_pretrained(
    "Ut14/clip-phone-view", subfolder="clip_processor", token=HF_TOKEN
)

# --- Load SVM Model ---
svm_model = joblib.load("svm_phone_view_model.joblib")
label_map = {0: "front", 1: "back", 2: "side"}

# --- Extract embedding ---
def extract_clip_embedding(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.squeeze().numpy()

# --- Serve HTML page with file picker ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Handle image prediction ---
@app.post("/predict-image")
async def predict_image(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        embedding = extract_clip_embedding(image)
        pred = svm_model.predict([embedding])[0]
        prediction = label_map[pred].upper()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e)
        })
