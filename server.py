from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import gdown

app = FastAPI()

# اسم الموديل
MODEL_PATH = "vit_best_advanced.pth"

MODEL_URL = "https://drive.google.com/uc?export=download&id=1EYOaIZMCixjkdPEkZzpQIKSXh40Fs26B"

# تحميل الموديل إذا غير موجود
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# 🔥 إنشاء موديل ViT
model = models.vit_b_16(weights=None)

# ⚠️ نفس structure التدريب
model.heads.head = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model.heads.head.in_features, 7)
)

# تحميل weights
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)

model.eval()

# أسماء الكلاسات
classes = [
    "Bacterial Leaf Spot",
    "Black Rot",
    "Downy Mildew",
    "ESCA",
    "Healthy",
    "Leaf Blight",
    "Powdery Mildew"
]

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted = torch.argmax(probs).item()

        return {
            "class": classes[predicted],
            "confidence": float(probs[predicted])
        }

    except Exception as e:
        return {"error": str(e)}