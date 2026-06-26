from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    return img
