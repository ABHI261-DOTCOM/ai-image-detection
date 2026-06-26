import torch
import torch.nn.functional as F
from model import ImageDetector
from preprocess import preprocess_image

model = ImageDetector()
model.load_state_dict(torch.load("weights/model.pth", map_location=torch.device('cpu')))
model.eval()

# IMPORTANT: match dataset.classes
classes = ["Fake", "Real"]

def detect_image(path):
    img = preprocess_image(path)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], float(conf.item())
