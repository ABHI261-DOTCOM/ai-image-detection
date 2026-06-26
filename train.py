import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from model import ImageDetector
import os

# -----------------------------
# TRANSFORMS (IMPORTANT)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = datasets.ImageFolder("dataset", transform=transform)

print("Classes:", dataset.classes)

# -----------------------------
# SPLIT DATASET (80% train / 20% val)
# -----------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# MODEL
# -----------------------------
model = ImageDetector()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(3):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # -----------------------------
    # VALIDATION
    # -----------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {total_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("-"*30)

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("weights", exist_ok=True)

torch.save(model.state_dict(), "weights/model.pth")

print("✅ Model trained and saved!")
