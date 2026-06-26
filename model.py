import torch.nn as nn
from torchvision import models

class ImageDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights="DEFAULT")

        # DO NOT freeze everything
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)
