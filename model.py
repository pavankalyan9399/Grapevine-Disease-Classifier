import torch
import torch.nn as nn
import torchvision.models as models
from data_loader import train_data  # Import training dataset

# Get number of classes dynamically
num_classes = len(train_data.classes)
print(f"Number of classes detected: {num_classes}")

# Load pre-trained ResNet18 model
def get_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Use correct number of classes
    return model
