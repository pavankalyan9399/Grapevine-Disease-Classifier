import torchvision.models as models
import torch.nn as nn

def get_model(num_classes=4):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
