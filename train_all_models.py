import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import get_loaders

# Map model names to their import paths and .pth filenames
model_info = {
    "cnn": ("model_cnn", "cnn.pth"),
    "alexnet": ("model_alexnet", "alexnet.pth"),
    "vgg": ("model_vgg", "vgg.pth"),
    "resnet": ("model_resnet", "resnet.pth"),
    "lstm": ("model_lstm", "lstm.pth"),
    "rcnn": ("model_rcnn", "rcnn.pth")
}

# Training configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1  # Increase if needed
lr = 0.001

train_loader, _ = get_loaders()
os.makedirs("models", exist_ok=True)

def train_and_save(model_name, model_module, save_name):
    # Dynamically import model
    module = __import__(model_module)
    model = module.get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"✅ [{model_name.upper()}] Epoch {epoch+1} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), os.path.join("models", save_name))
    print(f"📁 Saved {model_name} weights to models/{save_name}")

# Train all models
for model_name, (module, filename) in model_info.items():
    print(f"\n🚀 Training {model_name.upper()} model...")
    train_and_save(model_name, module, filename)
