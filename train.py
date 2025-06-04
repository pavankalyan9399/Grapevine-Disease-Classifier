import os
import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import train_loader, test_loader
from model import get_model

# Check device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = get_model().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create "models" directory before training
os.makedirs("models", exist_ok=True)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    print(f"\n🔄 Epoch {epoch+1}/{num_epochs} started...")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f"🟢 Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1}/{num_epochs} - Average Loss: {running_loss / len(train_loader):.4f}")

# Save trained model
model_save_path = "models/grapevine_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\n🎉 Model training complete and saved at {model_save_path} ✅")
