"""
Dental X-Ray Guardrail Classifier — Simple Custom CNN.
Trains a custom 3-layer Convolutional Neural Network on grail_dataset (Normal vs Xray).
Model Size: ~450 KB | Inference Time: < 1 ms | Pure PyTorch (No external weights)
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Simple Custom CNN Architecture (3 Conv Blocks + Classifier)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleXRayCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Input: 3 x 128 x 128
        self.features = nn.Sequential(
            # Block 1: 3 -> 16 channels (128x128 -> 64x64)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 16 -> 32 channels (64x64 -> 32x32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32 -> 64 channels (32x32 -> 16x16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Global average pooling (16x16 -> 1x1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────
def train_model(data_dir="grail_dataset", epochs=15, batch_size=32, lr=1e-3):
    dataset_path = Path(data_dir)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory '{dataset_path}' not found!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Fast preprocessing & augmentation
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset loading
    full_dataset = datasets.ImageFolder(str(dataset_path), transform=train_transform)
    classes = full_dataset.classes # ['Normal', 'Xray']
    print(f"Classes found: {classes} | Mapping: {full_dataset.class_to_idx}")
    print(f"Total Dataset Size: {len(full_dataset)} images")

    # 85% Train / 15% Val Split
    val_size = max(20, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Simple CNN
    model = SimpleXRayCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "xray_guardrail_simple_cnn.pth"

    best_acc = 0.0
    print(f"\nTraining Simple CNN for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # 1. Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_acc = train_correct / train_total * 100.0
        avg_train_loss = train_loss / train_total

        # 2. Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100.0
        avg_val_loss = val_loss / val_total

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (Acc: {train_acc:.1f}%) | "
            f"Val Loss: {avg_val_loss:.4f} (Val Acc: {val_acc:.1f}%)"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "class_to_idx": full_dataset.class_to_idx,
                "best_val_acc": best_acc,
                "arch": "SimpleXRayCNN",
            }, str(save_path))

    print(f"\n[DONE] Training Complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"[SAVED] Checkpoint saved to: {save_path} ({save_path.stat().st_size / 1024:.1f} KB)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Simple CNN for Dental X-Ray Guardrail")
    parser.add_argument("--data_dir", type=str, default="grail_dataset", help="Path to grail_dataset folder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    train_model(args.data_dir, args.epochs, args.batch_size, args.lr)
