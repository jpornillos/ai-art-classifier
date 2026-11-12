import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import get_model
from src.dataset import get_data_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train AI Artwork Classifier")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "resnet18"])
    parser.add_argument("--save_path", type=str, default="custom_cnn.pth")
    parser.add_argument("--history_path", type=str, default="history.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training '{args.model}' on device: {device}")

    train_loader, val_loader, _ = get_data_loaders(args.data_dir, batch_size=args.batch_size)
    model = get_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved new best checkpoint ({best_val_acc*100:.2f}%) to {args.save_path}")

    with open(args.history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    main()
