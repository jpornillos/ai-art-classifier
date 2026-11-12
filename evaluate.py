import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from src.model import get_model
from src.dataset import get_data_loaders


def plot_training_curves(history_path: str):
    if not os.path.exists(history_path):
        return
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="o", markersize=3)
    ax1.set_title("Training vs. Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", marker="o", markersize=3)
    ax2.set_title("Training vs. Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/training_curves.png", dpi=200)
    plt.close()


def generate_portfolio_assets(data_dir: str, weights_path: str, model_type: str, history_path: str):
    os.makedirs("assets", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_data_loaders(data_dir, batch_size=16)

    model = get_model(model_type).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    sample_images, sample_titles = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images)).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            if len(sample_images) < 8:
                for i in range(min(len(images), 8 - len(sample_images))):
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                    true_lbl = "REAL" if labels[i].item() == 1 else "FAKE"
                    pred_lbl = "REAL" if preds[i] == 1 else "FAKE"
                    conf = probs[i] if preds[i] == 1 else (1.0 - probs[i])
                    sample_images.append(img)
                    sample_titles.append(f"True: {true_lbl} | Pred: {pred_lbl} ({conf*100:.1f}%)")

    print("\n--- Test Set Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=["FAKE (AI)", "REAL (Human)"]))

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE (AI)", "REAL (Human)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Test Set Confusion Matrix ({model_type.upper()})")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=200)
    plt.close()

    # 2. Sample Predictions Grid
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, img, title in zip(axes.flat, sample_images, sample_titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("assets/sample_predictions.png", dpi=200)
    plt.close()

    # 3. Training Curves
    plot_training_curves(history_path)
    print("Saved charts to assets/confusion_matrix.png, assets/sample_predictions.png, and assets/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model and Generate Portfolio Charts")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--weights", type=str, default="custom_cnn.pth")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "resnet18"])
    parser.add_argument("--history", type=str, default="history.json")
    args = parser.parse_args()

    generate_portfolio_assets(args.data_dir, args.weights, args.model, args.history)
