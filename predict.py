import argparse
import torch
from PIL import Image
from torchvision import transforms
from src.model import get_model


def predict_image(image_path: str, model_path: str, model_type: str = "custom"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = get_model(model_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        prob_real = torch.sigmoid(logits).item()

    label = "REAL (Human Artwork)" if prob_real >= 0.5 else "FAKE (AI Generated)"
    confidence = prob_real if prob_real >= 0.5 else (1.0 - prob_real)

    print(f"File:       {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}% (Raw Score: {prob_real:.4f} | 0=Fake, 1=Real)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify Image as Real or AI Art")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="custom_cnn.pth", help="Path to saved .pth weights")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "resnet18"])
    args = parser.parse_args()

    predict_image(args.image, args.weights, args.model)
