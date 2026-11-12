import torch
import torch.nn as nn
import torchvision.models as models


class CustomArtCNN(nn.Module):
    """
    4-layer Sequential CNN architecture adapted from the undergraduate paper.
    Uses 4 Conv2D blocks with BatchNorm, ReLU non-linearity, and MaxPool,
    outputting binary logits for Fake (0) vs. Real (1) artwork classification.
    """
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_model(model_type: str = "custom", pretrained: bool = True) -> nn.Module:
    if model_type == "custom":
        return CustomArtCNN()
    elif model_type == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")
