import torch
from typing import Tuple
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, random_split


class TransformSubset(Dataset):
    """Applies independent transforms to train, validation, and test subsets."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: Tuple[int, int] = (224, 224),
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # ImageFolder maps alphabetically: {'FAKE': 0, 'REAL': 1}
    full_dataset = ImageFolder(root=data_dir)
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        TransformSubset(train_sub, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        TransformSubset(val_sub, eval_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        TransformSubset(test_sub, eval_transform),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
