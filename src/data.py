import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WasteDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, transform=None):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label, row["path"]


def load_classes(splits_dir):
    classes_path = Path(splits_dir) / "classes.json"
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    return classes, {label: idx for idx, label in enumerate(classes)}


def get_transforms(image_size=224, train=False):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_weighted_sampler(dataset):
    labels = [dataset.class_to_idx[label] for label in dataset.df["label"]]
    counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / counts.float().clamp_min(1)
    sample_weights = class_weights[torch.tensor(labels)]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def compute_class_weights(csv_path, class_to_idx):
    df = pd.read_csv(csv_path)
    labels = torch.tensor([class_to_idx[label] for label in df["label"]])
    counts = torch.bincount(labels, minlength=len(class_to_idx)).float()
    weights = counts.sum() / (len(class_to_idx) * counts.clamp_min(1))
    return weights
