import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        # The structure is root_dir/class_name/image.jpg
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.classes)
                    self.classes.append(class_name)
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(data_dir, batch_size=32, train_split=0.8, resize_size=(240, 240)):
    # Training transforms with augmentations for better accuracy
    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transforms without augmentations
    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create full dataset without transform to get classes
    full_dataset = PlantDataset(data_dir, transform=None)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(f"Dataset split resulted in empty train or val set. Total samples: {len(full_dataset)}, train_split: {train_split}")

    # Split indices
    indices = list(range(len(full_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets with appropriate transforms
    train_dataset = torch.utils.data.Subset(PlantDataset(data_dir, transform=train_transform), train_indices)
    val_dataset = torch.utils.data.Subset(PlantDataset(data_dir, transform=val_transform), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, full_dataset.classes
