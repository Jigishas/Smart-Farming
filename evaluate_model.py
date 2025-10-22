import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_model
from data_loader import PlantDataset
import json
import os

def evaluate_model(data_dir='MasterDataset', model_path='plant_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        print("Model file not found.")
        return None

    # Get classes from training dataset
    train_dataset = PlantDataset(os.path.join(data_dir, 'train'))
    classes = train_dataset.classes
    num_classes = len(classes)

    model = get_model(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load validation dataset
    val_dataset = PlantDataset(os.path.join(data_dir, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (i + 1) % 10 == 0 or (i + 1) == len(val_loader):
                print(f"Evaluated {i+1}/{len(val_loader)} batches. Current accuracy: {100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    print(f'Current model validation accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    evaluate_model()
