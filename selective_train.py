import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from model import get_model
from data_loader import PlantDataset
import json
import os
from collections import defaultdict
import time

def evaluate_per_class_accuracy(model, val_loader, classes):
    """Evaluate accuracy per class and return class-wise accuracies."""
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda() if torch.cuda.is_available() else inputs, labels.cuda() if torch.cuda.is_available() else labels
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for label, pred in zip(labels, predicted):
                class_total[classes[label.item()]] += 1
                if label == pred:
                    class_correct[classes[label.item()]] += 1

    class_accuracies = {}
    for cls in classes:
        if class_total[cls] > 0:
            class_accuracies[cls] = 100 * class_correct[cls] / class_total[cls]
        else:
            class_accuracies[cls] = 0.0

    return class_accuracies

def get_problematic_classes(class_accuracies, threshold=96.0):
    """Get classes with accuracy below threshold."""
    return [cls for cls, acc in class_accuracies.items() if acc < threshold]

def create_filtered_dataset(train_dataset, problematic_classes, classes):
    """Create a subset of the dataset containing only problematic classes."""
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    problematic_indices = []

    for idx, (_, label) in enumerate(train_dataset.samples):
        if classes[label] in problematic_classes:
            problematic_indices.append(idx)

    return Subset(train_dataset, problematic_indices)

def selective_train(model_path='plant_model.pth', data_dir='MasterDataset', epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training dataset to get classes
    train_dataset = PlantDataset(os.path.join(data_dir, 'train'))
    classes = train_dataset.classes
    num_classes = len(classes)

    # Load model
    model = get_model(num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model for fine-tuning.")
    else:
        print("No existing model found. Training from scratch.")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load validation dataset for evaluation
    val_dataset = PlantDataset(os.path.join(data_dir, 'val'), transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate per-class accuracy
    print("Evaluating per-class accuracy...")
    class_accuracies = evaluate_per_class_accuracy(model, val_loader, classes)

    # Print class accuracies
    print("\nClass-wise accuracies:")
    for cls, acc in sorted(class_accuracies.items(), key=lambda x: x[1]):
        print(f"{cls}: {acc:.2f}%")

    # Get problematic classes
    problematic_classes = get_problematic_classes(class_accuracies, threshold=96.0)
    print(f"\nProblematic classes (accuracy < 96%): {problematic_classes}")

    if not problematic_classes:
        print("All classes have accuracy >= 96%. No selective training needed.")
        return

    # Create filtered training dataset
    print("Creating filtered training dataset...")
    train_dataset.transform = transform
    filtered_train_dataset = create_filtered_dataset(train_dataset, problematic_classes, classes)
    train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Filtered dataset size: {len(filtered_train_dataset)} samples")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        # Evaluate after each epoch
        class_accuracies = evaluate_per_class_accuracy(model, val_loader, classes)
        overall_accuracy = sum(class_accuracies.values()) / len(class_accuracies)
        print(f'Epoch {epoch + 1} completed. Overall validation accuracy: {overall_accuracy:.2f}%')

        # Check if all problematic classes are now above threshold
        current_problematic = get_problematic_classes(class_accuracies, threshold=96.0)
        if not current_problematic:
            print("All classes now have accuracy >= 96%. Stopping early.")
            break

        # Pause for CPU cooling (5 seconds)
        print("Training paused for CPU cooling. Resuming in 5 seconds...")
        time.sleep(5)

    # Save the updated model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Final evaluation
    final_accuracies = evaluate_per_class_accuracy(model, val_loader, classes)
    final_overall = sum(final_accuracies.values()) / len(final_accuracies)
    print(f"\nFinal overall accuracy: {final_overall:.2f}%")
    print("Final class accuracies:")
    for cls, acc in sorted(final_accuracies.items(), key=lambda x: x[1]):
        print(f"{cls}: {acc:.2f}%")

if __name__ == '__main__':
    selective_train()
