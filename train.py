import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import get_model
from data_loader import PlantDataset, get_data_loaders
from torchvision import transforms
import os
import json
import time
from torch.cuda.amp import GradScaler, autocast

def train_model(data_dir, num_epochs=5, batch_size=32 , learning_rate=0.001, save_path='plant_model.pth', resume=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use data loaders for ResNet50 (224x224)
    train_loader, val_loader, classes = get_data_loaders(os.path.join(data_dir, 'train'), batch_size=batch_size, resize_size=(224, 224))

    # Initialize model
    num_classes = len(classes)
    model = get_model(num_classes).to(device)
    print(f"Using ResNet50 model with {num_classes} classes.")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Learning rate scheduler for faster convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Resume from saved state if available
    start_epoch = 0
    best_accuracy = 0.0
    patience = 5  # Number of epochs to wait for improvement before early stopping
    epochs_no_improve = 0
    if resume and os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint.get('best_accuracy', 0.0)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            print(f"Resuming training from epoch {start_epoch} with best accuracy {best_accuracy:.2f}%")
        else:
            # Old format: only model state_dict
            model.load_state_dict(checkpoint)
            print("Loaded model state_dict from old format. Starting from epoch 0.")
    else:
        print("Starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Pause after every 10 batches to prevent overheating
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)} completed. Pausing for 5 seconds to cool down...")
                time.sleep(5)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

        # Check for improvement
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'epochs_no_improve': epochs_no_improve
            }
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint at epoch {epoch+1} with accuracy {accuracy:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {patience} consecutive epochs.")
            break

        # Step the scheduler
        scheduler.step()

        # Pause between epochs to prevent overheating
        print("Epoch completed. Pausing for 10 seconds to cool down...")
        time.sleep(10)

    print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')
    return model, classes

if __name__ == '__main__':
    # Use the MasterDataset directory
    data_dir = 'MasterDataset'
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found. Please prepare your dataset in the correct structure.")
        exit(1)

    model, classes = train_model(data_dir, batch_size=32)

    # Save class names for later use
    with open('classes.json', 'w') as f:
        json.dump(classes, f)
