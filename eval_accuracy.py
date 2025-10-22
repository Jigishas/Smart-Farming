import torch
import os

def get_current_accuracy(save_path='plant_model.pth'):
    if not os.path.exists(save_path):
        print("Model file not found. Please train the model first.")
        return None

    checkpoint = torch.load(save_path)
    if isinstance(checkpoint, dict) and 'best_accuracy' in checkpoint:
        best_accuracy = checkpoint['best_accuracy']
        print(f"Current best accuracy of the model: {best_accuracy:.2f}%")
        return best_accuracy
    else:
        print("No accuracy information found in the saved model. The model might be in old format.")
        return None

if __name__ == '__main__':
    get_current_accuracy()
