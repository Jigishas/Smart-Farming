import torch
import torch.nn as nn
import torchvision.models as models

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, variant='b0'):
        super(PlantDiseaseModel, self).__init__()
        # Use EfficientNet as base model
        if variant == 'b0':
            self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        elif variant == 'b1':
            self.efficientnet = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
        # Replace the final classifier layer
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)

def get_model(num_classes, variant='b0'):
    model = PlantDiseaseModel(num_classes, variant)
    return model
