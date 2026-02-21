"""
MNIST CNN Model for Adversarial Robustness Research.

A standard CNN architecture for MNIST classification.
This serves as the baseline model for robustness evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Standard CNN for MNIST digit classification.
    
    Architecture:
    - Conv layers for feature extraction
    - Fully connected layers for classification
    """
    
    def __init__(self, num_classes: int = 10):
        super(MNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # Input: 128 * 3 * 3 = 1152 (after 2 poolings on 28x28)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv block 1: 1x28x28 -> 32x14x14
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2: 32x14x14 -> 64x7x7
        x = self.pool(F.relu(self.conv2(x)))
        
        # Conv block 3: 64x7x7 -> 128x3x3
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def get_model(num_classes: int = 10, device: str = "cpu") -> MNISTNet:
    """
    Factory function to create and initialize the model.
    
    Args:
        num_classes: Number of output classes (10 for MNIST)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Initialized MNISTNet model
    """
    model = MNISTNet(num_classes=num_classes)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = get_model()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print(f"Trainable parameters: {count_parameters(model):,}")
