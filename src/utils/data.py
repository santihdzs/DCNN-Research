"""
Data loading utilities for MNIST.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0
):
    """
    Get MNIST train and test data loaders.
    
    Args:
        batch_size: Batch size for DataLoaders
        data_dir: Directory to store/load MNIST data
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader


def get_data_info():
    """
    Get information about the MNIST dataset.
    
    Returns:
        Dictionary with dataset info
    """
    return {
        "classes": 10,
        "image_size": (1, 28, 28),
        "train_size": 60000,
        "test_size": 10000,
        "class_names": [str(i) for i in range(10)]
    }
