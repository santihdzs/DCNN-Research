"""
Training pipeline for the MNIST model.

Provides utilities for:
- Data loading
- Model training
- Checkpoint management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json


def get_optimizer(model: nn.Module, optimizer: str = "adam", lr: float = 1e-3):
    """
    Get optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer type ('adam', 'sgd', 'adamw')
        lr: Learning rate
    
    Returns:
        Optimizer instance
    """
    optimizers = {
        "adam": optim.Adam(model.parameters(), lr=lr),
        "sgd": optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "adamw": optim.AdamW(model.parameters(), lr=lr),
    }
    return optimizers.get(optimizer.lower(), optim.Adam(model.parameters(), lr=lr))


def get_criterion(loss: str = "crossentropy"):
    """
    Get loss function.
    
    Args:
        loss: Loss type ('crossentropy', 'mse')
    
    Returns:
        Loss function
    """
    losses = {
        "crossentropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
    }
    return losses.get(loss.lower(), nn.CrossEntropyLoss())


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu"
):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu"
):
    """
    Evaluate model on test/validation set.
    
    Args:
        model: PyTorch model
        dataloader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Dictionary with loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_dir: str = "checkpoints"
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Metrics dictionary
        save_dir: Directory to save checkpoint
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save with epoch number
    torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch}.pt")
    
    # Also save as latest
    torch.save(checkpoint, save_path / "latest.pt")
    
    print(f"Checkpoint saved: {save_path / f'checkpoint_epoch_{epoch}.pt'}")


def save_model(model: nn.Module, save_path: str):
    """
    Save just the model weights.
    
    Args:
        model: PyTorch model
        save_path: Path to save model
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved: {save_path}")
