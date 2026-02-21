"""
Benchmark pipeline for adversarial robustness evaluation.

Runs training + clean test + adversarial attacks (FGSM, BIM, JSMA, CW).
Supports caching of adversarial examples for faster future runs.

Usage:
    uv run python -m src.pipeline.benchmark
    uv run python -m src.pipeline.benchmark --epochs 5
    uv run python -m src.pipeline.benchmark --no-train
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mnist_model import get_model
from utils.data import get_mnist_loaders


# Attack configurations
ATTACK_CONFIGS = {
    "FGSM": {
        "class": "FGSM",
        "kwargs": {"eps": 0.3},
        "description": "Fast Gradient Sign Method (L∞)"
    },
    "BIM": {
        "class": "BIM", 
        "kwargs": {"eps": 0.3, "alpha": 1/255, "steps": 10},
        "description": "Basic Iterative Method (L∞)"
    },
    "JSMA": {
        "class": "JSMA",
        "kwargs": {"theta": 1.0, "gamma": 0.1},
        "description": "Jacobian-based Saliency Map Attack (L0)"
    },
    "CW": {
        "class": "CW",
        "kwargs": {"c": 1e-4, "kappa": 0, "steps": 20, "lr": 0.01},
        "description": "Carlini & Wagner (L2)"
    }
}


def get_attack(attack_name: str, model: nn.Module):
    """Create an attack instance by name."""
    import torchattacks
    
    config = ATTACK_CONFIGS[attack_name]
    attack_class = getattr(torchattacks, config["class"])
    return attack_class(model, **config["kwargs"])


def generate_adversarial_data(
    model: nn.Module,
    test_loader: DataLoader,
    attack_name: str,
    device: str = "cpu",
    cache_dir: str = "data/adversarial"
) -> tuple:
    """Generate or load cached adversarial test data."""
    cache_path = Path(cache_dir) / f"{attack_name}_test.pt"
    
    if cache_path.exists():
        print(f"  Loading cached {attack_name} adversarial data...")
        data = torch.load(cache_path, weights_only=True)
        return data["images"], data["labels"]
    
    print(f"  Generating {attack_name} adversarial examples...")
    
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    attack = get_attack(attack_name, model)
    
    batch_size = 100
    adv_images = []
    
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i+batch_size].to(device)
        labels = all_labels[i:i+batch_size].to(device)
        adv_batch = attack(batch, labels)
        adv_images.append(adv_batch.cpu())
        
        if (i // batch_size + 1) % 20 == 0:
            print(f"    Progress: {min(i + batch_size, len(all_images))}/{len(all_images)}")
    
    adv_images = torch.cat(adv_images, dim=0)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "images": adv_images,
        "labels": all_labels,
        "attack": attack_name,
        "timestamp": datetime.now().isoformat()
    }, cache_path)
    print(f"  Cached to {cache_path}")
    
    return adv_images, all_labels


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str = "cpu") -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return {
        "accuracy": 100. * correct / total,
        "loss": total_loss / len(data_loader),
        "correct": correct,
        "total": total
    }


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                epochs: int = 3, device: str = "cpu", checkpoint_dir: str = "checkpoints") -> dict:
    """Train the model."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_path / "mnist_cnn.pt"
    
    print(f"\n=== TRAINING ({epochs} epochs) ===")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        test_metrics = evaluate_model(model, test_loader, device)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_acc={100.*train_correct/train_total:.2f}%, "
              f"test_acc={test_metrics['accuracy']:.2f}%")
    
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return test_metrics


def run_benchmark(epochs: int = 3, device: str = "cpu", train: bool = True,
                  cache_dir: str = "data/adversarial", results_dir: str = "results",
                  fast: bool = False) -> dict:
    """Run full benchmark."""
    attacks_to_run = ["FGSM", "BIM"] if fast else list(ATTACK_CONFIGS.keys())
    
    print("=" * 50)
    print("ADVERSARIAL ROBUSTNESS BENCHMARK")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Train: {train} (epochs={epochs})")
    print(f"Attacks: {attacks_to_run}")
    
    print("\n=== LOADING DATA ===")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, data_dir="./data")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n=== MODEL ===")
    model = get_model(device=device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if train:
        final_metrics = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    else:
        model_path = Path("checkpoints/mnist_cnn.pt")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"Loaded model from {model_path}")
        else:
            print("No saved model found, training from scratch...")
            final_metrics = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
    
    print("\n=== CLEAN TEST ===")
    clean_metrics = evaluate_model(model, test_loader, device)
    print(f"Clean accuracy: {clean_metrics['accuracy']:.2f}%")
    
    print("\n=== ADVERSARIAL TESTS ===")
    results = {
        "clean": clean_metrics["accuracy"],
        "attacks": {}
    }
    
    for attack_name in attacks_to_run:
        print(f"\n--- {attack_name}: {ATTACK_CONFIGS[attack_name]['description']} ---")
        
        adv_images, labels = generate_adversarial_data(model, test_loader, attack_name, device, cache_dir)
        
        adv_dataset = TensorDataset(adv_images, labels)
        adv_loader = DataLoader(adv_dataset, batch_size=64, shuffle=False)
        
        adv_metrics = evaluate_model(model, adv_loader, device)
        drop = clean_metrics["accuracy"] - adv_metrics["accuracy"]
        
        results["attacks"][attack_name] = {
            "accuracy": adv_metrics["accuracy"],
            "drop": drop
        }
        
        print(f"  Accuracy: {adv_metrics['accuracy']:.2f}% (drop: {drop:.2f}%)")
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"{'Test':<15} {'Accuracy':>10} {'Drop':>10}")
    print("-" * 35)
    print(f"{'Clean':<15} {results['clean']:>9.2f}% {'--':>10}")
    for attack_name, metrics in results["attacks"].items():
        print(f"{attack_name:<15} {metrics['accuracy']:>9.2f}% {metrics['drop']:>9.2f}%")
    print("=" * 50)
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"benchmark_{timestamp}.json"
    
    results["timestamp"] = datetime.now().isoformat()
    results["epochs"] = epochs
    results["device"] = device
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Adversarial robustness benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--no-train", action="store_true", help="Skip training, use existing model")
    parser.add_argument("--fast", action="store_true", help="Only run FGSM and BIM (skip slow JSMA/CW)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached adversarial data")
    
    args = parser.parse_args()
    
    if args.clear_cache:
        import shutil
        cache_dir = Path("data/adversarial")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("Cleared adversarial cache")
    
    run_benchmark(
        epochs=args.epochs,
        device=args.device,
        train=not args.no_train,
        fast=args.fast
    )


if __name__ == "__main__":
    main()
