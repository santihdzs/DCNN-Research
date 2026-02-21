"""
Testing pipeline for model evaluation.

Provides utilities for:
- Automated model testing
- Metrics computation
- Adversarial attack testing (placeholder for future)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    save_results: bool = True,
    results_dir: str = "results"
):
    """
    Run standard test evaluation on the model.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to test on
        save_results: Whether to save results to file
        results_dir: Directory to save results
    
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    correct = 0
    total = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class accuracy
            for pred, true in zip(predicted, target):
                per_class_total[true.item()] += 1
                if pred.item() == true.item():
                    per_class_correct[true.item()] += 1
    
    accuracy = 100. * correct / total
    per_class_accuracy = {
        f"class_{i}": {
            "correct": per_class_correct[i],
            "total": per_class_total[i],
            "accuracy": 100. * per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
        }
        for i in range(10)
    }
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_accuracy
    }
    
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        filename = results_path / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {filename}")
    
    return results


def run_full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    save_results: bool = True,
    results_dir: str = "results"
):
    """
    Run full evaluation suite on the model.
    
    Currently runs standard test. More tests can be added here
    (adversarial robustness, calibration, etc.)
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to test on
        save_results: Whether to save results
        results_dir: Directory to save results
    
    Returns:
        Dictionary with all evaluation results
    """
    print("Running standard evaluation...")
    standard_results = test_model(model, test_loader, device, save_results, results_dir)
    
    # Future: Add adversarial robustness testing
    # print("Running adversarial robustness tests...")
    # adversarial_results = test_adversarial(model, test_loader, device)
    
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "standard": standard_results,
        # "adversarial": adversarial_results  # Future
    }
    
    return full_results


def compare_models(
    results_files: list,
    metrics: list = ["accuracy"]
):
    """
    Compare multiple model evaluation results.
    
    Args:
        results_files: List of paths to result JSON files
        metrics: List of metrics to compare
    
    Returns:
        Comparison results
    """
    comparison = {}
    
    for filepath in results_files:
        with open(filepath, "r") as f:
            results = json.load(f)
        
        model_name = Path(filepath).stem
        comparison[model_name] = {}
        
        for metric in metrics:
            if metric in results:
                comparison[model_name][metric] = results[metric]
    
    return comparison
