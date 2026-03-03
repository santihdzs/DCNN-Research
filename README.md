# DCNN Research

Undergraduate research project at ITESM focused on robustness evaluation of Deep Convolutional Neural Networks (DCNNs) against adversarial attacks.

## Overview

This repository contains the experimental framework for evaluating DCNN robustness. The goal is to develop a clean, automated pipeline for training models and testing their resilience against adversarial perturbations.

## Project Structure

```
DCNN-Research/
├── src/
│   ├── models/         # Model architectures
│   ├── pipeline/       # Training & testing pipelines
│   └── utils/          # Data loading & utilities
├── data/               # MNIST and adversarial datasets
├── checkpoints/        # Saved model checkpoints
├── results/            # Evaluation results
├── run_config.json     # Benchmark configuration
└── README.md
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync
```

## Usage

### Quick Start

```bash
# Run full benchmark (train + test all attacks)
uv run python -m src.pipeline.benchmark
```

### Benchmark Commands

```bash
# Run benchmark with existing model (skip training)
uv run python -m src.pipeline.benchmark --no-train

# Run only fast attacks (FGSM + BIM, skip slow JSMA/CW)
uv run python -m src.pipeline.benchmark --no-train --fast

# Clear cached adversarial data and regenerate
uv run python -m src.pipeline.benchmark --clear-cache

# Force CPU
uv run python -m src.pipeline.benchmark --device cpu
```

### Advanced Options

#### Configuration File

All benchmark options can be set in `run_config.json`:

```json
{
  "deterministic": false,
  "seed": 42,
  "multi_seed": false,
  "num_seeds": 3,
  "epsilon_sweep": false,
  "epsilon_values": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
  "per_class": false,
  "confusion_matrices": false,
  "runtime_benchmark": true,
  "confidence_analysis": false
}
```

#### CLI Flags

```bash
# Reproducibility
uv run python -m src.pipeline.benchmark --seed 42 --deterministic

# Evaluation features
uv run python -m src.pipeline.benchmark --epsilon-sweep    # Run at multiple epsilon values
uv run python -m src.pipeline.benchmark --per-class        # Per-class robustness metrics
uv run python -m src.pipeline.benchmark --confidence-analysis  # Confidence drop analysis

# Override config settings
uv run python -m src.pipeline.benchmark --config custom_config.json
uv run python -m src.pipeline.benchmark --deterministic --per-class
```

### Flags

| Flag | Description |
|------|-------------|
| `--no-train` | Skip training, use existing model in `checkpoints/` |
| `--epochs N` | Number of training epochs (default: 3) |
| `--fast` | Only run FGSM and BIM attacks |
| `--clear-cache` | Delete cached adversarial data and regenerate |
| `--device cpu` | Force CPU (auto-detects CUDA by default) |
| `--seed N` | Set random seed (overrides config) |
| `--deterministic` | Enable deterministic mode for reproducibility |
| `--epsilon-sweep` | Run attacks at multiple epsilon values |
| `--per-class` | Compute per-class robustness metrics |
| `--confidence-analysis` | Compute confidence drop metrics |
| `--runtime-benchmark` | Measure attack runtime |
| `--config FILE` | Path to config JSON file |

### Benchmark Output

The benchmark outputs JSON files with comprehensive metrics:

```json
{
  "clean": 99.03,
  "attacks": {
    "FGSM": {
      "accuracy": 97.31,
      "drop": 1.72,
      "metrics": {
        "clean_accuracy": 99.03,
        "adversarial_accuracy": 97.31,
        "attack_success_rate": 0.017
      },
      "perturbation_stats": {
        "linf_mean": 1.82,
        "l2_mean": 19.09
      },
      "confidence_analysis": {
        "clean_true_conf_mean": 0.992,
        "adv_true_conf_mean": 0.800,
        "mean_conf_drop": 0.192
      },
      "per_class_metrics": {
        "9": {"clean_acc": 0.967, "adv_acc": 0.897, "asr": 0.073}
      }
    }
  },
  "execution": {
    "timestamp": "2026-03-03T06:00:00",
    "device": "cpu",
    "batch_size": 128,
    "num_samples_evaluated": 10000
  },
  "versioning": {
    "git_commit": "abc123..."
  },
  "reproducibility": {
    "seed": 42,
    "deterministic": false
  }
}
```

### Cached Data

Adversarial examples are cached in `data/adversarial/` for faster runs:
- `FGSM_test.pt`, `BIM_test.pt`, `JSMA_test.pt`, `CW_test.pt`
- Epsilon sweep: `FGSM_eps0.1_test.pt`, etc.

## Metrics Explained

- **ASR (Attack Success Rate)**: % of originally-correct samples that become misclassified
- **Clean Accuracy**: Accuracy on unmodified test data
- **Adversarial Accuracy**: Accuracy on attacked test data
- **Perturbation Stats**: L∞ and L2 norms of adversarial perturbations
- **Confidence Drop**: How much the model's confidence decreases under attack

## Training

```python
from src.models import get_model
from src.utils import get_mnist_loaders

train_loader, test_loader = get_mnist_loaders(batch_size=64)
model = get_model(device="cpu")
```

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- torchattacks
- See `pyproject.toml` for full dependencies

## Author

Santiago Hernández Senosiain — Undergraduate Researcher, ITESM
