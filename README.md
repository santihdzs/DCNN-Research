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
├── data/               # MNIST and other datasets
├── checkpoints/        # Saved model checkpoints
├── results/            # Evaluation results
└── README.md
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Install in editable mode for development
uv pip install -e .
```

## Usage

### Quick Start

```bash
# Install dependencies
uv sync

# Run full benchmark (train + test all attacks)
uv run python -m src.pipeline.benchmark
```

### Benchmark Commands

```bash
# Run benchmark with existing model (skip training)
uv run python -m src.pipeline.benchmark --no-train

# Run with specific number of epochs
uv run python -m src.pipeline.benchmark --epochs 5

# Run only fast attacks (FGSM + BIM, skip slow JSMA/CW)
uv run python -m src.pipeline.benchmark --no-train --fast

# Clear cached adversarial data and regenerate
uv run python -m src.pipeline.benchmark --no-train --clear-cache

# Force CPU (instead of auto-detecting CUDA)
uv run python -m src.pipeline.benchmark --device cpu
```

### Flags

| Flag | Description |
|------|-------------|
| `--no-train` | Skip training, use existing model in `checkpoints/` |
| `--epochs N` | Number of training epochs (default: 3) |
| `--fast` | Only run FGSM and BIM attacks |
| `--clear-cache` | Delete cached adversarial data and regenerate |
| `--device cpu` | Force CPU (auto-detects CUDA by default) |

### Benchmark Output

The benchmark outputs:
- Clean test accuracy
- Accuracy for each attack (FGSM, BIM, JSMA, CW)
- Drop from clean accuracy
- Results saved to `results/benchmark_TIMESTAMP.json`

Cached adversarial data is stored in `data/adversarial/` for faster future runs.

### Training

```python
from src.models import get_model
from src.pipeline import get_optimizer, get_criterion, train_epoch
from src.utils import get_mnist_loaders

# Load data
train_loader, test_loader = get_mnist_loaders(batch_size=64)

# Create model
model = get_model(device="cuda" if torch.cuda.is_available() else "cpu")

# Training loop
optimizer = get_optimizer(model, optimizer="adam", lr=1e-3)
criterion = get_criterion()

for epoch in range(10):
    loss, accuracy = train_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
```

### Testing

```python
from src.pipeline import run_full_evaluation

# Run full evaluation suite
results = run_full_evaluation(model, test_loader)
print(f"Test Accuracy: {results['standard']['accuracy']:.2f}%")
```

## Pipeline

The project is organized as a pipeline:

1. **Models** (`src/models/`) — Model architectures for experimentation
2. **Training** (`src/pipeline/train.py`) — Training loops, checkpoints, optimization
3. **Testing** (`src/pipeline/test.py`) — Automated evaluation, metrics, result tracking
4. **Saved Models/Stats** — Checkpoints and results saved for analysis

## Current Status

- Basic MNIST CNN model implemented
- Training pipelines in place
- Adversarial Attack testing pipelines in place (FGSM, BIM, JSMA, CW)

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- See `pyproject.toml` for full dependencies

## Author

Santiago Hernández Senosiain — Undergraduate Researcher, ITESM
