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
