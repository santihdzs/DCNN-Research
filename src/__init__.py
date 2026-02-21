"""
DCNN Research Project.

Undergraduate research on DCNN robustness evaluation against adversarial attacks.
"""

from .models import MNISTNet, get_model, count_parameters
from .pipeline import train, test
from .utils import get_mnist_loaders, get_data_info

__version__ = "0.1.0"

__all__ = [
    "MNISTNet",
    "get_model", 
    "count_parameters",
    "train",
    "test",
    "get_mnist_loaders",
    "get_data_info"
]
