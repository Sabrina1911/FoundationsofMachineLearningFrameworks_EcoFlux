"""
EcoFlux src package.

This package contains the core Python modules for the EcoFlux project:
- Synthetic dataset generation
- Data loading utilities
- Model training helpers
"""

from .generate_data import generate_synthetic_energy_data
from .data_prep import load_energy_dataset

__all__ = [
    "generate_synthetic_energy_data",
    "load_energy_dataset",
]
