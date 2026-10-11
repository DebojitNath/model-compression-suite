"""
Data loading and benchmark dataset module.
"""
from compression_suite.data.datasets import (
    SyntheticVisionDataset,
    SyntheticTextDataset,
    get_dataloaders,
)

__all__ = [
    "SyntheticVisionDataset",
    "SyntheticTextDataset",
    "get_dataloaders",
]
