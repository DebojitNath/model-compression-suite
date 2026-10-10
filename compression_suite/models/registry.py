"""
Pluggable Model Registry and Factory for Model Compression Suite.
"""
from __future__ import annotations
from typing import Callable, Dict, List
import torch.nn as nn

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """
    Decorator to register a model architecture builder in the global registry.
    
    Usage:
        @register_model("resnet18")
        def build_resnet18(num_classes=10, **kwargs):
            return ResNet18(num_classes=num_classes)
    """
    def decorator(fn: Callable[..., nn.Module]):
        normalized_name = name.lower().strip()
        if normalized_name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{normalized_name}' is already registered.")
        _MODEL_REGISTRY[normalized_name] = fn
        return fn
    return decorator


def build_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Instantiates a registered model by name.

    Args:
        name: Name of registered model architecture
        num_classes: Target number of output classes
        kwargs: Additional arguments passed to the model factory

    Returns:
        Instantiated nn.Module
    """
    normalized_name = name.lower().strip()
    if normalized_name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model architecture '{name}'. Available registered models: [{available}]"
        )
    return _MODEL_REGISTRY[normalized_name](num_classes=num_classes, **kwargs)


def list_available_models() -> List[str]:
    """Returns sorted list of registered model names."""
    return sorted(list(_MODEL_REGISTRY.keys()))
