"""
Base compressible model definitions and utility interfaces.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class BaseCompressibleModel(nn.Module):
    """
    Abstract base class providing standard reflection and introspection
    interfaces required for pruning, quantization, and profiling.
    """

    def get_prunable_modules(self) -> Dict[str, nn.Module]:
        """
        Returns a dictionary mapping module names to modules that contain
        prunable weight tensors (e.g., nn.Conv2d, nn.Linear).
        """
        prunable = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable[name] = module
        return prunable

    def get_prunable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary mapping parameter names to weight tensors
        that are eligible for magnitude pruning.
        """
        params = {}
        for mod_name, module in self.get_prunable_modules().items():
            if hasattr(module, "weight") and module.weight is not None:
                params[f"{mod_name}.weight"] = module.weight
        return params

    def count_parameters(self) -> Tuple[int, int, float]:
        """
        Calculates total parameters, active (non-zero) parameters, and sparsity percentage.

        Returns:
            Tuple of (total_params, active_params, sparsity_ratio)
        """
        total = 0
        active = 0
        for p in self.parameters():
            numel = p.numel()
            total += numel
            active += int((p != 0).sum().item())
        sparsity = (1.0 - (active / total)) * 100.0 if total > 0 else 0.0
        return total, active, sparsity

    def get_model_size_mb(self) -> float:
        """
        Computes the theoretical in-memory parameter footprint in Megabytes (MB).
        """
        param_bytes = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_bytes + buffer_bytes) / (1024.0 * 1024.0)
