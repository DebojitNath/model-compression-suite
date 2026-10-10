"""
Dynamic parameter masking engine for unstructured magnitude pruning.
Pure PyTorch implementation using forward pre-hooks and backward gradient hooks.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class ParameterMask:
    """
    Manages a binary mask tensor for a specific named module parameter.
    Enforces W_eff = W * M on forward pass and grad_W = grad_W * M on backward pass.
    """
    def __init__(self, module: nn.Module, param_name: str = "weight"):
        self.module = module
        self.param_name = param_name
        orig_param = getattr(module, param_name)
        self.device = orig_param.device
        self.mask = torch.ones_like(orig_param.data, dtype=torch.float32, device=self.device)
        self._hook_handle = None
        self._grad_hook_handle = None
        self._register_hooks()

    def _register_hooks(self):
        # Forward pre-hook: ensures weights are masked before forward computation
        def forward_pre_hook(module, inputs):
            param = getattr(module, self.param_name, None)
            if isinstance(param, torch.Tensor) and hasattr(param, "data"):
                param.data.mul_(self.mask)

        # Backward hook on parameter tensor: ensures gradient for pruned weights is zeroed
        def grad_hook(grad):
            return grad * self.mask

        self._hook_handle = self.module.register_forward_pre_hook(forward_pre_hook)
        param = getattr(self.module, self.param_name, None)
        if isinstance(param, torch.Tensor) and param.requires_grad:
            self._grad_hook_handle = param.register_hook(grad_hook)

    def set_mask(self, new_mask: torch.Tensor):
        """Updates the binary mask and immediately applies it to the parameter."""
        self.mask = new_mask.to(dtype=torch.float32, device=self.device)
        param = getattr(self.module, self.param_name)
        param.data.mul_(self.mask)

    def remove(self):
        """Removes forward and backward hooks."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None


class MaskManager:
    """
    Coordinates ParameterMask instances across an entire model.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.masks: Dict[str, ParameterMask] = {}
        self._init_masks()

    def _init_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, "weight") and module.weight is not None:
                    self.masks[f"{name}.weight"] = ParameterMask(module, "weight")

    def apply_global_magnitude_pruning(self, target_sparsity: float) -> float:
        """
        Gathers all prunable weight tensors across the model, computes a global
        absolute magnitude threshold, and zeros out the bottom target_sparsity percentage.

        Args:
            target_sparsity: Fraction of total parameters to zero out (0.0 to 1.0)

        Returns:
            Actual achieved global sparsity percentage (0 to 100)
        """
        if not (0.0 <= target_sparsity < 1.0):
            raise ValueError(f"Target sparsity must be in [0.0, 1.0), got {target_sparsity}")

        if target_sparsity == 0.0:
            return 0.0

        all_weights = []
        for name, p_mask in self.masks.items():
            param = getattr(p_mask.module, p_mask.param_name)
            all_weights.append(param.data.abs().view(-1))

        flattened = torch.cat(all_weights)
        k = int(target_sparsity * flattened.numel())
        if k == 0:
            return 0.0

        # Find the kth smallest absolute magnitude
        threshold = torch.kthvalue(flattened, k).values.item()

        # Update binary masks
        total_elements = 0
        zero_elements = 0
        for name, p_mask in self.masks.items():
            param = getattr(p_mask.module, p_mask.param_name)
            # Binary mask: 1 where |w| > threshold, 0 otherwise
            new_mask = (param.data.abs() > threshold).float()
            p_mask.set_mask(new_mask)

            total_elements += new_mask.numel()
            zero_elements += int((new_mask == 0).sum().item())

        actual_sparsity = (zero_elements / total_elements) * 100.0 if total_elements > 0 else 0.0
        return actual_sparsity

    def apply_layerwise_magnitude_pruning(self, target_sparsity: float) -> float:
        """
        Applies magnitude pruning independently within each individual layer.
        """
        if not (0.0 <= target_sparsity < 1.0):
            raise ValueError(f"Target sparsity must be in [0.0, 1.0), got {target_sparsity}")

        total_elements = 0
        zero_elements = 0
        for name, p_mask in self.masks.items():
            param = getattr(p_mask.module, p_mask.param_name)
            flat = param.data.abs().view(-1)
            k = int(target_sparsity * flat.numel())
            if k > 0:
                thresh = torch.kthvalue(flat, k).values.item()
                new_mask = (param.data.abs() > thresh).float()
            else:
                new_mask = torch.ones_like(param.data)

            p_mask.set_mask(new_mask)
            total_elements += new_mask.numel()
            zero_elements += int((new_mask == 0).sum().item())

        actual_sparsity = (zero_elements / total_elements) * 100.0 if total_elements > 0 else 0.0
        return actual_sparsity

    def get_sparsity_breakdown(self) -> Dict[str, float]:
        """Returns per-layer sparsity percentages and overall sparsity."""
        breakdown = {}
        total_el = 0
        zero_el = 0
        for name, p_mask in self.masks.items():
            numel = p_mask.mask.numel()
            zeros = int((p_mask.mask == 0).sum().item())
            breakdown[name] = (zeros / numel) * 100.0 if numel > 0 else 0.0
            total_el += numel
            zero_el += zeros

        breakdown["overall"] = (zero_el / total_el) * 100.0 if total_el > 0 else 0.0
        return breakdown

    def remove_all(self):
        """Detaches all forward and backward hooks."""
        for p_mask in self.masks.values():
            p_mask.remove()

    def make_pruning_permanent(self):
        """Multiplies active masks into weights and detaches all forward/backward hooks."""
        for p_mask in self.masks.values():
            param = getattr(p_mask.module, p_mask.param_name, None)
            if isinstance(param, torch.Tensor) and hasattr(param, "data"):
                param.data.mul_(p_mask.mask)
            p_mask.remove()

