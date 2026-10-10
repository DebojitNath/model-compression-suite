"""
Statistical Calibration Engine for Post-Training Quantization.
Implements Min-Max and Percentile (Outlier-Clipping) Calibration.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CalibrationEngine:
    """
    Computes optimal scaling factors (s) and zero-points (z) for INT8 quantization
    under symmetric and asymmetric uniform affine schemes.
    """
    def __init__(
        self,
        method: str = "percentile",
        percentile: float = 99.99,
        symmetric: bool = True,
        q_min: int = -128,
        q_max: int = 127,
    ):
        self.method = method.lower()
        self.percentile = float(percentile)
        self.symmetric = bool(symmetric)
        self.q_min = int(q_min)
        self.q_max = int(q_max)

    def calculate_scale_and_zero_point(self, tensor: torch.Tensor) -> Tuple[float, int]:
        """
        Calculates scale (s) and zero-point (z) for a given tensor.
        """
        flat = tensor.detach().view(-1).float()

        if flat.numel() == 0 or flat.max() == flat.min():
            return 1.0, 0

        if self.method == "percentile":
            # If tensor is large, uniformly subsample up to 50,000 elements for fast and robust quantile estimation
            if flat.numel() > 50000:
                indices = torch.randint(0, flat.numel(), (50000,), dtype=torch.long)
                indices = torch.clamp(indices, 0, flat.numel() - 1)
                sample_flat = flat[indices]
            else:
                sample_flat = flat

            # Clip outliers to preserve dynamic range on sparse weights/activations
            alpha = (100.0 - self.percentile) / 2.0
            lower_pct = alpha
            upper_pct = 100.0 - alpha

            q_low = torch.quantile(sample_flat, lower_pct / 100.0).item()
            q_high = torch.quantile(sample_flat, upper_pct / 100.0).item()
            min_val, max_val = q_low, q_high
        else:  # "minmax"
            min_val = flat.min().item()
            max_val = flat.max().item()

        if self.symmetric:
            max_abs = max(abs(min_val), abs(max_val))
            max_abs = max(max_abs, 1e-8)
            # Symmetric mapping: zero_point = 0
            scale = max_abs / self.q_max
            zero_point = 0
        else:
            # Asymmetric mapping
            rng = max(max_val - min_val, 1e-8)
            scale = rng / (self.q_max - self.q_min)
            zero_point = round(-min_val / scale) + self.q_min
            zero_point = max(self.q_min, min(self.q_max, zero_point))

        return float(scale), int(zero_point)

    def calibrate_activations(
        self,
        model: nn.Module,
        calib_loader: DataLoader,
        device: torch.device,
        max_batches: int = 8,
    ) -> Dict[str, Tuple[float, int]]:
        """
        Runs a forward calibration pass over calibration data, recording activation ranges.
        """
        activation_stats: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def get_hook(name: str):
            def hook(module, inputs, outputs):
                if name not in activation_stats:
                    activation_stats[name] = []
                if isinstance(outputs, torch.Tensor):
                    activation_stats[name].append(outputs.detach().cpu())
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hooks.append(module.register_forward_hook(get_hook(name)))

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= max_batches:
                    break
                inputs = batch[0].to(device)
                model(inputs)

        for h in hooks:
            h.remove()

        scales_and_zeros: Dict[str, Tuple[float, int]] = {}
        for name, tensors in activation_stats.items():
            if tensors:
                combined = torch.cat([t.view(-1) for t in tensors])
                scales_and_zeros[name] = self.calculate_scale_and_zero_point(combined)

        return scales_and_zeros
