"""
Post-Training Quantization (PTQ) Engine.
Implements simulated affine INT8 quantization and native PyTorch dynamic quantization.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from compression_suite.quantization.calibration import CalibrationEngine


def quantize_tensor(
    tensor: torch.Tensor,
    scale: float,
    zero_point: int,
    q_min: int = -128,
    q_max: int = 127,
) -> torch.Tensor:
    """
    Uniform affine quantization:
    q = clip(round(x / s) + z, q_min, q_max)
    """
    scaled = (tensor / scale) + zero_point
    rounded = torch.round(scaled)
    clipped = torch.clamp(rounded, q_min, q_max)
    return clipped.to(torch.int8)


def dequantize_tensor(
    q_tensor: torch.Tensor,
    scale: float,
    zero_point: int,
) -> torch.Tensor:
    """
    Uniform affine dequantization:
    x_hat = s * (q - z)
    """
    return (q_tensor.float() - zero_point) * scale


def compute_sqnr(orig: torch.Tensor, quant: torch.Tensor) -> float:
    """
    Calculates Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB).
    Higher is better (less precision distortion).
    """
    signal_power = torch.mean(orig.float() ** 2).item()
    noise_power = torch.mean((orig.float() - quant.float()) ** 2).item()
    if noise_power <= 1e-12:
        return 100.0  # Near-lossless
    if signal_power <= 1e-12:
        return 0.0
    return 10.0 * math.log10(signal_power / noise_power)


class SimulatedPTQModel(nn.Module):
    """
    Simulates INT8 arithmetic on weights by replacing FP32 weights with
    dequantized INT8 representations (x_hat = s * (q - z)).
    Maintains exact FP32 tensor graphs for execution while reflecting true INT8 precision loss.
    """
    def __init__(self, base_model: nn.Module, calibrator: CalibrationEngine):
        super().__init__()
        self.model = copy.deepcopy(base_model)
        self.calibrator = calibrator
        self.layer_sqnr: Dict[str, float] = {}
        self.layer_scales: Dict[str, Tuple[float, int]] = {}
        self._quantize_weights()

    def _quantize_weights(self):
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if hasattr(module, "weight") and module.weight is not None:
                        orig = module.weight.data
                        s, z = self.calibrator.calculate_scale_and_zero_point(orig)
                        q = quantize_tensor(orig, s, z, self.calibrator.q_min, self.calibrator.q_max)
                        dequant = dequantize_tensor(q, s, z)

                        # Record SQNR (dB)
                        sqnr = compute_sqnr(orig, dequant)
                        self.layer_sqnr[f"{name}.weight"] = sqnr
                        self.layer_scales[f"{name}.weight"] = (s, z)

                        # Replace weight with simulated quantized tensor
                        module.weight.data.copy_(dequant)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)


class PostTrainingQuantizer:
    """
    Orchestrates Post-Training Quantization (PTQ) via statistical calibration,
    simulated INT8 verification, and native PyTorch dynamic CPU quantization.
    """
    def __init__(
        self,
        method: str = "percentile",
        percentile: float = 99.99,
        symmetric: bool = True,
        dtype: str = "int8",
    ):
        self.method = method
        self.percentile = percentile
        self.symmetric = symmetric
        self.dtype = dtype.lower()
        self.calibrator = CalibrationEngine(
            method=method,
            percentile=percentile,
            symmetric=symmetric,
            q_min=-128 if dtype == "int8" else 0,
            q_max=127 if dtype == "int8" else 255,
        )

    def quantize(
        self,
        model: nn.Module,
        calib_loader: Optional[DataLoader] = None,
        use_native_torch: bool = True,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        """
        Applies Post-Training Quantization to the model.

        Args:
            model: PyTorch model in FP32
            calib_loader: Optional DataLoader for activation calibration
            use_native_torch: If True, attempts PyTorch native dynamic quantization for real CPU speedup
            device: Target evaluation device (e.g. cpu or cuda)

        Returns:
            Quantized nn.Module
        """
        cpu_model = copy.deepcopy(model).cpu().eval()

        if self.dtype == "fp16":
            # Direct Half Precision
            return cpu_model.half()

        # Step 1: Calibrate activations if dataloader provided
        if calib_loader is not None:
            self.calibrator.calibrate_activations(
                cpu_model,
                calib_loader,
                device=torch.device("cpu"),
            )

        # Target execution device
        target_device = device
        if target_device is None:
            target_device = (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if any(p.is_cuda for p in model.parameters())
                else torch.device("cpu")
            )

        # Step 2: Native PyTorch Dynamic Quantization for Real Speedup (CPU only)
        # Note: PyTorch's native dynamic quantization is strictly CPU-only.
        # CUDA execution requires simulated PTQ or specialized TensorRT/cuDNN kernels.
        if use_native_torch and target_device.type == "cpu":
            try:
                # Quantize Linear modules to torch.qint8
                quantized_native = torch.ao.quantization.quantize_dynamic(
                    cpu_model,
                    {nn.Linear},
                    dtype=torch.qint8,
                )
                # Verify that the quantized model actually executes forward without internal PyTorch bugs
                # (e.g. PyTorch Transformer fastpath issues with dynamic quantized MultiheadAttention)
                if calib_loader is not None:
                    for batch in calib_loader:
                        inp = batch[0] if isinstance(batch, (list, tuple)) else batch
                        _ = quantized_native(inp[:1].to(torch.device("cpu")))
                        break
                return quantized_native
            except Exception:
                # Fallback to simulated INT8 if native quantization is unsupported or fails forward pass
                pass

        # Simulated INT8 model (robust on both CPU and CUDA)
        return SimulatedPTQModel(cpu_model, self.calibrator)
