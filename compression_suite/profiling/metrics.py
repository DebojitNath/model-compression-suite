"""
Hardware and performance profiling metrics suite.
Measures Top-1/Top-5 accuracy, disk size, peak memory, and P50/P90/P99 latency.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import os
import io
import time
import tempfile
import tracemalloc
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from compression_suite.distillation.trainer import evaluate_accuracy


class ModelProfiler:
    """
    Unified profiler evaluating task accuracy, model artifact footprint,
    peak RAM/VRAM consumption, and latency distributions (P50, P90, P99).
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        warmup_iterations: int = 15,
        benchmark_iterations: int = 60,
    ):
        self.device = device or torch.device("cpu")
        self.warmup_iterations = max(2, int(warmup_iterations))
        self.benchmark_iterations = max(5, int(benchmark_iterations))

    def measure_accuracy(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Measures Top-1 classification accuracy and evaluation loss."""
        model = model.to(self.device).eval()
        return evaluate_accuracy(model, test_loader, self.device)

    def measure_disk_size_mb(self, model: nn.Module) -> float:
        """
        Measures real on-disk artifact size by serializing the state_dict to a temp file.
        """
        model = model.cpu()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torch.save(model.state_dict(), tmp_path)
            size_bytes = os.path.getsize(tmp_path)
            size_mb = size_bytes / (1024.0 * 1024.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return float(round(size_mb, 4))

    def measure_parameter_counts(self, model: nn.Module) -> Dict[str, float]:
        """Calculates total parameters, active non-zero parameters, and overall sparsity."""
        total = 0
        active = 0
        for p in model.parameters():
            numel = p.numel()
            total += numel
            active += int((p != 0).sum().item())

        sparsity = (1.0 - (active / total)) * 100.0 if total > 0 else 0.0
        return {
            "total_params": total,
            "active_params": active,
            "sparsity_pct": float(round(sparsity, 2)),
        }

    def measure_latency(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Profiles inference latency over multiple iterations using high-resolution timers.
        Reports P50 (median), P90, P99, mean, min, max, and throughput.
        """
        model = model.to(self.device).eval()
        inp = sample_input.to(self.device)
        is_cuda = self.device.type == "cuda"

        # Warmup passes
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = model(inp)
                if is_cuda:
                    torch.cuda.synchronize()

        # Benchmark passes
        latencies_ms: List[float] = []
        with torch.no_grad():
            for _ in range(self.benchmark_iterations):
                t0 = time.perf_counter()
                _ = model(inp)
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        arr = np.array(latencies_ms)
        batch_size = sample_input.size(0)
        mean_lat = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))
        throughput = (batch_size / (mean_lat / 1000.0)) if mean_lat > 0 else 0.0

        return {
            "latency_p50_ms": float(round(p50, 3)),
            "latency_p90_ms": float(round(p90, 3)),
            "latency_p99_ms": float(round(p99, 3)),
            "latency_mean_ms": float(round(mean_lat, 3)),
            "throughput_samples_per_sec": float(round(throughput, 1)),
        }

    def measure_peak_memory_mb(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measures peak RAM and VRAM during inference execution.
        """
        model = model.to(self.device).eval()
        inp = sample_input.to(self.device)

        tracemalloc.start()
        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024.0 * 1024.0)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(inp)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        ram_after = process.memory_info().rss / (1024.0 * 1024.0)
        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_ram_mb = max(ram_after, ram_before + (peak_traced / (1024.0 * 1024.0)))

        vram_mb = 0.0
        if self.device.type == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        return {
            "peak_ram_mb": float(round(peak_ram_mb, 2)),
            "peak_vram_mb": float(round(vram_mb, 2)),
        }

    def profile_full(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        sample_input: torch.Tensor,
        pipeline_name: str = "model",
    ) -> Dict[str, any]:
        """
        Runs comprehensive profile across all dimensions.
        """
        acc_stats = self.measure_accuracy(model, test_loader)
        disk_size = self.measure_disk_size_mb(model)
        param_stats = self.measure_parameter_counts(model)
        latency_stats = self.measure_latency(model, sample_input)
        mem_stats = self.measure_peak_memory_mb(model, sample_input)

        profile = {
            "pipeline": pipeline_name,
            "accuracy_top1": acc_stats["accuracy_top1"],
            "eval_loss": acc_stats["loss"],
            "disk_size_mb": disk_size,
            "total_params": param_stats["total_params"],
            "active_params": param_stats["active_params"],
            "sparsity_pct": param_stats["sparsity_pct"],
            "latency_p50_ms": latency_stats["latency_p50_ms"],
            "latency_p90_ms": latency_stats["latency_p90_ms"],
            "latency_p99_ms": latency_stats["latency_p99_ms"],
            "throughput": latency_stats["throughput_samples_per_sec"],
            "peak_ram_mb": mem_stats["peak_ram_mb"],
            "peak_vram_mb": mem_stats["peak_vram_mb"],
        }
        return profile
