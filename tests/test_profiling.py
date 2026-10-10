"""
Unit tests for ModelProfiler (accuracy, latency, disk size, memory).
"""
import torch
import pytest
from compression_suite.profiling.metrics import ModelProfiler
from compression_suite.models.registry import build_model
from compression_suite.data.datasets import get_dataloaders


def test_profiler_disk_size_and_params():
    model = build_model("fast_convnet_student", num_classes=10)
    profiler = ModelProfiler(device=torch.device("cpu"))

    size_mb = profiler.measure_disk_size_mb(model)
    assert size_mb > 0.05
    assert size_mb < 20.0

    counts = profiler.measure_parameter_counts(model)
    assert counts["total_params"] > 50000
    assert counts["sparsity_pct"] < 1.0


def test_profiler_latency_distribution():
    model = build_model("fast_convnet_student", num_classes=10)
    profiler = ModelProfiler(device=torch.device("cpu"), warmup_iterations=3, benchmark_iterations=10)

    inp = torch.randn(1, 3, 32, 32)
    lat = profiler.measure_latency(model, inp)

    assert "latency_p50_ms" in lat
    assert "latency_p90_ms" in lat
    assert "latency_p99_ms" in lat
    assert lat["latency_p50_ms"] > 0.0
    assert lat["latency_p99_ms"] >= lat["latency_p50_ms"]
    assert lat["throughput_samples_per_sec"] > 0.0
