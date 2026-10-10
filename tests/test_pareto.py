"""
Unit tests for ParetoFrontier non-dominated filtering in 2D and 3D.
"""
import numpy as np
import pytest
from compression_suite.profiling.pareto import ParetoFrontier


def test_pareto_2d_extraction():
    # Latency (minimize) vs Accuracy (maximize)
    # Point A: Latency=10ms, Acc=90% -> Optimal
    # Point B: Latency=5ms, Acc=85%  -> Optimal (faster, slightly lower acc)
    # Point C: Latency=12ms, Acc=80% -> Dominated by A (slower AND lower acc)
    x = np.array([10.0, 5.0, 12.0])
    y = np.array([90.0, 85.0, 80.0])

    is_optimal = ParetoFrontier.is_pareto_efficient_2d(x, y, minimize_x=True, maximize_y=True)

    assert is_optimal[0] == True   # Point A
    assert is_optimal[1] == True   # Point B
    assert is_optimal[2] == False  # Point C is dominated


def test_pareto_annotation_on_results():
    results = [
        {"pipeline": "A", "latency_p50_ms": 10.0, "disk_size_mb": 50.0, "accuracy_top1": 92.0},
        {"pipeline": "B", "latency_p50_ms": 4.0,  "disk_size_mb": 10.0, "accuracy_top1": 88.0},
        {"pipeline": "C", "latency_p50_ms": 12.0, "disk_size_mb": 60.0, "accuracy_top1": 80.0},
    ]

    annotated = ParetoFrontier.identify_pareto_points(results)

    assert annotated[0]["is_pareto_latency"] == True
    assert annotated[1]["is_pareto_latency"] == True
    assert annotated[2]["is_pareto_latency"] == False
    assert annotated[2]["is_pareto_3d"] == False
