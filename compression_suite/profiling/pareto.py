"""
Multi-objective Pareto Frontier extraction and trade-off analysis algorithms.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


class ParetoFrontier:
    """
    Computes non-dominated sets across arbitrary trade-off metrics
    (e.g., maximizing accuracy while minimizing latency and model footprint).
    """

    @staticmethod
    def is_pareto_efficient_2d(
        x: np.ndarray,
        y: np.ndarray,
        minimize_x: bool = True,
        maximize_y: bool = True,
    ) -> np.ndarray:
        """
        Determines Pareto-efficient points in 2D space.
        Default: minimize x (e.g. latency/size), maximize y (e.g. accuracy).
        """
        n = len(x)
        is_efficient = np.ones(n, dtype=bool)

        # Standardize so both are minimized: lower is better
        x_norm = x if minimize_x else -x
        y_norm = -y if maximize_y else y

        for i in range(n):
            for j in range(n):
                if i != j and is_efficient[i]:
                    # Point j dominates point i if j is <= i in all and < i in at least one
                    if (x_norm[j] <= x_norm[i] and y_norm[j] <= y_norm[i]) and (
                        x_norm[j] < x_norm[i] or y_norm[j] < y_norm[i]
                    ):
                        is_efficient[i] = False
                        break

        return is_efficient

    @staticmethod
    def is_pareto_efficient_3d(
        costs: np.ndarray,
    ) -> np.ndarray:
        """
        Determines Pareto-efficient points in multi-dimensional space where lower is better.
        costs is (N, D) array.
        """
        n_points = costs.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_efficient[i]:
                # Find if any other point dominates point i
                dominates = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
                if np.any(dominates):
                    is_efficient[i] = False

        return is_efficient

    @classmethod
    def identify_pareto_points(cls, results: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Annotates a list of pipeline result dictionaries with Pareto optimality flags:
        - is_pareto_latency: 2D optimal across (latency, accuracy)
        - is_pareto_size: 2D optimal across (size, accuracy)
        - is_pareto_3d: 3D optimal across (latency, size, accuracy)
        """
        if not results:
            return results

        latencies = np.array([r["latency_p50_ms"] for r in results])
        sizes = np.array([r["disk_size_mb"] for r in results])
        accuracies = np.array([r["accuracy_top1"] for r in results])

        # 2D Latency vs Accuracy
        pareto_lat = cls.is_pareto_efficient_2d(latencies, accuracies, minimize_x=True, maximize_y=True)

        # 2D Size vs Accuracy
        pareto_size = cls.is_pareto_efficient_2d(sizes, accuracies, minimize_x=True, maximize_y=True)

        # 3D: [latency (min), size (min), -accuracy (min)]
        costs_3d = np.column_stack([latencies, sizes, -accuracies])
        pareto_3d = cls.is_pareto_efficient_3d(costs_3d)

        annotated = []
        for idx, res in enumerate(results):
            r = dict(res)
            r["is_pareto_latency"] = bool(pareto_lat[idx])
            r["is_pareto_size"] = bool(pareto_size[idx])
            r["is_pareto_3d"] = bool(pareto_3d[idx])
            annotated.append(r)

        return annotated
