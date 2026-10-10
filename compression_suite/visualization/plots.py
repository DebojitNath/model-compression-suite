"""
Publication-grade Pareto visualization suite using Matplotlib and Seaborn.
Generates 2D Pareto frontiers, 3D projections, and comprehensive compression breakdown charts.
"""
from __future__ import annotations
from typing import Dict, List
import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set unified aesthetic theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})


def plot_accuracy_vs_latency_pareto(results: List[Dict[str, any]], output_path: str):
    """
    Plots 2D Task Accuracy (%) vs CPU Latency P50 (ms), tracing the Pareto-optimal frontier.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    latencies = [r["latency_p50_ms"] for r in results]
    accuracies = [r["accuracy_top1"] for r in results]
    labels = [r["pipeline"] for r in results]
    pareto_flags = [r.get("is_pareto_latency", False) for r in results]

    # Non-Pareto points
    non_pareto_x = [latencies[i] for i in range(len(results)) if not pareto_flags[i]]
    non_pareto_y = [accuracies[i] for i in range(len(results)) if not pareto_flags[i]]
    ax.scatter(non_pareto_x, non_pareto_y, color="#94a3b8", s=100, alpha=0.7, edgecolors="none", label="Sub-optimal Pipelines")

    # Pareto-optimal points
    pareto_indices = [i for i in range(len(results)) if pareto_flags[i]]
    # Sort by latency for frontier line
    pareto_indices.sort(key=lambda idx: latencies[idx])
    pareto_x = [latencies[i] for i in pareto_indices]
    pareto_y = [accuracies[i] for i in pareto_indices]

    ax.scatter(pareto_x, pareto_y, color="#2563eb", s=160, edgecolors="#1e3a8a", linewidths=1.5, zorder=5, label="Pareto-Optimal Frontier")
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, color="#2563eb", linestyle="--", linewidth=2.0, alpha=0.85, zorder=4)

    # Label key points
    for i, label in enumerate(labels):
        offset = (8, 4) if pareto_flags[i] else (6, -10)
        color = "#0f172a" if pareto_flags[i] else "#64748b"
        weight = "bold" if pareto_flags[i] else "normal"
        ax.annotate(
            label.replace("_", " "),
            (latencies[i], accuracies[i]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight=weight,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e2e8f0", alpha=0.8)
        )

    ax.set_title("Pareto Frontier: Task Accuracy vs. Inference Latency (P50)", pad=15, fontweight="bold")
    ax.set_xlabel("P50 Inference Latency (ms) — Lower is Better")
    ax.set_ylabel("Top-1 Accuracy (%) — Higher is Better")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_size_pareto(results: List[Dict[str, any]], output_path: str):
    """
    Plots 2D Task Accuracy (%) vs Model Artifact Footprint (MB).
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sizes = [r["disk_size_mb"] for r in results]
    accuracies = [r["accuracy_top1"] for r in results]
    labels = [r["pipeline"] for r in results]
    pareto_flags = [r.get("is_pareto_size", False) for r in results]

    non_pareto_x = [sizes[i] for i in range(len(results)) if not pareto_flags[i]]
    non_pareto_y = [accuracies[i] for i in range(len(results)) if not pareto_flags[i]]
    ax.scatter(non_pareto_x, non_pareto_y, color="#94a3b8", s=100, alpha=0.7, edgecolors="none", label="Sub-optimal Pipelines")

    pareto_indices = [i for i in range(len(results)) if pareto_flags[i]]
    pareto_indices.sort(key=lambda idx: sizes[idx])
    pareto_x = [sizes[i] for i in pareto_indices]
    pareto_y = [accuracies[i] for i in pareto_indices]

    ax.scatter(pareto_x, pareto_y, color="#059669", s=160, edgecolors="#064e3b", linewidths=1.5, zorder=5, label="Pareto-Optimal Frontier")
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, color="#059669", linestyle="--", linewidth=2.0, alpha=0.85, zorder=4)

    for i, label in enumerate(labels):
        offset = (8, 4) if pareto_flags[i] else (6, -10)
        color = "#0f172a" if pareto_flags[i] else "#64748b"
        weight = "bold" if pareto_flags[i] else "normal"
        ax.annotate(
            label.replace("_", " "),
            (sizes[i], accuracies[i]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            fontweight=weight,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e2e8f0", alpha=0.8)
        )

    ax.set_title("Pareto Frontier: Task Accuracy vs. Artifact Disk Size", pad=15, fontweight="bold")
    ax.set_xlabel("Artifact Disk Footprint (MB) — Lower is Better")
    ax.set_ylabel("Top-1 Accuracy (%) — Higher is Better")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_3d_pareto_frontier(results: List[Dict[str, any]], output_path: str):
    """
    Plots 3D Pareto space: Latency (X) x Model Size (Y) x Accuracy (Z).
    """
    fig = plt.figure(figsize=(11, 8), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    latencies = np.array([r["latency_p50_ms"] for r in results])
    sizes = np.array([r["disk_size_mb"] for r in results])
    accuracies = np.array([r["accuracy_top1"] for r in results])
    pareto_flags = np.array([r.get("is_pareto_3d", False) for r in results])

    # Plot sub-optimal points
    sub_mask = ~pareto_flags
    ax.scatter(
        latencies[sub_mask], sizes[sub_mask], accuracies[sub_mask],
        color="#94a3b8", s=80, alpha=0.6, label="Sub-optimal"
    )

    # Plot 3D Pareto optimal points
    p_mask = pareto_flags
    scatter = ax.scatter(
        latencies[p_mask], sizes[p_mask], accuracies[p_mask],
        c=accuracies[p_mask], cmap="viridis", s=180, edgecolors="#1e293b",
        linewidths=1.5, depthshade=False, label="3D Pareto-Optimal"
    )

    # Add text labels for Pareto points
    for i, r in enumerate(results):
        if r.get("is_pareto_3d", False):
            ax.text(
                r["latency_p50_ms"],
                r["disk_size_mb"],
                r["accuracy_top1"] + 0.5,
                r["pipeline"].replace("_", " "),
                fontsize=8,
                fontweight="bold"
            )

    ax.set_title("3D Pareto Space: Latency vs. Footprint vs. Accuracy", pad=20, fontweight="bold")
    ax.set_xlabel("P50 Latency (ms)", labelpad=10)
    ax.set_ylabel("Disk Size (MB)", labelpad=10)
    ax.set_zlabel("Top-1 Accuracy (%)", labelpad=10)
    ax.view_init(elev=25, azim=45)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_comparison_bars(results: List[Dict[str, any]], output_path: str):
    """
    Generates comparative bar charts displaying relative size reduction, latency speedup, and accuracy.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, dpi=300)

    # Find baseline teacher
    base_teacher = next((r for r in results if "teacher" in r["pipeline"] and "baseline" in r["pipeline"]), results[0])
    base_size = base_teacher["disk_size_mb"]
    base_lat = base_teacher["latency_p50_ms"]

    names = [r["pipeline"].replace("_", "\n") for r in results]
    accuracies = [r["accuracy_top1"] for r in results]
    size_reductions = [base_size / max(r["disk_size_mb"], 1e-4) for r in results]
    speedups = [base_lat / max(r["latency_p50_ms"], 1e-4) for r in results]

    x = np.arange(len(results))

    # Panel 1: Accuracy
    bars1 = ax1.bar(x, accuracies, color="#3b82f6", width=0.6)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, max(accuracies) * 1.15)
    ax1.grid(True, linestyle=":", alpha=0.6)
    for b in bars1:
        yval = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    # Panel 2: Footprint Reduction Factor
    bars2 = ax2.bar(x, size_reductions, color="#10b981", width=0.6)
    ax2.set_ylabel("Size Reduction (x)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    for b in bars2:
        yval = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, yval + 0.1, f"{yval:.1f}x", ha="center", va="bottom", fontsize=8)

    # Panel 3: CPU Latency Speedup Factor
    bars3 = ax3.bar(x, speedups, color="#8b5cf6", width=0.6)
    ax3.set_ylabel("Latency Speedup (x)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.6)
    for b in bars3:
        yval = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2, yval + 0.1, f"{yval:.1f}x", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Compression Suite: Multi-Metric Benchmark Breakdown", y=0.99, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_sparsity_stress_curve(
    grid_results: List[Dict[str, float]],
    optimal_sparsity: float,
    output_path: str,
    max_allowed_drop: float = 2.0,
):
    """
    Plots the Task Accuracy vs. Sparsity stress curve, highlighting the SLA tolerance boundary and the chosen sweet spot.
    """
    if not grid_results:
        return

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sparsities = [r["sparsity_pct"] for r in grid_results]
    accuracies = [r["accuracy"] for r in grid_results]

    baseline_acc = accuracies[0] + grid_results[0]["drop"]
    sla_line = baseline_acc - max_allowed_drop

    ax.plot(sparsities, accuracies, marker="o", markersize=8, color="#2563eb", linewidth=2.5, label="Healed Accuracy (%)")
    ax.axhline(sla_line, color="#dc2626", linestyle="--", linewidth=1.8, label=f"SLA Boundary (-{max_allowed_drop:.1f}%)")

    opt_pct = optimal_sparsity * 100.0
    opt_item = next((r for r in grid_results if abs(r["sparsity_pct"] - opt_pct) < 1e-2), grid_results[0])
    ax.scatter([opt_pct], [opt_item["accuracy"]], color="#16a34a", s=220, edgecolors="#14532d", linewidths=2.0, zorder=5, label=f"Selected Sweet Spot ({opt_pct:.0f}%)")

    for r in grid_results:
        color = "#16a34a" if r["within_sla"] else "#dc2626"
        ax.annotate(
            f"{r['accuracy']:.1f}%\n({r['drop']:+.1f}%)",
            (r["sparsity_pct"], r["accuracy"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=color,
            fontweight="bold"
        )

    ax.set_title("Sparsity Stress Test: Accuracy Degradation vs. Pruning Target", pad=15, fontweight="bold")
    ax.set_xlabel("Target Sparsity (%) — Higher is More Compressed")
    ax.set_ylabel("Healed Top-1 Accuracy (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.replace(".png", ".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    results: List[Dict[str, any]],
    output_dir: str,
    sparsity_grid_results: Optional[List[Dict[str, float]]] = None,
    optimal_sparsity: Optional[float] = None,
):
    """
    Renders all Pareto frontier plots and saves them in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_accuracy_vs_latency_pareto(results, os.path.join(output_dir, "pareto_accuracy_vs_latency.png"))
    plot_accuracy_vs_size_pareto(results, os.path.join(output_dir, "pareto_accuracy_vs_size.png"))
    plot_3d_pareto_frontier(results, os.path.join(output_dir, "pareto_3d_frontier.png"))
    plot_pipeline_comparison_bars(results, os.path.join(output_dir, "pipeline_comparison_bars.png"))
    if sparsity_grid_results and optimal_sparsity is not None:
        plot_sparsity_stress_curve(
            sparsity_grid_results,
            optimal_sparsity,
            os.path.join(output_dir, "sparsity_stress_curve.png"),
        )

