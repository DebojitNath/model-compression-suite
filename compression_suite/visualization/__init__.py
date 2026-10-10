"""
Visualization module init.
"""
from compression_suite.visualization.plots import (
    plot_accuracy_vs_latency_pareto,
    plot_accuracy_vs_size_pareto,
    plot_3d_pareto_frontier,
    plot_pipeline_comparison_bars,
    generate_all_plots,
)

__all__ = [
    "plot_accuracy_vs_latency_pareto",
    "plot_accuracy_vs_size_pareto",
    "plot_3d_pareto_frontier",
    "plot_pipeline_comparison_bars",
    "generate_all_plots",
]
