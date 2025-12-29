"""Visualization module for publication-quality figures."""

from .plots import (
    plot_signal_timeseries,
    plot_energy_metric,
    plot_subspace_metric,
    plot_spatial_voting_heatmap,
    plot_anomaly_detection_comparison,
    plot_sensitivity_curves,
)

__all__ = [
    "plot_signal_timeseries",
    "plot_energy_metric",
    "plot_subspace_metric",
    "plot_spatial_voting_heatmap",
    "plot_anomaly_detection_comparison",
    "plot_sensitivity_curves",
]
