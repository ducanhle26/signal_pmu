"""
Publication-quality visualization functions for PMU anomaly detection.

Generates figures for IEEE TSG submission:
- Figure 1: Raw PMU signal timeseries with event windows
- Figure 2: Residual energy metric and detections
- Figure 3: Subspace distance metric
- Figure 4: Spatial voting heatmap (cross-terminal consensus)
- Figure 5: Anomaly detection comparison (energy vs subspace vs spatial)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def plot_signal_timeseries(
    terminal_data: Dict[str, pd.DataFrame],
    labeled_time: pd.Timestamp,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot raw PMU voltage signals for multiple terminals.

    Parameters
    ----------
    terminal_data : Dict[str, pd.DataFrame]
        {term_id: dataframe with VP_M, VA_M, VB_M, VC_M columns}
    labeled_time : pd.Timestamp
        Ground truth event time (vertical line)
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(
        len(terminal_data), 1, figsize=figsize, sharex=True
    )
    if len(terminal_data) == 1:
        axes = [axes]

    voltage_cols = ["VP_M", "VA_M", "VB_M", "VC_M"]

    for ax, (term_id, data) in enumerate(sorted(terminal_data.items())):
        axis = axes[ax]
        
        # Plot available voltage channels
        for col in voltage_cols:
            if col in data.columns:
                axis.plot(
                    data.index, data[col], label=col, linewidth=0.8, alpha=0.8
                )

        # Mark labeled event time
        axis.axvline(labeled_time, color="red", linestyle="--", 
                     linewidth=2, label="Labeled Event", alpha=0.7)

        axis.set_ylabel(f"Term {term_id}\nVoltage (kV)", fontsize=10)
        axis.grid(True, alpha=0.3)
        if ax == 0:
            axis.legend(loc="upper right", ncol=5, fontsize=8)

    axes[-1].set_xlabel("UTC Time", fontsize=11)
    fig.suptitle("PMU Voltage Signals - Multi-Terminal Analysis", 
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def plot_energy_metric(
    energy_metric: np.ndarray,
    threshold: float,
    detections: np.ndarray,
    timestamps: pd.DatetimeIndex,
    labeled_time: pd.Timestamp,
    term_id: str = "249",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot residual energy metric with threshold and detections.

    Parameters
    ----------
    energy_metric : np.ndarray
        Energy scores over time
    threshold : float
        Detection threshold (99th percentile)
    detections : np.ndarray
        Binary anomaly flags (0/1)
    timestamps : pd.DatetimeIndex
        Time index
    labeled_time : pd.Timestamp
        Ground truth event time
    term_id : str
        Terminal ID for label
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot energy metric
    ax.plot(timestamps, energy_metric, linewidth=1, label="Residual Energy",
            color="steelblue", alpha=0.8)

    # Threshold
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=2,
               label=f"Threshold (99th %ile)", alpha=0.7)

    # Shade anomalous regions
    anomaly_indices = np.where(detections > 0)[0]
    if len(anomaly_indices) > 0:
        # Find contiguous regions
        diffs = np.diff(anomaly_indices)
        splits = np.where(diffs > 1)[0] + 1
        regions = np.split(anomaly_indices, splits)
        
        for region in regions:
            if len(region) > 0:
                ax.axvspan(timestamps[region[0]], timestamps[region[-1]],
                          alpha=0.2, color="red", label="Detection" if region is regions[0] else "")

    # Mark labeled event
    ax.axvline(labeled_time, color="green", linestyle="--", linewidth=2,
               label="Labeled Event", alpha=0.7)

    ax.set_ylabel("Energy (normalized)", fontsize=11)
    ax.set_xlabel("UTC Time", fontsize=11)
    ax.set_title(f"Residual Excitation Energy - Terminal {term_id}", 
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def plot_subspace_metric(
    subspace_metric: np.ndarray,
    threshold: float,
    detections: np.ndarray,
    timestamps: pd.DatetimeIndex,
    labeled_time: pd.Timestamp,
    term_id: str = "249",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot subspace distance metric with threshold and detections.

    Parameters
    ----------
    subspace_metric : np.ndarray
        Subspace distance scores
    threshold : float
        Detection threshold (95th percentile)
    detections : np.ndarray
        Binary anomaly flags
    timestamps : pd.DatetimeIndex
        Time index
    labeled_time : pd.Timestamp
        Ground truth event time
    term_id : str
        Terminal ID
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot metric
    ax.plot(timestamps, subspace_metric, linewidth=1, label="Subspace Distance",
            color="darkviolet", alpha=0.8)

    # Threshold
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=2,
               label=f"Threshold (95th %ile)", alpha=0.7)

    # Shade anomalous regions
    anomaly_indices = np.where(detections > 0)[0]
    if len(anomaly_indices) > 0:
        diffs = np.diff(anomaly_indices)
        splits = np.where(diffs > 1)[0] + 1
        regions = np.split(anomaly_indices, splits)
        
        for i, region in enumerate(regions):
            if len(region) > 0:
                ax.axvspan(timestamps[region[0]], timestamps[region[-1]],
                          alpha=0.2, color="red", 
                          label="Detection" if i == 0 else "")

    # Mark labeled event
    ax.axvline(labeled_time, color="green", linestyle="--", linewidth=2,
               label="Labeled Event", alpha=0.7)

    ax.set_ylabel("Distance (principal angles)", fontsize=11)
    ax.set_xlabel("UTC Time", fontsize=11)
    ax.set_title(f"Subspace Distance - Terminal {term_id}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def plot_spatial_voting_heatmap(
    terminal_anomalies: Dict[str, np.ndarray],
    timestamps: pd.DatetimeIndex,
    labeled_time: pd.Timestamp,
    metric_name: str = "Energy",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cross-terminal anomaly voting heatmap.

    Parameters
    ----------
    terminal_anomalies : Dict[str, np.ndarray]
        {term_id: binary anomaly flags}
    timestamps : pd.DatetimeIndex
        Time index
    labeled_time : pd.Timestamp
        Ground truth event time
    metric_name : str
        Metric name for title
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    plt.Figure
    """
    # Build matrix: terminals x time
    term_ids = sorted(terminal_anomalies.keys())
    anomaly_matrix = np.vstack([
        terminal_anomalies[tid] for tid in term_ids
    ])

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(anomaly_matrix, aspect="auto", cmap="RdYlGn_r",
                   interpolation="nearest", vmin=0, vmax=1)

    ax.set_yticks(range(len(term_ids)))
    ax.set_yticklabels([f"Term {tid}" for tid in term_ids])
    ax.set_xlabel("Time Index", fontsize=11)
    ax.set_title(f"Cross-Terminal Anomaly Voting - {metric_name} Metric",
                 fontsize=12, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Anomaly Flag")

    # Mark labeled event
    if labeled_time in timestamps:
        labeled_idx = timestamps.get_loc(labeled_time)
        ax.axvline(labeled_idx, color="cyan", linestyle="--", linewidth=2,
                  label="Labeled Event")
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def plot_anomaly_detection_comparison(
    energy_detections: np.ndarray,
    subspace_detections: np.ndarray,
    spatial_detections: np.ndarray,
    timestamps: pd.DatetimeIndex,
    labeled_time: pd.Timestamp,
    term_id: str = "249",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Comparison of anomaly detections from three metrics.

    Parameters
    ----------
    energy_detections : np.ndarray
        Binary flags from energy metric
    subspace_detections : np.ndarray
        Binary flags from subspace metric
    spatial_detections : np.ndarray
        Binary flags from spatial voting
    timestamps : pd.DatetimeIndex
        Time index
    labeled_time : pd.Timestamp
        Ground truth event time
    term_id : str
        Terminal ID
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    metrics = [
        (energy_detections, "Energy", axes[0]),
        (subspace_detections, "Subspace", axes[1]),
        (spatial_detections, "Spatial Consensus", axes[2]),
    ]

    for detections, label, ax in metrics:
        # Plot as filled area
        ax.fill_between(timestamps, 0, detections, alpha=0.6, step="mid",
                       label=f"{label} Detection", color="steelblue")

        # Mark labeled event
        ax.axvline(labeled_time, color="red", linestyle="--", linewidth=2,
                  label="Labeled Event", alpha=0.7)

        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Normal", "Anomaly"])
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("UTC Time", fontsize=11)
    fig.suptitle(f"Anomaly Detection Comparison - Terminal {term_id}",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig


def plot_sensitivity_curves(
    sensitivity_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot detection count vs threshold (sensitivity analysis).

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        From sensitivity_analysis() with columns:
        - Energy_Threshold
        - N_Energy_Detections
        - Subspace_Threshold
        - N_Subspace_Detections
    figsize : Tuple[int, int]
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Energy sensitivity
    energy_df = sensitivity_df[["Energy_Threshold", "N_Energy_Detections"]].drop_duplicates()
    energy_df = energy_df.sort_values("Energy_Threshold")
    ax1.plot(energy_df["Energy_Threshold"], energy_df["N_Energy_Detections"],
            marker="o", linewidth=2, markersize=8, color="steelblue")
    ax1.set_xlabel("Energy Threshold", fontsize=11)
    ax1.set_ylabel("Number of Detections", fontsize=11)
    ax1.set_title("Energy Metric Sensitivity", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Subspace sensitivity
    subspace_df = sensitivity_df[["Subspace_Threshold", "N_Subspace_Detections"]].drop_duplicates()
    subspace_df = subspace_df.sort_values("Subspace_Threshold")
    ax2.plot(subspace_df["Subspace_Threshold"], subspace_df["N_Subspace_Detections"],
            marker="s", linewidth=2, markersize=8, color="darkviolet")
    ax2.set_xlabel("Subspace Threshold", fontsize=11)
    ax2.set_ylabel("Number of Detections", fontsize=11)
    ax2.set_title("Subspace Metric Sensitivity", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Sensitivity Analysis: Detection Count vs Threshold",
                fontsize=12, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {save_path}")

    return fig
