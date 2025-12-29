#!/usr/bin/env python
"""Generate publication-quality figures for multi-event analysis results.

Creates figures for IEEE TSG submission:
- Figure 1: Latency distribution histogram
- Figure 2: Detection performance by voltage level
- Figure 3: Event detection timeline
- Figure 4: Summary performance table
- Figure 5: Latency vs voltage scatter plot
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Publication-quality settings
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})


def load_results(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load multi-event analysis results."""
    results_df = pd.read_csv(results_dir / "all_events_results.csv")
    with open(results_dir / "analysis_summary.json") as f:
        summary = json.load(f)
    return results_df, summary


def plot_latency_distribution(
    results_df: pd.DataFrame,
    save_path: Path,
) -> plt.Figure:
    """
    Figure 1: Latency distribution histogram with statistics.
    
    Shows distribution of detection latencies across all events.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter detected events
    detected = results_df[results_df["latency_minutes"].notna()]
    latencies = detected["latency_minutes"].values
    
    if len(latencies) == 0:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center", fontsize=14)
        return fig
    
    # Histogram
    bins = np.linspace(-40, 40, 17)
    n, bins_out, patches = ax.hist(
        latencies, bins=bins, color="steelblue", edgecolor="white",
        alpha=0.7, label=f"Detected Events (n={len(latencies)})"
    )
    
    # Color bars by sign
    for i, patch in enumerate(patches):
        if bins_out[i] < 0:
            patch.set_facecolor("forestgreen")
        else:
            patch.set_facecolor("coral")
    
    # Statistics
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    std_lat = np.std(latencies)
    
    ax.axvline(mean_lat, color="red", linestyle="--", linewidth=2, 
               label=f"Mean: {mean_lat:+.1f} min")
    ax.axvline(median_lat, color="blue", linestyle=":", linewidth=2,
               label=f"Median: {median_lat:+.1f} min")
    ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    
    # Annotations
    ax.set_xlabel("Detection Latency (minutes from labeled event)", fontsize=11)
    ax.set_ylabel("Number of Events", fontsize=11)
    ax.set_title("Detection Latency Distribution Across All Events", 
                 fontsize=12, fontweight="bold")
    
    # Add text box with statistics
    stats_text = (
        f"Statistics (n={len(latencies)}):\n"
        f"Mean: {mean_lat:+.1f} min\n"
        f"Median: {median_lat:+.1f} min\n"
        f"Std Dev: {std_lat:.1f} min\n"
        f"Range: [{min(latencies):+.1f}, {max(latencies):+.1f}]"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    # Legend for colors
    ax.text(0.02, 0.95, "■ Early detection (negative)", transform=ax.transAxes,
            fontsize=9, color="forestgreen", verticalalignment="top")
    ax.text(0.02, 0.90, "■ Late detection (positive)", transform=ax.transAxes,
            fontsize=9, color="coral", verticalalignment="top")
    
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_voltage_performance(
    results_df: pd.DataFrame,
    save_path: Path,
) -> plt.Figure:
    """
    Figure 2: Detection performance grouped by voltage level.
    
    Bar chart showing detection rate and mean latency per voltage class.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Group by voltage
    results_df["voltage_class"] = (results_df["voltage_kv"] / 1000).astype(int).astype(str) + "kV"
    voltage_groups = results_df.groupby("voltage_class")
    
    # Detection rate by voltage
    detection_stats = voltage_groups.apply(
        lambda g: pd.Series({
            "total": len(g),
            "detected": g["latency_minutes"].notna().sum(),
            "detection_rate": g["latency_minutes"].notna().mean() * 100,
            "mean_latency": g["latency_minutes"].mean(),
            "std_latency": g["latency_minutes"].std(),
            "mean_agreement": g["spatial_agreement"].mean() * 100,
        })
    ).reset_index()
    
    # Sort by voltage
    voltage_order = ["69kV", "138kV", "161kV"]
    detection_stats["sort_key"] = detection_stats["voltage_class"].apply(
        lambda x: voltage_order.index(x) if x in voltage_order else 99
    )
    detection_stats = detection_stats.sort_values("sort_key")
    
    # Plot 1: Detection rate
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars1 = ax1.bar(
        detection_stats["voltage_class"],
        detection_stats["detection_rate"],
        color=colors[:len(detection_stats)],
        edgecolor="black",
        alpha=0.8,
    )
    
    # Add count labels on bars
    for bar, (_, row) in zip(bars1, detection_stats.iterrows()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f"{int(row['detected'])}/{int(row['total'])}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax1.set_ylabel("Detection Rate (%)", fontsize=11)
    ax1.set_xlabel("Voltage Level", fontsize=11)
    ax1.set_title("Detection Rate by Voltage Level", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 120)
    ax1.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Plot 2: Mean latency with error bars
    detected_stats = detection_stats[detection_stats["detected"] > 0]
    
    bars2 = ax2.bar(
        detected_stats["voltage_class"],
        detected_stats["mean_latency"],
        yerr=detected_stats["std_latency"].fillna(0),
        color=colors[:len(detected_stats)],
        edgecolor="black",
        alpha=0.8,
        capsize=5,
    )
    
    ax2.axhline(0, color="black", linestyle="-", linewidth=1)
    ax2.set_ylabel("Mean Latency (minutes)", fontsize=11)
    ax2.set_xlabel("Voltage Level", fontsize=11)
    ax2.set_title("Mean Detection Latency by Voltage Level", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Add annotations
    for bar, (_, row) in zip(bars2, detected_stats.iterrows()):
        height = bar.get_height()
        y_pos = height + 2 if height >= 0 else height - 4
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f"{height:+.1f}",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_event_timeline(
    results_df: pd.DataFrame,
    save_path: Path,
) -> plt.Figure:
    """
    Figure 3: Event detection timeline.
    
    Shows labeled vs detected times for all events on a timeline.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Parse timestamps
    results_df = results_df.copy()
    results_df["labeled_time"] = pd.to_datetime(results_df["labeled_time"])
    results_df["first_detection_time"] = pd.to_datetime(results_df["first_detection_time"])
    
    # Sort by labeled time
    results_df = results_df.sort_values("labeled_time")
    
    y_positions = range(len(results_df))
    colors = {"69kV": "#2ecc71", "138kV": "#e74c3c", "161kV": "#3498db"}
    
    for y, (_, row) in zip(y_positions, results_df.iterrows()):
        voltage_class = f"{int(row['voltage_kv']/1000)}kV"
        color = colors.get(voltage_class, "gray")
        
        # Plot labeled time
        ax.scatter(row["labeled_time"], y, marker="o", s=150, color=color,
                  edgecolor="black", linewidth=1.5, zorder=3)
        
        # Plot detection time if available
        if pd.notna(row["first_detection_time"]):
            ax.scatter(row["first_detection_time"], y, marker="D", s=100, 
                      color=color, edgecolor="black", linewidth=1.5, 
                      alpha=0.7, zorder=3)
            
            # Draw connecting line
            ax.plot([row["labeled_time"], row["first_detection_time"]], [y, y],
                   color=color, linewidth=2, alpha=0.5, zorder=2)
        
        # Label
        label = f"Section {int(row['section_id'])} ({voltage_class})"
        ax.text(row["labeled_time"], y + 0.3, label, fontsize=8, 
                ha="center", va="bottom")
    
    # Legend
    legend_elements = [
        plt.scatter([], [], marker="o", s=100, color="gray", edgecolor="black",
                   label="Labeled Event Time"),
        plt.scatter([], [], marker="D", s=80, color="gray", edgecolor="black",
                   alpha=0.7, label="First Detection Time"),
    ]
    for voltage, color in colors.items():
        legend_elements.append(
            plt.scatter([], [], marker="s", s=80, color=color, label=voltage)
        )
    
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    
    ax.set_yticks([])
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_title("Event Detection Timeline", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Format x-axis
    fig.autofmt_xdate(rotation=45)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_summary_table(
    results_df: pd.DataFrame,
    summary: dict,
    save_path: Path,
) -> plt.Figure:
    """
    Figure 4: Summary performance table as figure.
    
    Publication-ready table with key metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    
    # Prepare table data
    results_df = results_df.copy()
    results_df["voltage_class"] = (results_df["voltage_kv"] / 1000).astype(int).astype(str) + "kV"
    
    table_data = []
    for _, row in results_df.iterrows():
        status = "Yes" if pd.notna(row["latency_minutes"]) else "No"
        latency_str = f"{row['latency_minutes']:+.1f}" if pd.notna(row["latency_minutes"]) else "N/A"
        agreement_str = f"{row['spatial_agreement']*100:.1f}%" if row["spatial_agreement"] > 0 else "N/A"
        
        table_data.append([
            int(row["section_id"]),
            row["voltage_class"],
            status,
            int(row["n_energy_events"]),
            int(row["n_subspace_events"]),
            latency_str,
            agreement_str,
            row["processing_status"],
        ])
    
    columns = [
        "Section ID", "Voltage", "Detected", "Energy\nEvents",
        "Subspace\nEvents", "Latency\n(min)", "Agreement", "Status"
    ]
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor("#4a90d9")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    
    # Row styling
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if table_data[i-1][2] == "Yes":
                table[(i, j)].set_facecolor("#d4edda")
            elif table_data[i-1][7] == "insufficient_data":
                table[(i, j)].set_facecolor("#fff3cd")
            else:
                table[(i, j)].set_facecolor("#f8d7da")
    
    # Title
    ax.set_title("Multi-Event Detection Results Summary", 
                 fontsize=14, fontweight="bold", pad=20)
    
    # Summary statistics below table
    latency_stats = summary["latency_stats"]
    summary_text = (
        f"Overall: {latency_stats['n_detected']}/{latency_stats['n_total']} detected "
        f"({latency_stats['detection_rate']*100:.1f}%) | "
        f"Mean Latency: {latency_stats['latency_mean_min']:+.1f} min | "
        f"Median: {latency_stats['latency_median_min']:+.1f} min"
    )
    fig.text(0.5, 0.08, summary_text, ha="center", fontsize=11, 
             style="italic", color="gray")
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    
    return fig


def plot_latency_scatter(
    results_df: pd.DataFrame,
    save_path: Path,
) -> plt.Figure:
    """
    Figure 5: Latency vs spatial agreement scatter plot.
    
    Shows relationship between detection quality metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter detected events
    detected = results_df[results_df["latency_minutes"].notna()].copy()
    
    if len(detected) == 0:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center", fontsize=14)
        return fig
    
    detected["voltage_class"] = (detected["voltage_kv"] / 1000).astype(int).astype(str) + "kV"
    
    colors = {"69kV": "#2ecc71", "138kV": "#e74c3c", "161kV": "#3498db"}
    
    for voltage_class in detected["voltage_class"].unique():
        subset = detected[detected["voltage_class"] == voltage_class]
        ax.scatter(
            subset["latency_minutes"],
            subset["spatial_agreement"] * 100,
            s=200,
            c=colors.get(voltage_class, "gray"),
            label=voltage_class,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )
        
        # Add section labels
        for _, row in subset.iterrows():
            ax.annotate(
                f"S{int(row['section_id'])}",
                (row["latency_minutes"], row["spatial_agreement"] * 100),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
    
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(95, color="green", linestyle=":", alpha=0.5, label="95% Agreement")
    
    ax.set_xlabel("Detection Latency (minutes from labeled event)", fontsize=11)
    ax.set_ylabel("Spatial Agreement (%)", fontsize=11)
    ax.set_title("Detection Latency vs Spatial Agreement", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(90, 101)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {save_path}")
    
    return fig


def main():
    """Generate all multi-event publication figures."""
    print("=" * 70)
    print("MULTI-EVENT PUBLICATION FIGURES")
    print("=" * 70)
    
    root = Path(__file__).parent
    results_dir = root / "results" / "multi_event"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_df, summary = load_results(results_dir)
    print(f"\nLoaded {len(results_df)} event results")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Figure 1: Latency distribution
    plot_latency_distribution(
        results_df,
        figures_dir / "fig1_latency_distribution.png"
    )
    
    # Figure 2: Voltage performance
    plot_voltage_performance(
        results_df,
        figures_dir / "fig2_voltage_performance.png"
    )
    
    # Figure 3: Event timeline
    plot_event_timeline(
        results_df,
        figures_dir / "fig3_event_timeline.png"
    )
    
    # Figure 4: Summary table
    plot_summary_table(
        results_df,
        summary,
        figures_dir / "fig4_summary_table.png"
    )
    
    # Figure 5: Latency scatter
    plot_latency_scatter(
        results_df,
        figures_dir / "fig5_latency_scatter.png"
    )
    
    print(f"\n✓ All figures saved to {figures_dir}")
    print("\nFigures generated:")
    print("  1. fig1_latency_distribution.png - Detection latency histogram")
    print("  2. fig2_voltage_performance.png - Performance by voltage level")
    print("  3. fig3_event_timeline.png - Event detection timeline")
    print("  4. fig4_summary_table.png - Results summary table")
    print("  5. fig5_latency_scatter.png - Latency vs agreement scatter")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
