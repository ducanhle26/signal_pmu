"""Phase 6: Cross-event aggregation and latency analysis."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EventResult:
    """Detection results for a single event."""

    section_id: int
    term_ids: list[int]
    voltage_kv: int
    cause: str
    labeled_time: pd.Timestamp
    n_energy_events: int
    n_subspace_events: int
    first_detection_time: Optional[pd.Timestamp]
    latency_minutes: Optional[float]
    spatial_agreement: float
    false_alarm_rate: float
    processing_status: str
    error_message: Optional[str] = None


def compute_latency_stats(results: list[EventResult]) -> dict:
    """
    Compute latency distribution statistics across events.

    Returns:
        Dict with mean, median, std, min, max, and per-event latencies
    """
    latencies = [r.latency_minutes for r in results if r.latency_minutes is not None]

    if not latencies:
        return {
            "n_detected": 0,
            "n_total": len(results),
            "detection_rate": 0.0,
            "latency_mean_min": None,
            "latency_median_min": None,
            "latency_std_min": None,
            "latency_min_min": None,
            "latency_max_min": None,
            "latencies": [],
        }

    return {
        "n_detected": len(latencies),
        "n_total": len(results),
        "detection_rate": len(latencies) / len(results),
        "latency_mean_min": float(np.mean(latencies)),
        "latency_median_min": float(np.median(latencies)),
        "latency_std_min": float(np.std(latencies)),
        "latency_min_min": float(np.min(latencies)),
        "latency_max_min": float(np.max(latencies)),
        "latencies": latencies,
    }


def compute_voltage_breakdown(results: list[EventResult]) -> pd.DataFrame:
    """
    Breakdown detection performance by voltage level.

    Returns:
        DataFrame with per-voltage statistics
    """
    rows = []
    for result in results:
        rows.append(
            {
                "section_id": result.section_id,
                "voltage_kv": result.voltage_kv,
                "detected": result.first_detection_time is not None,
                "latency_min": result.latency_minutes,
                "spatial_agreement": result.spatial_agreement,
                "far": result.false_alarm_rate,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    breakdown = (
        df.groupby("voltage_kv")
        .agg(
            {
                "section_id": "count",
                "detected": "sum",
                "latency_min": ["mean", "std"],
                "spatial_agreement": "mean",
                "far": "mean",
            }
        )
        .round(3)
    )

    breakdown.columns = [
        "n_events",
        "n_detected",
        "latency_mean",
        "latency_std",
        "agreement_mean",
        "far_mean",
    ]
    breakdown["detection_rate"] = breakdown["n_detected"] / breakdown["n_events"]

    return breakdown.reset_index()


def generate_summary_report(
    results: list[EventResult], output_path: Optional[Path] = None
) -> str:
    """
    Generate comprehensive multi-event summary report.

    Returns:
        Report text
    """
    latency_stats = compute_latency_stats(results)
    voltage_breakdown = compute_voltage_breakdown(results)

    lines = [
        "=" * 70,
        "PHASE 6: MULTI-EVENT ANALYSIS SUMMARY",
        "=" * 70,
        "",
        "OVERALL STATISTICS",
        "-" * 50,
        f"Total events analyzed: {len(results)}",
        f"Events with detections: {latency_stats['n_detected']}",
        f"Detection rate: {latency_stats['detection_rate']:.1%}",
        "",
    ]

    if latency_stats["latency_mean_min"] is not None:
        lines.extend(
            [
                "LATENCY DISTRIBUTION (minutes from label)",
                "-" * 50,
                f"Mean: {latency_stats['latency_mean_min']:+.1f}",
                f"Median: {latency_stats['latency_median_min']:+.1f}",
                f"Std Dev: {latency_stats['latency_std_min']:.1f}",
                f"Range: [{latency_stats['latency_min_min']:+.1f}, {latency_stats['latency_max_min']:+.1f}]",
                "",
            ]
        )

    if not voltage_breakdown.empty:
        lines.extend(
            [
                "BREAKDOWN BY VOLTAGE LEVEL",
                "-" * 50,
            ]
        )
        for _, row in voltage_breakdown.iterrows():
            lines.append(
                f"  {int(row['voltage_kv']/1000)}kV: "
                f"{int(row['n_detected'])}/{int(row['n_events'])} detected "
                f"({row['detection_rate']:.0%}), "
                f"latency={row['latency_mean']:.1f}±{row['latency_std']:.1f} min"
                if pd.notna(row["latency_mean"])
                else f"  {int(row['voltage_kv']/1000)}kV: "
                f"{int(row['n_detected'])}/{int(row['n_events'])} detected"
            )
        lines.append("")

    lines.extend(
        [
            "PER-EVENT RESULTS",
            "-" * 50,
        ]
    )

    for r in sorted(results, key=lambda x: x.section_id):
        status = "✓" if r.first_detection_time else "✗"
        latency_str = f"{r.latency_minutes:+.1f} min" if r.latency_minutes else "N/A"
        lines.append(
            f"  [{status}] Section {r.section_id} ({int(r.voltage_kv/1000)}kV): "
            f"latency={latency_str}, agreement={r.spatial_agreement:.1%}"
        )

    lines.extend(
        [
            "",
            "FAILED EVENTS",
            "-" * 50,
        ]
    )

    failed = [r for r in results if r.processing_status != "success"]
    if failed:
        for r in failed:
            lines.append(f"  Section {r.section_id}: {r.error_message}")
    else:
        lines.append("  None")

    lines.extend(
        [
            "",
            "=" * 70,
        ]
    )

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report)
        logger.info(f"Summary report saved to {output_path}")

    return report


def results_to_dataframe(results: list[EventResult]) -> pd.DataFrame:
    """Convert results list to DataFrame for export."""
    rows = []
    for r in results:
        rows.append(
            {
                "section_id": r.section_id,
                "term_ids": str(r.term_ids),
                "voltage_kv": r.voltage_kv,
                "cause": r.cause,
                "labeled_time": r.labeled_time,
                "n_energy_events": r.n_energy_events,
                "n_subspace_events": r.n_subspace_events,
                "first_detection_time": r.first_detection_time,
                "latency_minutes": r.latency_minutes,
                "spatial_agreement": r.spatial_agreement,
                "false_alarm_rate": r.false_alarm_rate,
                "processing_status": r.processing_status,
                "error_message": r.error_message,
            }
        )
    return pd.DataFrame(rows)
