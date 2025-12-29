"""
Time alignment validation: Compare detected vs labeled event timing.

Principle: Event labels may be unreliable (±20-30 min offset observed).
This module analyzes timing relationships to establish confidence.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Tuple, Optional


def compute_time_offsets(
    detected_times: List[pd.Timestamp],
    labeled_time: pd.Timestamp,
) -> Dict[str, float]:
    """
    Compute time offsets between detected events and labeled event.

    Parameters
    ----------
    detected_times : List[pd.Timestamp]
        Times of detected anomalies (sorted)
    labeled_time : pd.Timestamp
        Ground truth event time from label

    Returns
    -------
    Dict[str, float]
        - 'offsets_minutes': List of offsets in minutes
        - 'closest_offset': Offset to nearest detection (minutes)
        - 'closest_index': Index of nearest detection
    """
    if not detected_times:
        return {
            "offsets_minutes": [],
            "closest_offset": None,
            "closest_index": None,
        }

    # Convert to minutes
    offsets = [
        (t - labeled_time).total_seconds() / 60.0 for t in detected_times
    ]

    # Find closest
    abs_offsets = [abs(o) for o in offsets]
    closest_idx = np.argmin(abs_offsets)
    closest_offset = offsets[closest_idx]

    return {
        "offsets_minutes": offsets,
        "closest_offset": closest_offset,
        "closest_index": closest_idx,
        "abs_offsets_minutes": abs_offsets,
        "all_detections": detected_times,
    }


def analyze_detection_timing(
    energy_detections: Dict[str, any],
    subspace_detections: Dict[str, any],
    spatial_detections: Dict[str, any],
    labeled_time: pd.Timestamp,
) -> Dict[str, any]:
    """
    Analyze timing of detections from all three metrics.

    Parameters
    ----------
    energy_detections : Dict
        From residual_energy.detect_excitation_anomalies()
        Expected keys: 'event_times' or similar
    subspace_detections : Dict
        From subspace_change.detect_subspace_anomalies()
    spatial_detections : Dict
        From spatial_coherence.extract_spatially_consistent_events()
    labeled_time : pd.Timestamp
        Ground truth event time

    Returns
    -------
    Dict
        Timing analysis for all metrics
    """
    results = {}

    # Energy metric
    if isinstance(energy_detections, dict) and "event_intervals" in energy_detections:
        event_times = [
            iv["start_time"] for iv in energy_detections["event_intervals"]
        ]
        results["energy"] = compute_time_offsets(event_times, labeled_time)
    else:
        results["energy"] = {"offsets_minutes": [], "closest_offset": None}

    # Subspace metric
    if isinstance(subspace_detections, dict) and "event_intervals" in subspace_detections:
        event_times = [
            iv["start_time"] for iv in subspace_detections["event_intervals"]
        ]
        results["subspace"] = compute_time_offsets(event_times, labeled_time)
    else:
        results["subspace"] = {"offsets_minutes": [], "closest_offset": None}

    # Spatial consensus
    if isinstance(spatial_detections, dict) and "events" in spatial_detections:
        event_times = [ev["start_time"] for ev in spatial_detections["events"]]
        results["spatial"] = compute_time_offsets(event_times, labeled_time)
    else:
        results["spatial"] = {"offsets_minutes": [], "closest_offset": None}

    return results


def generate_timing_statistics(
    timing_analysis: Dict[str, any],
) -> pd.DataFrame:
    """
    Generate summary statistics for detection timing.

    Parameters
    ----------
    timing_analysis : Dict
        From analyze_detection_timing()

    Returns
    -------
    pd.DataFrame
        Summary table with columns:
        - Metric: metric name
        - N_Detections: number of detections
        - Closest_Offset_Minutes: timing offset to closest detection
        - Mean_Offset_Minutes: mean offset of all detections
        - Std_Offset_Minutes: standard deviation
        - Min_Offset_Minutes: minimum absolute offset
    """
    stats = []

    for metric_name, analysis in timing_analysis.items():
        offsets = analysis.get("offsets_minutes", [])
        
        if offsets:
            abs_offsets = [abs(o) for o in offsets]
            stats.append({
                "Metric": metric_name,
                "N_Detections": len(offsets),
                "Closest_Offset_Minutes": analysis.get("closest_offset", np.nan),
                "Mean_Offset_Minutes": np.mean(offsets),
                "Std_Offset_Minutes": np.std(offsets),
                "Min_Abs_Offset_Minutes": np.min(abs_offsets),
                "Max_Abs_Offset_Minutes": np.max(abs_offsets),
            })
        else:
            stats.append({
                "Metric": metric_name,
                "N_Detections": 0,
                "Closest_Offset_Minutes": np.nan,
                "Mean_Offset_Minutes": np.nan,
                "Std_Offset_Minutes": np.nan,
                "Min_Abs_Offset_Minutes": np.nan,
                "Max_Abs_Offset_Minutes": np.nan,
            })

    return pd.DataFrame(stats)
