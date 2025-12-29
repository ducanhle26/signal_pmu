"""
Internal consistency validation: Cross-metric agreement and sensitivity analysis.

Principle: Multiple independent metrics should detect same events.
This module quantifies agreement and tests robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import timedelta


def compute_metric_agreement(
    energy_anomalies: np.ndarray,
    subspace_anomalies: np.ndarray,
    spatial_anomalies: np.ndarray,
    tolerance_samples: int = 30,
) -> Dict[str, float]:
    """
    Compute pairwise agreement between anomaly detection metrics.

    Parameters
    ----------
    energy_anomalies : np.ndarray
        Boolean or binary anomaly flags (shape: n_samples)
    subspace_anomalies : np.ndarray
        Boolean or binary anomaly flags
    spatial_anomalies : np.ndarray
        Boolean or binary anomaly flags (final consensus)
    tolerance_samples : int
        Time window (samples) for considering detections aligned

    Returns
    -------
    Dict[str, float]
        - 'energy_vs_subspace': Agreement ratio
        - 'energy_vs_spatial': Agreement ratio
        - 'subspace_vs_spatial': Agreement ratio
        - 'all_three': Ratio where all agree
        - 'at_least_two': Ratio where at least 2 agree
    """
    assert len(energy_anomalies) == len(subspace_anomalies) == len(spatial_anomalies)

    # Find anomaly windows (consecutive anomalies)
    def extract_windows(anomalies: np.ndarray, tol: int) -> List[Tuple[int, int]]:
        """Extract start/end indices of anomalous periods."""
        if not np.any(anomalies):
            return []
        
        # Dilate by tolerance
        dilated = np.zeros_like(anomalies)
        anomaly_indices = np.where(anomalies)[0]
        
        for idx in anomaly_indices:
            start = max(0, idx - tol)
            end = min(len(anomalies), idx + tol + 1)
            dilated[start:end] = 1
        
        # Extract contiguous windows
        windows = []
        in_window = False
        win_start = 0
        
        for i, val in enumerate(dilated):
            if val and not in_window:
                in_window = True
                win_start = i
            elif not val and in_window:
                in_window = False
                windows.append((win_start, i))
        
        if in_window:
            windows.append((win_start, len(dilated)))
        
        return windows

    windows_energy = extract_windows(energy_anomalies, tolerance_samples)
    windows_subspace = extract_windows(subspace_anomalies, tolerance_samples)
    windows_spatial = extract_windows(spatial_anomalies, tolerance_samples)

    # Compute overlaps
    def window_overlap(windows1: List, windows2: List) -> float:
        """Fraction of samples in windows1 that overlap with windows2."""
        if not windows1:
            return 0.0 if windows2 else 1.0
        
        union_len = 0
        intersect_len = 0
        
        for w1_start, w1_end in windows1:
            w1_len = w1_end - w1_start
            union_len += w1_len
            
            # Find overlaps
            for w2_start, w2_end in windows2:
                overlap_start = max(w1_start, w2_start)
                overlap_end = min(w1_end, w2_end)
                if overlap_end > overlap_start:
                    intersect_len += overlap_end - overlap_start
        
        if union_len == 0:
            return 1.0 if not windows2 else 0.0
        
        return intersect_len / union_len

    energy_vs_subspace = window_overlap(windows_energy, windows_subspace)
    energy_vs_spatial = window_overlap(windows_energy, windows_spatial)
    subspace_vs_spatial = window_overlap(windows_subspace, windows_spatial)

    # All three agree
    all_three_agreement = np.mean(
        (energy_anomalies == 1) & 
        (subspace_anomalies == 1) & 
        (spatial_anomalies == 1)
    )

    # At least two agree
    at_least_two = np.mean(
        (energy_anomalies + subspace_anomalies + spatial_anomalies) >= 2
    )

    return {
        "energy_vs_subspace": energy_vs_subspace,
        "energy_vs_spatial": energy_vs_spatial,
        "subspace_vs_spatial": subspace_vs_spatial,
        "all_three": all_three_agreement,
        "at_least_two": at_least_two,
    }


def generate_confusion_matrix(
    metric1_anomalies: np.ndarray,
    metric2_anomalies: np.ndarray,
    metric_name1: str = "Metric1",
    metric_name2: str = "Metric2",
) -> pd.DataFrame:
    """
    Generate 2x2 confusion matrix for two binary anomaly signals.

    Parameters
    ----------
    metric1_anomalies : np.ndarray
        Binary anomaly flags from first metric
    metric2_anomalies : np.ndarray
        Binary anomaly flags from second metric
    metric_name1, metric_name2 : str
        Metric names for labels

    Returns
    -------
    pd.DataFrame
        2x2 confusion matrix
    """
    assert len(metric1_anomalies) == len(metric2_anomalies)

    # Convert to binary
    m1 = (metric1_anomalies > 0).astype(int)
    m2 = (metric2_anomalies > 0).astype(int)

    # Counts
    both_anomaly = np.sum((m1 == 1) & (m2 == 1))
    m1_only = np.sum((m1 == 1) & (m2 == 0))
    m2_only = np.sum((m1 == 0) & (m2 == 1))
    neither = np.sum((m1 == 0) & (m2 == 0))

    matrix = pd.DataFrame(
        {
            f"{metric_name2} Anomaly": [both_anomaly, m2_only],
            f"{metric_name2} Normal": [m1_only, neither],
        },
        index=pd.Index([f"{metric_name1} Anomaly", f"{metric_name1} Normal"],
                       name=metric_name1),
    )

    return matrix


def compute_cross_terminal_agreement(
    terminal_anomalies: Dict[str, np.ndarray],
    n_terminals_required: int = 2,
) -> Dict[str, any]:
    """
    Compute cross-terminal agreement for anomaly detection.

    Parameters
    ----------
    terminal_anomalies : Dict[str, np.ndarray]
        Dict of {term_id: anomaly_flags}
    n_terminals_required : int
        Minimum terminals agreeing for consensus (default: 2 of 3)

    Returns
    -------
    Dict
        - 'consensus': Consensus anomaly signal
        - 'agreement_fraction': Fraction where n+ terminals agree
        - 'n_terminals': Number of terminals
        - 'per_terminal_detections': Count per terminal
    """
    term_ids = sorted(terminal_anomalies.keys())
    n_terminals = len(term_ids)

    # Stack anomaly signals
    anomaly_matrix = np.column_stack(
        [terminal_anomalies[tid] for tid in term_ids]
    )

    # Count agreements
    n_agreeing = np.sum(anomaly_matrix, axis=1)
    
    # Consensus: n_terminals_required or more agree
    consensus = (n_agreeing >= n_terminals_required).astype(int)

    # Agreement fraction
    agreement_fraction = np.mean(n_agreeing >= n_terminals_required)

    # Per-terminal detection counts
    per_terminal = {
        tid: np.sum(terminal_anomalies[tid])
        for tid in term_ids
    }

    return {
        "consensus": consensus,
        "agreement_fraction": agreement_fraction,
        "n_terminals": n_terminals,
        "n_terminals_required": n_terminals_required,
        "per_terminal_detections": per_terminal,
        "terminal_ids": term_ids,
    }


def sensitivity_analysis(
    energy_scores: np.ndarray,
    subspace_scores: np.ndarray,
    thresholds_energy: Optional[List[float]] = None,
    thresholds_subspace: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Sensitivity analysis: How many detections at different thresholds?

    Parameters
    ----------
    energy_scores : np.ndarray
        Raw energy metric scores (not binary)
    subspace_scores : np.ndarray
        Raw subspace metric scores
    thresholds_energy : List[float], optional
        Test thresholds for energy metric
    thresholds_subspace : List[float], optional
        Test thresholds for subspace metric

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - Energy_Threshold
        - N_Energy_Detections
        - Subspace_Threshold
        - N_Subspace_Detections
    """
    if thresholds_energy is None:
        # Use percentiles: 90, 95, 99
        thresholds_energy = [
            np.percentile(energy_scores, p) for p in [90, 95, 99]
        ]

    if thresholds_subspace is None:
        thresholds_subspace = [
            np.percentile(subspace_scores, p) for p in [90, 95, 99]
        ]

    results = []

    for thresh_e in thresholds_energy:
        n_energy = np.sum(energy_scores >= thresh_e)
        for thresh_s in thresholds_subspace:
            n_subspace = np.sum(subspace_scores >= thresh_s)
            results.append({
                "Energy_Threshold": thresh_e,
                "N_Energy_Detections": n_energy,
                "Subspace_Threshold": thresh_s,
                "N_Subspace_Detections": n_subspace,
            })

    return pd.DataFrame(results)
