"""Secondary anomaly detection metric: subspace change detection."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def select_subspace_threshold(
    distance_series: pd.Series,
    percentile: float = 95.0,
    baseline_end_idx: int = None
) -> float:
    """
    Select subspace distance threshold based on baseline.
    
    Args:
        distance_series: Subspace distance time series
        percentile: Percentile for threshold
        baseline_end_idx: If set, use only data before this index
    
    Returns:
        Threshold value
    """
    if baseline_end_idx is None:
        baseline_end_idx = len(distance_series)
    
    baseline = distance_series.iloc[:baseline_end_idx]
    threshold = baseline.quantile(percentile / 100.0)
    
    logger.info(f"Selected subspace threshold at {percentile}th percentile: {threshold:.4f}")
    
    return threshold


def detect_subspace_anomalies(
    distance_series: pd.Series,
    threshold: float,
    persistence_k: int = 2,
    min_gap_windows: int = 10
) -> pd.DataFrame:
    """
    Detect anomalies using subspace distance with persistence.
    
    Principle B: Detect structural changes in dynamic modes.
    
    Args:
        distance_series: Subspace distance time series
        threshold: Distance threshold
        persistence_k: Number of consecutive windows to require
        min_gap_windows: Minimum windows between events
    
    Returns:
        DataFrame with columns: timestamp, distance, is_anomaly, event_id
    """
    above_threshold = distance_series > threshold
    transitions = above_threshold.astype(int).diff().fillna(0)
    group_id = (transitions != 0).cumsum()
    
    result_rows = []
    last_group_end = -float('inf')
    event_count = 0
    
    for gid in group_id.unique():
        group_mask = group_id == gid
        group_distance = distance_series[group_mask]
        group_size = group_mask.sum()
        
        if above_threshold[group_mask].any():
            if group_size >= persistence_k:
                group_start_idx = np.where(group_mask)[0][0]
                
                if group_start_idx - last_group_end >= min_gap_windows:
                    event_count += 1
                    last_group_end = np.where(group_mask)[0][-1]
                    
                    for idx, (ts, dist) in enumerate(group_distance.items()):
                        result_rows.append({
                            'timestamp': ts,
                            'distance': dist,
                            'is_anomaly': True,
                            'event_id': event_count,
                            'position_in_event': idx + 1
                        })
        else:
            for ts, dist in group_distance.items():
                result_rows.append({
                    'timestamp': ts,
                    'distance': dist,
                    'is_anomaly': False,
                    'event_id': None,
                    'position_in_event': None
                })
    
    result = pd.DataFrame(result_rows)
    result.set_index('timestamp', inplace=True)
    
    anomaly_count = result['is_anomaly'].sum()
    event_count = result['event_id'].max() if result['event_id'].notna().any() else 0
    
    logger.info(f"Detected {anomaly_count} anomalous windows in {event_count} events (subspace)")
    
    return result


def compute_combined_metric(
    energy_anomalies: pd.DataFrame,
    subspace_anomalies: pd.DataFrame,
    weight_energy: float = 0.6,
    weight_subspace: float = 0.4
) -> pd.DataFrame:
    """
    Combine energy and subspace metrics for robust detection.
    
    Args:
        energy_anomalies: Output from detect_excitation_anomalies
        subspace_anomalies: Output from detect_subspace_anomalies
        weight_energy: Weight for energy metric
        weight_subspace: Weight for subspace metric
    
    Returns:
        DataFrame with combined anomaly score
    """
    # Align indices
    all_timestamps = sorted(set(energy_anomalies.index) | set(subspace_anomalies.index))
    
    combined = pd.DataFrame(index=pd.DatetimeIndex(all_timestamps))
    
    # Reindex and fill missing
    energy_reindexed = energy_anomalies.reindex(all_timestamps, fill_value=False)
    subspace_reindexed = subspace_anomalies.reindex(all_timestamps, fill_value=False)
    
    # Convert boolean to scores
    energy_score = energy_reindexed['is_anomaly'].astype(float)
    subspace_score = subspace_reindexed['is_anomaly'].astype(float)
    
    # Combine
    combined['energy_score'] = energy_score
    combined['subspace_score'] = subspace_score
    combined['combined_score'] = (weight_energy * energy_score + weight_subspace * subspace_score)
    combined['is_anomaly_combined'] = combined['combined_score'] > 0.5
    
    logger.info(f"Combined metrics: agreement on {(combined['is_anomaly_combined']).sum()} windows")
    
    return combined
