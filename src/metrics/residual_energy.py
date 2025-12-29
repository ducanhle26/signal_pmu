"""Primary anomaly detection metric: residual excitation energy."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def select_threshold(
    energy_series: pd.Series,
    percentile: float = 99.0,
    baseline_end_idx: int = None
) -> float:
    """
    Select threshold based on baseline data.
    
    Args:
        energy_series: Energy time series
        percentile: Percentile for threshold (default: 99th)
        baseline_end_idx: If set, use only data before this index for baseline
    
    Returns:
        Threshold value
    """
    if baseline_end_idx is None:
        baseline_end_idx = len(energy_series)
    
    baseline = energy_series.iloc[:baseline_end_idx]
    threshold = baseline.quantile(percentile / 100.0)
    
    logger.info(f"Selected threshold at {percentile}th percentile: {threshold:.2f}")
    
    return threshold


def detect_excitation_anomalies(
    energy_series: pd.Series,
    threshold: float,
    persistence_k: int = 2,
    min_gap_windows: int = 10
) -> pd.DataFrame:
    """
    Detect anomalies using excitation energy with persistence requirement.
    
    Principle A: Flag windows exceeding threshold with K consecutive windows required.
    
    Args:
        energy_series: Energy time series (pd.Series with datetime index)
        threshold: Energy threshold
        persistence_k: Number of consecutive windows to require
        min_gap_windows: Minimum windows between events
    
    Returns:
        DataFrame with columns: timestamp, energy, is_anomaly, group_id
    """
    # Find windows above threshold
    above_threshold = energy_series > threshold
    
    # Find consecutive groups
    transitions = above_threshold.astype(int).diff().fillna(0)
    group_id = (transitions != 0).cumsum()
    
    # Count windows in each group
    result_rows = []
    last_group_end = -float('inf')
    event_count = 0
    
    for gid in group_id.unique():
        group_mask = group_id == gid
        group_energy = energy_series[group_mask]
        group_size = group_mask.sum()
        
        if above_threshold[group_mask].any():
            # This is an "above threshold" group
            if group_size >= persistence_k:
                # Check minimum gap from last event
                group_start_idx = np.where(group_mask)[0][0]
                
                if group_start_idx - last_group_end >= min_gap_windows:
                    event_count += 1
                    last_group_end = np.where(group_mask)[0][-1]
                    
                    for idx, (ts, energy) in enumerate(group_energy.items()):
                        result_rows.append({
                            'timestamp': ts,
                            'energy': energy,
                            'is_anomaly': True,
                            'event_id': event_count,
                            'position_in_event': idx + 1,
                            'event_size': group_size
                        })
        else:
            # Normal period
            for ts, energy in group_energy.items():
                result_rows.append({
                    'timestamp': ts,
                    'energy': energy,
                    'is_anomaly': False,
                    'event_id': None,
                    'position_in_event': None,
                    'event_size': None
                })
    
    result = pd.DataFrame(result_rows)
    result.set_index('timestamp', inplace=True)
    
    anomaly_count = result['is_anomaly'].sum()
    event_count = result['event_id'].max() if result['event_id'].notna().any() else 0
    
    logger.info(f"Detected {anomaly_count} anomalous windows in {event_count} events")
    
    return result


def extract_event_intervals(anomaly_df: pd.DataFrame) -> list:
    """
    Extract start/end timestamps for detected events.
    
    Args:
        anomaly_df: Output from detect_excitation_anomalies
    
    Returns:
        List of dicts with event_id, start_time, end_time, duration_sec, peak_energy, n_windows
    """
    events = []
    
    for event_id in sorted(anomaly_df['event_id'].dropna().unique()):
        event_rows = anomaly_df[anomaly_df['event_id'] == event_id]
        
        start_time = event_rows.index[0]
        end_time = event_rows.index[-1]
        duration_sec = (end_time - start_time).total_seconds()
        peak_energy = event_rows['energy'].max()
        n_windows = len(event_rows)
        
        events.append({
            'event_id': int(event_id),
            'start_time': start_time,
            'end_time': end_time,
            'duration_sec': duration_sec,
            'peak_energy': peak_energy,
            'n_windows': n_windows
        })
    
    logger.info(f"Extracted {len(events)} events")
    
    return events


def compute_false_alarm_rate(
    anomaly_df: pd.DataFrame,
    baseline_end: int = None,
    window_duration_sec: float = 10.0
) -> dict:
    """
    Compute false alarm rate in baseline period.
    
    Args:
        anomaly_df: Output from detect_excitation_anomalies
        baseline_end: End index for baseline (default: 2700 windows = 7.5 hours)
        window_duration_sec: Duration of each window
    
    Returns:
        Dict with false_alarms, baseline_windows, false_alarm_rate_per_hour
    """
    if baseline_end is None:
        # Default: first 2700 windows (7.5 hours @ 10 sec/window)
        baseline_end = min(2700, len(anomaly_df))
    
    baseline = anomaly_df.iloc[:baseline_end]
    n_false_alarms = baseline['is_anomaly'].sum()
    baseline_hours = len(baseline) * window_duration_sec / 3600.0
    
    rate_per_hour = n_false_alarms / baseline_hours if baseline_hours > 0 else 0
    
    logger.info(f"False alarm rate: {n_false_alarms} alarms in {baseline_hours:.1f} hours "
               f"= {rate_per_hour:.2f} per hour")
    
    return {
        'false_alarms': n_false_alarms,
        'baseline_windows': len(baseline),
        'baseline_hours': baseline_hours,
        'false_alarm_rate_per_hour': rate_per_hour
    }
