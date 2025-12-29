"""Signal preprocessing and channel selection."""

import pandas as pd
import numpy as np
from typing import Optional
from scipy import signal as sp_signal
import logging

logger = logging.getLogger(__name__)


def preprocess_pmu_signals(
    df: pd.DataFrame,
    channels: Optional[list] = None,
    remove_dc: bool = False,
    bandpass_hz: Optional[tuple] = None,
    interpolate_missing: bool = True
) -> pd.DataFrame:
    """
    Clean and prepare PMU signals for analysis.
    
    Args:
        df: PMU DataFrame with datetime index
        channels: List of columns to process (default: all numeric)
        remove_dc: If True, remove DC offset
        bandpass_hz: (low, high) for bandpass filter (applied after other steps)
        interpolate_missing: If True, linearly interpolate NaNs
    
    Returns:
        Preprocessed DataFrame
    """
    if channels is None:
        channels = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result = df[channels].copy()
    
    # Interpolate missing values
    if interpolate_missing:
        n_nan_before = result.isna().sum().sum()
        result = result.interpolate(method='linear', limit_direction='both')
        result = result.fillna(method='bfill').fillna(method='ffill')
        n_nan_after = result.isna().sum().sum()
        
        if n_nan_before > 0:
            logger.info(f"Interpolated {n_nan_before - n_nan_after} missing values")
    
    # Remove DC offset (mean)
    if remove_dc:
        result = result - result.mean()
    
    # Bandpass filtering
    if bandpass_hz is not None:
        low_hz, high_hz = bandpass_hz
        # Assume 30 Hz sampling rate
        nyquist = 15.0  # 30 / 2
        
        low_norm = low_hz / nyquist
        high_norm = high_hz / nyquist
        
        # Clamp to valid range (0, 1)
        low_norm = max(0.001, min(0.999, low_norm))
        high_norm = max(0.001, min(0.999, high_norm))
        
        if low_norm < high_norm:
            sos = sp_signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
            result = result.apply(lambda x: sp_signal.sosfilt(sos, x), axis=0)
            logger.info(f"Applied bandpass filter [{low_hz}, {high_hz}] Hz")
    
    return result


def select_analysis_channels(
    df: pd.DataFrame,
    mode: str = 'voltage_magnitude'
) -> pd.DataFrame:
    """
    Extract relevant channels for analysis.
    
    Args:
        df: PMU DataFrame
        mode: 'voltage_magnitude', 'current_magnitude', 'frequency', 'all'
    
    Returns:
        DataFrame with selected channels
    """
    voltage_mag = ['VP_M', 'VA_M', 'VB_M', 'VC_M']
    voltage_ang = ['VP_A', 'VA_A', 'VB_A', 'VC_A']
    current_mag = ['IP_M', 'IA_M', 'IB_M', 'IC_M']
    current_ang = ['IP_A', 'IA_A', 'IB_A', 'IC_A']
    freq = ['F', 'DF']
    
    channels = []
    
    if mode == 'voltage_magnitude':
        channels = voltage_mag
    elif mode == 'current_magnitude':
        channels = current_mag
    elif mode == 'voltage_and_current':
        channels = voltage_mag + current_mag
    elif mode == 'all_magnitude':
        channels = voltage_mag + current_mag
    elif mode == 'all':
        channels = voltage_mag + voltage_ang + current_mag + current_ang + freq
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Filter to available columns
    available = [c for c in channels if c in df.columns]
    
    if not available:
        logger.warning(f"No channels found for mode {mode}; returning all numeric columns")
        return df.select_dtypes(include=[np.number])
    
    result = df[available].copy()
    logger.info(f"Selected {len(available)} channels: {available}")
    
    return result


def compute_signal_statistics(df: pd.DataFrame) -> dict:
    """
    Compute basic statistics on signals.
    
    Returns:
        Dict with mean, std, min, max per channel
    """
    stats = {
        'mean': df.mean().to_dict(),
        'std': df.std().to_dict(),
        'min': df.min().to_dict(),
        'max': df.max().to_dict(),
        'rms': (df**2).mean().pow(0.5).to_dict()
    }
    
    return stats


def detect_outliers_zscore(
    df: pd.DataFrame,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers using z-score method.
    
    Args:
        df: Data frame
        threshold: Z-score threshold
    
    Returns:
        Boolean DataFrame (True = outlier)
    """
    from scipy import stats
    
    z = np.abs(stats.zscore(df, nan_policy='omit'))
    return z > threshold
