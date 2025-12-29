"""Unit tests for preprocessing.py."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.preprocessing import (
    preprocess_pmu_signals,
    select_analysis_channels,
    compute_signal_statistics,
    detect_outliers_zscore
)


@pytest.fixture
def synthetic_pmu_df():
    """Generate synthetic PMU data."""
    n_samples = 1800  # 60 seconds @ 30 Hz
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="33.33ms", tz="UTC")
    
    data = {
        'VP_M': 40000 + 100 * np.sin(2 * np.pi * 0.1 * np.arange(n_samples) / 30),
        'VA_M': 40000 + 100 * np.sin(2 * np.pi * 0.1 * np.arange(n_samples) / 30 + np.pi/3),
        'VB_M': 40000 + 100 * np.sin(2 * np.pi * 0.1 * np.arange(n_samples) / 30 + 2*np.pi/3),
        'VC_M': 40000 + 100 * np.sin(2 * np.pi * 0.1 * np.arange(n_samples) / 30 + 4*np.pi/3),
        'IP_M': 60 + 10 * np.sin(2 * np.pi * 0.1 * np.arange(n_samples) / 30),
        'F': 60.0 + 0.1 * np.sin(2 * np.pi * 0.05 * np.arange(n_samples) / 30),
    }
    
    df = pd.DataFrame(data, index=timestamps)
    return df


def test_select_voltage_magnitude(synthetic_pmu_df):
    """Test voltage magnitude channel selection."""
    result = select_analysis_channels(synthetic_pmu_df, mode='voltage_magnitude')
    
    assert list(result.columns) == ['VP_M', 'VA_M', 'VB_M', 'VC_M']
    assert len(result) == len(synthetic_pmu_df)


def test_select_current_magnitude(synthetic_pmu_df):
    """Test current magnitude channel selection."""
    result = select_analysis_channels(synthetic_pmu_df, mode='current_magnitude')
    
    assert 'IP_M' in result.columns
    assert 'VP_M' not in result.columns


def test_select_voltage_and_current(synthetic_pmu_df):
    """Test combined voltage and current."""
    result = select_analysis_channels(synthetic_pmu_df, mode='voltage_and_current')
    
    assert 'VP_M' in result.columns
    assert 'IP_M' in result.columns


def test_preprocess_removes_nan(synthetic_pmu_df):
    """Test NaN interpolation."""
    # Add some NaNs
    df_with_nan = synthetic_pmu_df.copy()
    df_with_nan.iloc[100:110, 0] = np.nan
    
    result = preprocess_pmu_signals(df_with_nan, interpolate_missing=True)
    
    assert result.isna().sum().sum() == 0


def test_preprocess_dc_removal(synthetic_pmu_df):
    """Test DC offset removal."""
    channels = ['VP_M', 'VA_M']
    result = preprocess_pmu_signals(synthetic_pmu_df, channels=channels, remove_dc=True)
    
    # Mean should be close to zero
    assert np.allclose(result.mean(axis=0), 0, atol=1)


def test_compute_statistics(synthetic_pmu_df):
    """Test signal statistics computation."""
    stats = compute_signal_statistics(synthetic_pmu_df)
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'rms' in stats
    assert 'VP_M' in stats['mean']


def test_detect_outliers(synthetic_pmu_df):
    """Test outlier detection."""
    df_with_outliers = synthetic_pmu_df.copy()
    df_with_outliers.iloc[100, 0] = 100000  # Obvious outlier
    
    outliers = detect_outliers_zscore(df_with_outliers, threshold=3.0)
    
    assert outliers.iloc[100, 0]  # Outlier at row 100
    assert not outliers.iloc[0, 0]  # Normal point


def test_unknown_mode(synthetic_pmu_df):
    """Test error on unknown mode."""
    with pytest.raises(ValueError):
        select_analysis_channels(synthetic_pmu_df, mode='unknown_mode')


def test_preprocess_preserves_index(synthetic_pmu_df):
    """Test that preprocessing preserves datetime index."""
    result = preprocess_pmu_signals(synthetic_pmu_df)
    
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.equals(synthetic_pmu_df.index)
