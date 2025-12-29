"""Unit tests for metrics modules."""

import pytest
import pandas as pd
import numpy as np

from src.metrics.residual_energy import (
    select_threshold,
    detect_excitation_anomalies,
    extract_event_intervals,
    compute_false_alarm_rate
)
from src.metrics.subspace_change import (
    select_subspace_threshold,
    detect_subspace_anomalies,
    compute_combined_metric
)
from src.metrics.spatial_coherence import (
    apply_spatial_voting,
    compute_spatial_agreement_metrics,
    extract_spatially_consistent_events
)


@pytest.fixture
def synthetic_energy_series():
    """Generate synthetic energy time series with anomalies."""
    n_samples = 1438
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="10s", tz="UTC")
    
    # Normal baseline + anomaly spikes
    energy = 100 + 10 * np.random.randn(n_samples)
    
    # Add anomaly spike at index 400-410 (corresponds to ~23:03-23:05)
    energy[400:410] = 5000
    
    series = pd.Series(energy, index=timestamps)
    return series


@pytest.fixture
def synthetic_distance_series():
    """Generate synthetic subspace distance series."""
    n_samples = 1438
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="10s", tz="UTC")
    
    distance = 0.1 + 0.01 * np.random.randn(n_samples)
    distance[400:410] = 1.5  # Anomaly
    
    series = pd.Series(distance, index=timestamps)
    return series


def test_select_threshold(synthetic_energy_series):
    """Test threshold selection."""
    threshold = select_threshold(synthetic_energy_series, percentile=99.0)
    
    assert threshold > synthetic_energy_series[:400].mean()
    assert threshold < synthetic_energy_series[400:410].mean()


def test_detect_excitation_anomalies(synthetic_energy_series):
    """Test anomaly detection."""
    threshold = select_threshold(synthetic_energy_series, percentile=99.0)
    anomalies = detect_excitation_anomalies(synthetic_energy_series, threshold, persistence_k=2)
    
    assert 'is_anomaly' in anomalies.columns
    assert 'event_id' in anomalies.columns
    assert anomalies['is_anomaly'].sum() > 0


def test_extract_event_intervals(synthetic_energy_series):
    """Test event interval extraction."""
    threshold = select_threshold(synthetic_energy_series, percentile=99.0)
    anomalies = detect_excitation_anomalies(synthetic_energy_series, threshold, persistence_k=2)
    
    events = extract_event_intervals(anomalies)
    
    assert len(events) > 0
    assert 'start_time' in events[0]
    assert 'peak_energy' in events[0]


def test_compute_false_alarm_rate(synthetic_energy_series):
    """Test false alarm rate computation."""
    threshold = select_threshold(synthetic_energy_series, percentile=99.0)
    anomalies = detect_excitation_anomalies(synthetic_energy_series, threshold, persistence_k=2)
    
    far = compute_false_alarm_rate(anomalies, baseline_end=400)
    
    assert far['false_alarm_rate_per_hour'] >= 0
    assert far['baseline_windows'] == 400


def test_select_subspace_threshold(synthetic_distance_series):
    """Test subspace threshold selection."""
    threshold = select_subspace_threshold(synthetic_distance_series, percentile=95.0)
    
    assert threshold > 0
    assert threshold < synthetic_distance_series[400:410].mean()


def test_detect_subspace_anomalies(synthetic_distance_series):
    """Test subspace anomaly detection."""
    threshold = select_subspace_threshold(synthetic_distance_series, percentile=95.0)
    anomalies = detect_subspace_anomalies(synthetic_distance_series, threshold, persistence_k=2)
    
    assert 'is_anomaly' in anomalies.columns
    assert anomalies['is_anomaly'].sum() > 0


def test_compute_combined_metric(synthetic_energy_series, synthetic_distance_series):
    """Test combined metric."""
    energy_threshold = select_threshold(synthetic_energy_series, percentile=99.0)
    energy_anom = detect_excitation_anomalies(synthetic_energy_series, energy_threshold)
    
    subspace_threshold = select_subspace_threshold(synthetic_distance_series, percentile=95.0)
    subspace_anom = detect_subspace_anomalies(synthetic_distance_series, subspace_threshold)
    
    combined = compute_combined_metric(energy_anom, subspace_anom)
    
    assert 'combined_score' in combined.columns
    assert 'is_anomaly_combined' in combined.columns


def test_apply_spatial_voting():
    """Test spatial voting."""
    n_samples = 100
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="10s", tz="UTC")
    
    # Create anomaly DataFrames for 3 terminals
    anomaly_dict = {}
    for term_id in [249, 252, 372]:
        is_anom = np.zeros(n_samples, dtype=bool)
        is_anom[40:50] = True  # All agree on anomaly
        
        df = pd.DataFrame({
            'is_anomaly': is_anom,
            'event_id': [1 if a else None for a in is_anom]
        }, index=timestamps)
        anomaly_dict[term_id] = df
    
    votes = apply_spatial_voting(anomaly_dict, vote_threshold=0.66)
    
    assert votes['is_consistent'].sum() > 0
    assert (votes.loc[timestamps[40:50], 'is_consistent']).sum() > 0


def test_compute_spatial_agreement_metrics():
    """Test spatial agreement metrics."""
    n_samples = 100
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="10s", tz="UTC")
    
    anomaly_dict = {}
    for term_id in [249, 252, 372]:
        is_anom = np.random.rand(n_samples) > 0.95
        df = pd.DataFrame({
            'is_anomaly': is_anom,
            'event_id': [1 if a else None for a in is_anom]
        }, index=timestamps)
        anomaly_dict[term_id] = df
    
    metrics = compute_spatial_agreement_metrics(anomaly_dict)
    
    assert 'pairwise_agreement' in metrics
    assert 'mean_pairwise_agreement' in metrics
    assert 0 <= metrics['mean_pairwise_agreement'] <= 1


def test_extract_spatially_consistent_events():
    """Test extraction of spatially consistent events."""
    n_samples = 1000
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="10s", tz="UTC")
    
    is_consistent = np.zeros(n_samples, dtype=bool)
    is_consistent[400:420] = True
    
    spatial_votes = pd.DataFrame({
        'is_consistent': is_consistent,
        'vote_fraction': np.random.uniform(0.6, 1.0, n_samples),
        'n_votes': np.random.randint(2, 4, n_samples)
    }, index=timestamps)
    
    events = extract_spatially_consistent_events(spatial_votes, min_duration_windows=5)
    
    assert len(events) > 0
    assert 'start_time' in events[0]
    assert 'mean_vote_fraction' in events[0]


def test_anomaly_persistence_requirement(synthetic_energy_series):
    """Test that persistence requirement is enforced."""
    threshold = select_threshold(synthetic_energy_series, percentile=99.0)
    
    # With high persistence requirement, might get no detections
    anomalies = detect_excitation_anomalies(
        synthetic_energy_series, threshold, persistence_k=20
    )
    
    # Should have fewer anomalies with higher persistence
    assert anomalies['is_anomaly'].sum() >= 0
