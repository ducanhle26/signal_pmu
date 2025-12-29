"""Unit tests for dynamic_models.py."""

import pytest
import pandas as pd
import numpy as np

from src.dynamic_models import (
    VARModel,
    fit_var_model,
    compute_residual_excitation,
    extract_dynamic_subspace,
    compute_subspace_distance
)


@pytest.fixture
def synthetic_multivariate_data():
    """Generate synthetic multi-channel time series."""
    n_samples = 1200
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="33.33ms", tz="UTC")
    
    # Create correlated signals
    t = np.arange(n_samples) / 30.0
    data = {
        'ch1': 10 * np.sin(2 * np.pi * 0.1 * t) + np.random.randn(n_samples) * 0.5,
        'ch2': 10 * np.cos(2 * np.pi * 0.1 * t) + np.random.randn(n_samples) * 0.5,
        'ch3': 5 * np.sin(2 * np.pi * 0.05 * t) + np.random.randn(n_samples) * 0.5,
    }
    
    df = pd.DataFrame(data, index=timestamps)
    return df


def test_var_model_init():
    """Test VAR model initialization."""
    model = VARModel(order=10, n_features=3)
    
    assert model.order == 10
    assert model.n_features == 3
    assert model.coeff is None


def test_var_model_fit(synthetic_multivariate_data):
    """Test VAR model fitting."""
    model = VARModel(order=10, n_features=3)
    model.fit(synthetic_multivariate_data.values)
    
    assert model.coeff is not None
    assert model.coeff.shape == (3, 30)  # (n_features, order * n_features)
    assert model.residuals is not None
    assert model.cov_residuals is not None


def test_var_model_predict_residuals(synthetic_multivariate_data):
    """Test VAR residual prediction."""
    model = VARModel(order=10, n_features=3)
    model.fit(synthetic_multivariate_data.values)
    
    residuals = model.predict_residuals(synthetic_multivariate_data.values)
    
    assert residuals.shape[0] == len(synthetic_multivariate_data) - model.order
    assert residuals.shape[1] == 3


def test_var_model_insufficient_data():
    """Test error on insufficient data."""
    model = VARModel(order=100, n_features=3)
    small_data = np.random.randn(50, 3)
    
    with pytest.raises(ValueError):
        model.fit(small_data)


def test_fit_var_model(synthetic_multivariate_data):
    """Test convenience function for VAR fitting."""
    model = fit_var_model(synthetic_multivariate_data, order=10)
    
    assert model.coeff is not None
    assert model.mean is not None


def test_fit_var_model_with_window(synthetic_multivariate_data):
    """Test VAR fitting with window."""
    model = fit_var_model(synthetic_multivariate_data, order=10, window_size=500)
    
    assert model.coeff is not None


def test_compute_residual_excitation(synthetic_multivariate_data):
    """Test residual excitation computation."""
    model = fit_var_model(synthetic_multivariate_data, order=10)
    excitation = compute_residual_excitation(
        synthetic_multivariate_data, model, window_size=300, overlap_ratio=0.5
    )
    
    assert 'energy' in excitation.columns
    assert len(excitation) > 0
    assert excitation['energy'].dtype == np.float64


def test_extract_dynamic_subspace_pca(synthetic_multivariate_data):
    """Test subspace extraction via PCA."""
    basis, singular_values = extract_dynamic_subspace(
        synthetic_multivariate_data, n_components=2, method='pca'
    )
    
    assert basis.shape == (3, 2)
    assert len(singular_values) == 2
    assert all(s > 0 for s in singular_values)


def test_extract_dynamic_subspace_svd(synthetic_multivariate_data):
    """Test subspace extraction via SVD."""
    basis, singular_values = extract_dynamic_subspace(
        synthetic_multivariate_data, n_components=2, method='svd'
    )
    
    assert basis.shape == (3, 2)
    assert len(singular_values) == 2


def test_extract_subspace_invalid_method(synthetic_multivariate_data):
    """Test error on invalid method."""
    with pytest.raises(ValueError):
        extract_dynamic_subspace(synthetic_multivariate_data, method='invalid')


def test_compute_subspace_distance(synthetic_multivariate_data):
    """Test rolling subspace distance."""
    # Get baseline from first half
    baseline_data = synthetic_multivariate_data.iloc[:600]
    baseline_basis, _ = extract_dynamic_subspace(baseline_data, n_components=2)
    
    # Compute distance for full data
    distance = compute_subspace_distance(
        synthetic_multivariate_data, baseline_basis, window_size=300,
        overlap_ratio=0.5, method='principal_angles'
    )
    
    assert 'distance' in distance.columns
    assert len(distance) > 0


def test_compute_subspace_distance_projection_error(synthetic_multivariate_data):
    """Test subspace distance with projection error method."""
    baseline_data = synthetic_multivariate_data.iloc[:600]
    baseline_basis, _ = extract_dynamic_subspace(baseline_data, n_components=2)
    
    distance = compute_subspace_distance(
        synthetic_multivariate_data, baseline_basis, window_size=300,
        overlap_ratio=0.5, method='projection_error'
    )
    
    assert 'distance' in distance.columns
    assert len(distance) > 0


def test_var_model_wrong_feature_count():
    """Test error on wrong feature count."""
    model = VARModel(order=10, n_features=3)
    model.fit(np.random.randn(100, 3))
    
    with pytest.raises(ValueError):
        model.predict_residuals(np.random.randn(100, 4))
