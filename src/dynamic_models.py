"""Dynamic modeling using VAR and subspace methods."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.decomposition import PCA
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


class VARModel:
    """Vector Autoregression model for multi-channel signals."""
    
    def __init__(self, order: int, n_features: int):
        """
        Initialize VAR model.
        
        Args:
            order: AR order (lag in samples)
            n_features: Number of variables (channels)
        """
        self.order = order
        self.n_features = n_features
        self.coeff = None  # Shape: (n_features, n_features * order)
        self.mean = None
        self.residuals = None
        self.cov_residuals = None
    
    def fit(self, X: np.ndarray) -> "VARModel":
        """
        Fit VAR model using OLS.
        
        Args:
            X: Data array (n_samples, n_features)
        
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")
        
        if n_samples <= self.order:
            raise ValueError(f"Not enough samples ({n_samples}) for order {self.order}")
        
        # Remove mean
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        
        # Build lagged design matrix
        X_lag = []
        Y = []
        
        for i in range(self.order, n_samples):
            # Stack lags [t-1, t-2, ..., t-order]
            lag_vec = X_centered[i - self.order:i][::-1].flatten()
            X_lag.append(lag_vec)
            Y.append(X_centered[i])
        
        X_lag = np.array(X_lag)  # (n_obs, n_features * order)
        Y = np.array(Y)  # (n_obs, n_features)
        
        # OLS: solve Y = X_lag @ coeff.T
        try:
            self.coeff = np.linalg.lstsq(X_lag, Y, rcond=None)[0].T
        except np.linalg.LinAlgError:
            logger.warning("VAR fitting failed; using regularized solution")
            reg = 1e-6
            self.coeff = np.linalg.solve(
                X_lag.T @ X_lag + reg * np.eye(X_lag.shape[1]),
                X_lag.T @ Y
            ).T
        
        # Compute residuals
        residuals = Y - (X_lag @ self.coeff.T)
        self.residuals = residuals
        self.cov_residuals = np.cov(residuals.T)
        
        logger.info(f"VAR({self.order}) fitted: {self.n_features} variables, "
                   f"{len(Y)} observations")
        
        return self
    
    def predict_residuals(self, X: np.ndarray) -> np.ndarray:
        """
        Predict residuals (forecast error) for new data.
        
        Args:
            X: Data array (n_samples, n_features)
        
        Returns:
            Residuals (n_samples - order, n_features)
        """
        if self.coeff is None:
            raise ValueError("Model not fitted yet")
        
        n_samples = X.shape[0]
        X_centered = X - self.mean
        
        residuals = []
        for i in range(self.order, n_samples):
            lag_vec = X_centered[i - self.order:i][::-1].flatten()
            pred = lag_vec @ self.coeff.T
            residual = X_centered[i] - pred
            residuals.append(residual)
        
        return np.array(residuals)


def fit_var_model(
    data: pd.DataFrame,
    order: int = 10,
    window_size: Optional[int] = None
) -> VARModel:
    """
    Fit VAR model to multi-channel signal.
    
    Args:
        data: DataFrame with channels as columns, datetime index
        order: AR order in samples
        window_size: If set, fit on first window_size samples
    
    Returns:
        Fitted VARModel
    """
    X = data.values
    
    if window_size is not None:
        X = X[:window_size]
    
    n_features = X.shape[1]
    model = VARModel(order, n_features)
    model.fit(X)
    
    return model


def compute_residual_excitation(
    data: pd.DataFrame,
    model: VARModel,
    window_size: int = 300,
    overlap_ratio: float = 0.5
) -> pd.DataFrame:
    """
    Compute rolling residual excitation energy.
    
    Args:
        data: DataFrame
        model: Fitted VARModel
        window_size: Window size in samples
        overlap_ratio: Overlap between windows (0 to 1)
    
    Returns:
        DataFrame with (timestamp, energy) for each window
    """
    X = data.values
    residuals = model.predict_residuals(X)
    
    # Add NaN padding for alignment
    residuals = np.vstack([np.full((model.order, X.shape[1]), np.nan), residuals])
    
    step = int(window_size * (1 - overlap_ratio))
    windows = []
    
    for i in range(0, len(residuals) - window_size, step):
        window = residuals[i:i + window_size]
        # Energy: sum of squared residuals, normalized by variance
        energy = np.nansum(window**2, axis=0).sum()
        
        # Use middle timestamp of window
        ts_idx = i + window_size // 2
        if ts_idx < len(data):
            ts = data.index[ts_idx]
            windows.append({'timestamp': ts, 'energy': energy, 'window_idx': i})
    
    if not windows:
        return pd.DataFrame(columns=['energy'])
    
    result = pd.DataFrame(windows)
    result.set_index('timestamp', inplace=True)
    
    logger.info(f"Computed {len(result)} residual energy windows")
    
    return result[['energy']]


def extract_dynamic_subspace(
    data: pd.DataFrame,
    n_components: int = 3,
    method: str = 'pca'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract dominant dynamic modes via PCA/SVD.
    
    Args:
        data: DataFrame
        n_components: Number of components
        method: 'pca' or 'svd'
    
    Returns:
        (basis vectors, singular values)
    """
    X = data.values
    
    # Center
    X_centered = X - X.mean(axis=0)
    
    if method == 'pca':
        pca = PCA(n_components=n_components)
        pca.fit(X_centered)
        basis = pca.components_.T
        singular_values = np.sqrt(pca.explained_variance_)
        
    elif method == 'svd':
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        basis = U[:, :n_components]
        singular_values = s[:n_components] / np.sqrt(len(X_centered))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Extracted {n_components} components via {method}")
    logger.info(f"Explained variance ratio: {singular_values / singular_values.sum()}")
    
    return basis, singular_values


def compute_subspace_distance(
    data: pd.DataFrame,
    baseline_basis: np.ndarray,
    window_size: int = 300,
    overlap_ratio: float = 0.5,
    method: str = 'principal_angles'
) -> pd.DataFrame:
    """
    Compute rolling subspace distance from baseline.
    
    Args:
        data: DataFrame
        baseline_basis: Baseline subspace (n_features, n_components)
        window_size: Window size in samples
        overlap_ratio: Overlap
        method: 'principal_angles' or 'projection_error'
    
    Returns:
        DataFrame with (timestamp, distance) for each window
    """
    X = data.values
    step = int(window_size * (1 - overlap_ratio))
    
    windows = []
    
    for i in range(0, len(X) - window_size, step):
        window = X[i:i + window_size]
        window_centered = window - window.mean(axis=0)
        
        # Extract subspace for this window
        U, _ = np.linalg.svd(window_centered, full_matrices=False)
        window_basis = U[:, :baseline_basis.shape[1]]
        
        # Compute distance
        if method == 'principal_angles':
            # Principal angles between subspaces
            M = baseline_basis.T @ window_basis
            s = np.linalg.svd(M, compute_uv=False)
            distance = np.sqrt(np.sum((np.arccos(np.clip(s, 0, 1)))**2))
        
        elif method == 'projection_error':
            # Projection error
            proj = window_basis @ window_basis.T @ baseline_basis
            distance = np.linalg.norm(baseline_basis - proj, 'fro')
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        ts_idx = i + window_size // 2
        if ts_idx < len(data):
            ts = data.index[ts_idx]
            windows.append({'timestamp': ts, 'distance': distance, 'window_idx': i})
    
    if not windows:
        return pd.DataFrame(columns=['distance'])
    
    result = pd.DataFrame(windows)
    result.set_index('timestamp', inplace=True)
    
    logger.info(f"Computed {len(result)} subspace distance windows")
    
    return result[['distance']]
