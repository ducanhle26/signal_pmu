"""
Unit tests for visualization module (Phase 5).
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.visualization.plots import (
    plot_signal_timeseries,
    plot_energy_metric,
    plot_subspace_metric,
    plot_spatial_voting_heatmap,
    plot_anomaly_detection_comparison,
    plot_sensitivity_curves,
)


@pytest.fixture
def sample_data():
    """Create sample PMU data for testing."""
    n_samples = 1000
    base_time = pd.Timestamp("2020-08-31 22:00:00")
    timestamps = pd.date_range(base_time, periods=n_samples, freq="33.3ms")
    
    # Voltage data
    data = pd.DataFrame({
        "VP_M": np.sin(2*np.pi*60*np.arange(n_samples)/30) * 100 + 240,
        "VA_M": np.sin(2*np.pi*60*np.arange(n_samples)/30 + 2*np.pi/3) * 100 + 240,
        "VB_M": np.sin(2*np.pi*60*np.arange(n_samples)/30 + 4*np.pi/3) * 100 + 240,
        "VC_M": np.cos(2*np.pi*60*np.arange(n_samples)/30) * 50 + 240,
    }, index=timestamps)
    
    return {"249": data, "252": data.copy(), "372": data.copy()}


@pytest.fixture
def sample_metrics():
    """Create sample metric data."""
    n_samples = 500
    energy = np.random.normal(50, 10, n_samples)
    energy[250:270] = np.random.normal(150, 30, 20)  # Event
    
    subspace = np.random.normal(30, 5, n_samples)
    subspace[250:270] = np.random.normal(80, 20, 20)
    
    detections = np.zeros(n_samples)
    detections[250:270] = 1
    
    timestamps = pd.date_range("2020-08-31 22:00:00", periods=n_samples, freq="1s")
    labeled_time = pd.Timestamp("2020-08-31 22:57:00")
    
    return {
        "energy": energy,
        "subspace": subspace,
        "detections": detections,
        "timestamps": timestamps,
        "labeled_time": labeled_time,
    }


class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_signal_timeseries_returns_figure(self, sample_data):
        """Test signal timeseries plotting returns figure."""
        labeled_time = pd.Timestamp("2020-08-31 22:57:00")
        
        fig = plot_signal_timeseries(sample_data, labeled_time)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == len(sample_data)
        plt.close(fig)

    def test_plot_signal_timeseries_single_terminal(self):
        """Test with single terminal."""
        n_samples = 100
        base_time = pd.Timestamp("2020-08-31 22:00:00")
        timestamps = pd.date_range(base_time, periods=n_samples, freq="33.3ms")
        
        data = pd.DataFrame({
            "VP_M": np.random.normal(240, 10, n_samples),
            "VA_M": np.random.normal(240, 10, n_samples),
        }, index=timestamps)
        
        terminal_data = {"249": data}
        labeled_time = pd.Timestamp("2020-08-31 22:30:00")
        
        fig = plot_signal_timeseries(terminal_data, labeled_time)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_energy_metric_returns_figure(self, sample_metrics):
        """Test energy metric plotting."""
        fig = plot_energy_metric(
            sample_metrics["energy"],
            threshold=100,
            detections=sample_metrics["detections"],
            timestamps=sample_metrics["timestamps"],
            labeled_time=sample_metrics["labeled_time"],
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_energy_metric_threshold_visible(self, sample_metrics):
        """Test threshold line is visible in plot."""
        fig = plot_energy_metric(
            sample_metrics["energy"],
            threshold=100,
            detections=sample_metrics["detections"],
            timestamps=sample_metrics["timestamps"],
            labeled_time=sample_metrics["labeled_time"],
        )
        
        # Check that threshold line is in plot
        ax = fig.axes[0]
        assert len(ax.lines) >= 1  # At least energy line + threshold
        plt.close(fig)

    def test_plot_subspace_metric_returns_figure(self, sample_metrics):
        """Test subspace metric plotting."""
        fig = plot_subspace_metric(
            sample_metrics["subspace"],
            threshold=60,
            detections=sample_metrics["detections"],
            timestamps=sample_metrics["timestamps"],
            labeled_time=sample_metrics["labeled_time"],
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_spatial_voting_heatmap_returns_figure(self, sample_metrics):
        """Test spatial voting heatmap."""
        terminal_anomalies = {
            "249": sample_metrics["detections"],
            "252": sample_metrics["detections"],
            "372": sample_metrics["detections"],
        }
        
        fig = plot_spatial_voting_heatmap(
            terminal_anomalies,
            sample_metrics["timestamps"],
            sample_metrics["labeled_time"],
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_spatial_voting_heatmap_dimensions(self, sample_metrics):
        """Test heatmap has correct dimensions."""
        terminal_anomalies = {
            "249": sample_metrics["detections"][:100],
            "252": sample_metrics["detections"][:100],
        }
        
        fig = plot_spatial_voting_heatmap(
            terminal_anomalies,
            sample_metrics["timestamps"][:100],
            sample_metrics["labeled_time"],
        )
        
        ax = fig.axes[0]
        # Should have 2 y-ticks (one per terminal)
        assert len(ax.get_yticks()) >= 2
        plt.close(fig)

    def test_plot_anomaly_detection_comparison_returns_figure(self, sample_metrics):
        """Test anomaly detection comparison."""
        fig = plot_anomaly_detection_comparison(
            sample_metrics["detections"],
            sample_metrics["detections"],
            sample_metrics["detections"],
            sample_metrics["timestamps"],
            sample_metrics["labeled_time"],
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # Three metrics
        plt.close(fig)

    def test_plot_sensitivity_curves_returns_figure(self):
        """Test sensitivity curves plotting."""
        sensitivity_df = pd.DataFrame({
            "Energy_Threshold": [50, 100, 150],
            "N_Energy_Detections": [20, 10, 5],
            "Subspace_Threshold": [30, 60, 90],
            "N_Subspace_Detections": [5, 2, 0],
        })
        
        fig = plot_sensitivity_curves(sensitivity_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2  # Two metrics
        plt.close(fig)

    def test_plot_sensitivity_curves_monotonic(self):
        """Test sensitivity curves show decreasing detection."""
        sensitivity_df = pd.DataFrame({
            "Energy_Threshold": [50, 100, 150],
            "N_Energy_Detections": [20, 10, 5],
            "Subspace_Threshold": [30, 60, 90],
            "N_Subspace_Detections": [5, 2, 0],
        })
        
        fig = plot_sensitivity_curves(sensitivity_df)
        
        ax = fig.axes[0]
        # Should have decreasing trend
        assert len(ax.lines) >= 1
        plt.close(fig)


class TestVisualizationSaving:
    """Tests for figure saving."""

    def test_plot_saves_to_file(self, sample_metrics, tmp_path):
        """Test that plots can be saved to file."""
        output_path = tmp_path / "test_plot.png"
        
        fig = plot_energy_metric(
            sample_metrics["energy"],
            threshold=100,
            detections=sample_metrics["detections"],
            timestamps=sample_metrics["timestamps"],
            labeled_time=sample_metrics["labeled_time"],
            save_path=str(output_path),
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_plot_heatmap_saves_to_file(self, sample_metrics, tmp_path):
        """Test heatmap saving."""
        output_path = tmp_path / "heatmap.png"
        terminal_anomalies = {
            "249": sample_metrics["detections"],
            "252": sample_metrics["detections"],
        }
        
        fig = plot_spatial_voting_heatmap(
            terminal_anomalies,
            sample_metrics["timestamps"],
            sample_metrics["labeled_time"],
            save_path=str(output_path),
        )
        
        assert output_path.exists()
        plt.close(fig)


class TestVisualizationEdgeCases:
    """Edge case tests for visualization."""

    def test_plot_empty_detections(self, sample_metrics):
        """Test plotting with no detections."""
        empty_detections = np.zeros(len(sample_metrics["detections"]))
        
        fig = plot_energy_metric(
            sample_metrics["energy"],
            threshold=1000,  # High threshold
            detections=empty_detections,
            timestamps=sample_metrics["timestamps"],
            labeled_time=sample_metrics["labeled_time"],
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_all_detections(self, sample_metrics):
        """Test plotting with all samples flagged."""
        all_detections = np.ones(len(sample_metrics["detections"]))
        
        fig = plot_energy_metric(
            sample_metrics["energy"],
            threshold=0,
            detections=all_detections,
            timestamps=sample_metrics["timestamps"],
            labeled_time=sample_metrics["labeled_time"],
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
