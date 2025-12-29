"""
Unit tests for validation module (Phase 4).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.validation.time_alignment import (
    compute_time_offsets,
    analyze_detection_timing,
    generate_timing_statistics,
)
from src.validation.internal_consistency import (
    compute_metric_agreement,
    generate_confusion_matrix,
    compute_cross_terminal_agreement,
    sensitivity_analysis,
)


class TestTimeAlignment:
    """Tests for time alignment module."""

    def test_compute_time_offsets_basic(self):
        """Test basic time offset computation."""
        labeled_time = pd.Timestamp("2020-08-31 22:57:00")
        detected_times = [
            pd.Timestamp("2020-08-31 23:20:30"),  # +23.5 min
            pd.Timestamp("2020-08-31 23:25:00"),  # +28 min
        ]

        offsets = compute_time_offsets(detected_times, labeled_time)

        assert len(offsets["offsets_minutes"]) == 2
        assert abs(offsets["offsets_minutes"][0] - 23.5) < 0.1
        assert abs(offsets["offsets_minutes"][1] - 28.0) < 0.1
        assert offsets["closest_offset"] is not None
        assert offsets["closest_index"] == 0  # First one is closest

    def test_compute_time_offsets_empty(self):
        """Test with no detections."""
        labeled_time = pd.Timestamp("2020-08-31 22:57:00")
        offsets = compute_time_offsets([], labeled_time)

        assert offsets["offsets_minutes"] == []
        assert offsets["closest_offset"] is None
        assert offsets["closest_index"] is None

    def test_generate_timing_statistics(self):
        """Test timing statistics generation."""
        timing_analysis = {
            "energy": {
                "offsets_minutes": [10, 20, 30],
                "closest_offset": 10,
            },
            "subspace": {
                "offsets_minutes": [],
                "closest_offset": None,
            },
        }

        stats = generate_timing_statistics(timing_analysis)

        assert len(stats) == 2
        assert stats.iloc[0]["N_Detections"] == 3
        assert stats.iloc[1]["N_Detections"] == 0
        assert abs(stats.iloc[0]["Mean_Offset_Minutes"] - 20.0) < 0.1


class TestInternalConsistency:
    """Tests for internal consistency module."""

    def test_compute_metric_agreement_perfect(self):
        """Test agreement with identical signals."""
        anomalies = np.array([0, 0, 1, 1, 1, 0, 0])

        agreement = compute_metric_agreement(anomalies, anomalies, anomalies)

        # Window-based overlap should be perfect
        assert agreement["energy_vs_subspace"] == 1.0
        assert agreement["energy_vs_spatial"] == 1.0
        # all_three ratio of anomalies = 3/7 since tolerance creates dilation
        assert agreement["all_three"] > 0.3

    def test_compute_metric_agreement_partial(self):
        """Test partial agreement."""
        signal1 = np.array([0, 0, 1, 1, 1, 0, 0])
        signal2 = np.array([0, 0, 0, 1, 1, 0, 0])  # Offset by 1

        agreement = compute_metric_agreement(signal1, signal2, signal1)

        # With tolerance, offset should still produce high overlap
        assert agreement["energy_vs_subspace"] >= 0.6
        assert agreement["all_three"] >= 0.2

    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation."""
        metric1 = np.array([1, 1, 0, 0, 1, 1])
        metric2 = np.array([1, 0, 0, 1, 1, 0])

        cm = generate_confusion_matrix(metric1, metric2, "M1", "M2")

        assert cm.shape == (2, 2)
        assert cm.iloc[0, 0] == 2  # Both anomaly
        assert cm.iloc[0, 1] == 2  # M1 anomaly, M2 normal
        assert cm.iloc[1, 0] == 1  # M1 normal, M2 anomaly

    def test_compute_cross_terminal_agreement_unanimous(self):
        """Test unanimous agreement."""
        terminal_anomalies = {
            "249": np.array([0, 0, 1, 1, 0]),
            "252": np.array([0, 0, 1, 1, 0]),
            "372": np.array([0, 0, 1, 1, 0]),
        }

        agreement = compute_cross_terminal_agreement(
            terminal_anomalies, n_terminals_required=2
        )

        assert agreement["consensus"].tolist() == [0, 0, 1, 1, 0]
        # agreement_fraction = fraction where n_terminals_required agree (2+ of 3)
        # At indices [0,1,4]: 0 agree; at [2,3]: 3 agree → 2/5 = 0.4
        assert abs(agreement["agreement_fraction"] - 0.4) < 0.01
        assert agreement["n_terminals"] == 3

    def test_compute_cross_terminal_agreement_partial(self):
        """Test partial agreement (2 of 3)."""
        terminal_anomalies = {
            "249": np.array([1, 0, 1, 0]),
            "252": np.array([1, 0, 0, 0]),
            "372": np.array([0, 0, 1, 0]),
        }

        agreement = compute_cross_terminal_agreement(
            terminal_anomalies, n_terminals_required=2
        )

        # At indices: 0 (2 agree), 1 (0), 2 (2 agree), 3 (0)
        assert agreement["consensus"].tolist() == [1, 0, 1, 0]
        assert abs(agreement["agreement_fraction"] - 0.5) < 0.01

    def test_sensitivity_analysis(self):
        """Test sensitivity analysis."""
        energy_scores = np.random.normal(100, 20, 1000)
        subspace_scores = np.random.normal(50, 10, 1000)

        sensitivity = sensitivity_analysis(
            energy_scores,
            subspace_scores,
            thresholds_energy=[100, 110, 120],
            thresholds_subspace=[45, 50, 55],
        )

        assert len(sensitivity) == 9  # 3 x 3 combinations
        assert "Energy_Threshold" in sensitivity.columns
        assert "N_Energy_Detections" in sensitivity.columns
        assert all(sensitivity["N_Energy_Detections"].diff().fillna(0) <= 0)  # Decreasing


class TestValidationIntegration:
    """Integration tests for validation pipeline."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        # Create synthetic detection results
        labeled_time = pd.Timestamp("2020-08-31 22:57:00")

        # Synthetic metric scores
        n_samples = 3600  # 2 hours at 1 Hz
        energy_scores = np.concatenate([
            np.random.normal(50, 10, 2700),  # Baseline
            np.random.normal(150, 30, 900),  # Event period
        ])
        subspace_scores = np.concatenate([
            np.random.normal(30, 5, 2700),
            np.random.normal(80, 20, 900),
        ])

        # Run sensitivity
        sens = sensitivity_analysis(
            energy_scores, subspace_scores,
            thresholds_energy=[50, 100],
            thresholds_subspace=[30, 60],
        )

        assert len(sens) == 4
        assert all(sens["N_Energy_Detections"] >= 0)

    def test_cross_terminal_multiple_terminals(self):
        """Test with many terminals."""
        n_terminals = 5
        n_samples = 100
        
        terminal_anomalies = {
            f"term_{i}": (np.random.random(n_samples) > 0.7).astype(int)
            for i in range(n_terminals)
        }

        agreement = compute_cross_terminal_agreement(
            terminal_anomalies, n_terminals_required=3
        )

        assert agreement["n_terminals"] == n_terminals
        assert 0 <= agreement["agreement_fraction"] <= 1.0
        assert len(agreement["per_terminal_detections"]) == n_terminals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
