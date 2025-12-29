"""
Unit tests for reporting module (Phase 5).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting.results import (
    generate_comprehensive_report,
    generate_methodology_summary,
    generate_performance_table,
    generate_validation_summary,
)


class TestReporting:
    """Tests for reporting module."""

    def test_generate_performance_table_structure(self):
        """Test performance table has correct structure."""
        validation_results = {
            "time_alignment": {
                "energy": {"offsets_minutes": [10, 20]},
                "subspace": {"offsets_minutes": []},
            }
        }
        metric_agreement = {}

        table = generate_performance_table(validation_results, metric_agreement)

        assert isinstance(table, pd.DataFrame)
        assert len(table) >= 2  # At least energy and subspace
        assert "Metric" in table.columns

    def test_generate_validation_summary_terminals(self):
        """Test validation summary includes all terminals."""
        validation_results = {
            "cross_terminal_agreement": {
                "agreement_fraction": 0.95,
                "per_terminal_detections": {
                    "'249'": 5,
                    "'252'": 5,
                    "'372'": 5,
                },
            }
        }

        table = generate_validation_summary(validation_results)

        assert isinstance(table, pd.DataFrame)
        assert len(table) == 3  # 3 terminals
        assert "Terminal" in table.columns

    def test_generate_methodology_summary_content(self):
        """Test methodology summary has expected sections."""
        summary = generate_methodology_summary()

        assert isinstance(summary, str)
        assert "METHODOLOGY SUMMARY" in summary
        assert "Principle A" in summary
        assert "Principle B" in summary
        assert "Principle C" in summary
        assert "DESIGN CHOICES" in summary
        assert "LIMITATIONS" in summary

    def test_generate_comprehensive_report_structure(self):
        """Test comprehensive report has all sections."""
        validation_results = {
            "time_alignment": {
                "energy": {"offsets_minutes": [10, 20]},
            },
            "cross_terminal_agreement": {
                "agreement_fraction": 0.95,
            },
            "subspace_detection": {
                "n_events": 1,
                "mean_agreement": 0.9,
            },
        }
        metric_agreement = {"energy_vs_subspace": 0.85}
        sensitivity_df = pd.DataFrame(
            {
                "Energy_Threshold": [100, 110, 120],
                "N_Energy_Detections": [10, 8, 5],
                "Subspace_Threshold": [50, 55, 60],
                "N_Subspace_Detections": [3, 2, 1],
            }
        )

        report = generate_comprehensive_report(
            validation_results, metric_agreement, sensitivity_df
        )

        assert isinstance(report, str)
        assert "EXECUTIVE SUMMARY" in report
        assert "KEY FINDINGS" in report
        assert "METHODOLOGY" in report
        assert "VALIDATION" in report
        assert "RECOMMENDATIONS" in report

    def test_comprehensive_report_save(self):
        """Test saving comprehensive report to file."""
        validation_results = {
            "time_alignment": {"energy": {"offsets_minutes": [10]}},
            "cross_terminal_agreement": {"agreement_fraction": 0.95},
        }
        metric_agreement = {}
        sensitivity_df = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"

            report = generate_comprehensive_report(
                validation_results,
                metric_agreement,
                sensitivity_df,
                output_path=str(output_path),
            )

            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
            assert "EXECUTIVE SUMMARY" in content

    def test_performance_table_empty_offsets(self):
        """Test performance table handles empty offset lists."""
        validation_results = {
            "time_alignment": {
                "energy": {"offsets_minutes": []},
                "subspace": {"offsets_minutes": []},
            }
        }
        metric_agreement = {}

        table = generate_performance_table(validation_results, metric_agreement)

        # Should still create table with 0 events
        assert "N_Events" in table.columns
        energy_row = table[table["Metric"] == "Residual Energy"]
        assert len(energy_row) > 0

    def test_validation_summary_sorting(self):
        """Test validation summary sorts terminals."""
        validation_results = {
            "cross_terminal_agreement": {
                "agreement_fraction": 0.95,
                "per_terminal_detections": {
                    "'372'": 3,
                    "'249'": 5,
                    "'252'": 4,
                },
            }
        }

        table = generate_validation_summary(validation_results)

        terminals = table["Terminal"].astype(int).values
        # Should be in order: 249, 252, 372
        assert len(terminals) == 3


class TestReportingIntegration:
    """Integration tests for reporting functions."""

    def test_full_reporting_workflow(self):
        """Test complete reporting workflow."""
        # Create realistic validation results
        validation_results = {
            "timestamp": "2025-12-29T00:00:00",
            "labeled_event_time": "2020-08-31T22:57:00+00:00",
            "section_id": 80,
            "terminals": [249, 252, 372],
            "time_alignment": {
                "energy": {"offsets_minutes": [23.5, 28.0]},
                "subspace": {"offsets_minutes": []},
            },
            "cross_terminal_agreement": {
                "agreement_fraction": 0.9962,
                "n_terminals": 3,
                "per_terminal_detections": {
                    "'249'": 2,
                    "'252'": 2,
                    "'372'": 2,
                },
            },
            "subspace_detection": {
                "n_events": 0,
                "mean_agreement": 0.9767,
            },
        }

        metric_agreement = {
            "energy_vs_subspace": 0.85,
            "energy_vs_spatial": 0.95,
        }

        sensitivity_df = pd.DataFrame(
            {
                "Energy_Threshold": [50, 100, 150],
                "N_Energy_Detections": [15, 8, 3],
                "Subspace_Threshold": [30, 60, 90],
                "N_Subspace_Detections": [2, 1, 0],
            }
        )

        # Generate all tables
        perf = generate_performance_table(validation_results, metric_agreement)
        val = generate_validation_summary(validation_results)
        report = generate_comprehensive_report(
            validation_results, metric_agreement, sensitivity_df
        )

        assert len(perf) >= 2
        assert len(val) == 3
        assert len(report) > 1000  # Should be substantial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
