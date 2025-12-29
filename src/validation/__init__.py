"""Validation module for anomaly detection metrics."""

from .time_alignment import (
    compute_time_offsets,
    analyze_detection_timing,
    generate_timing_statistics,
)
from .internal_consistency import (
    compute_metric_agreement,
    generate_confusion_matrix,
    compute_cross_terminal_agreement,
    sensitivity_analysis,
)

__all__ = [
    "compute_time_offsets",
    "analyze_detection_timing",
    "generate_timing_statistics",
    "compute_metric_agreement",
    "generate_confusion_matrix",
    "compute_cross_terminal_agreement",
    "sensitivity_analysis",
]
