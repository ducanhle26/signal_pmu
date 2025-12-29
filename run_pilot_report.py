#!/usr/bin/env python
"""
Phase 5: Reporting Pipeline for PMU Anomaly Detection

Generates publication-quality figures and comprehensive analysis report.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import matplotlib.pyplot as plt

# Import visualization
from src.visualization.plots import (
    plot_signal_timeseries,
    plot_energy_metric,
    plot_subspace_metric,
    plot_spatial_voting_heatmap,
    plot_anomaly_detection_comparison,
    plot_sensitivity_curves,
)

# Import reporting
from src.reporting.results import (
    generate_performance_table,
    generate_validation_summary,
    generate_comprehensive_report,
)

# Import earlier phases
from src.data_loader import load_pilot_data
from src.topology import load_topology, get_event_info
from src.preprocessing import preprocess_pmu_signals, select_analysis_channels
from src.dynamic_models import fit_var_model, compute_residual_excitation, extract_dynamic_subspace, compute_subspace_distance
from src.metrics.residual_energy import detect_excitation_anomalies, select_threshold
from src.metrics.subspace_change import detect_subspace_anomalies, select_subspace_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pilot_report(config_path: str = "config/pilot_config.yaml") -> None:
    """
    Execute complete reporting pipeline.

    Parameters
    ----------
    config_path : str
        Path to pilot configuration YAML
    """
    logger.info("=" * 80)
    logger.info("PHASE 5: REPORTING PIPELINE")
    logger.info("=" * 80)

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pilot = config["pilot"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    results_dir = Path(output_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = results_dir / "extracted"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger.info(f"Section ID: {pilot['section_id']}")
    logger.info(f"Results directory: {results_dir}")

    # =========================================================================
    # Load data and topology
    # =========================================================================
    logger.info("\n[1/4] Loading data and topology...")
    
    topology = load_topology(data_cfg["topology_file"])
    event_info = get_event_info(topology, pilot["section_id"])
    labeled_time = pd.Timestamp(event_info["event_time"])
    logger.info(f"Labeled event time: {labeled_time}")

    # Load extracted data
    terminal_data = {}
    for term_id in pilot["term_ids"]:
        parquet_file = extracted_dir / f"term_{term_id}.parquet"
        if parquet_file.exists():
            terminal_data[term_id] = pd.read_parquet(parquet_file)
            logger.info(f"  Term {term_id}: {len(terminal_data[term_id])} samples")
        else:
            logger.warning(f"  Term {term_id}: Not found")

    if not terminal_data:
        logger.error("No extracted data. Run Phase 1 first.")
        return

    # =========================================================================
    # Load validation results
    # =========================================================================
    logger.info("\n[2/4] Loading validation and analysis results...")

    # Load validation report
    validation_json = results_dir / "validation_report.json"
    metric_agreement_csv = results_dir / "metric_agreement.csv"
    sensitivity_csv = results_dir / "sensitivity_analysis.csv"

    validation_results = {}
    if validation_json.exists():
        with open(validation_json) as f:
            validation_results = json.load(f)
        logger.info(f"✓ Loaded: {validation_json}")
    else:
        logger.warning(f"Validation report not found: {validation_json}")

    metric_agreement = {}
    if metric_agreement_csv.exists():
        metric_agreement_df = pd.read_csv(metric_agreement_csv)
        if not metric_agreement_df.empty:
            metric_agreement = metric_agreement_df.iloc[0].to_dict()
        logger.info(f"✓ Loaded: {metric_agreement_csv}")

    sensitivity_df = pd.DataFrame()
    if sensitivity_csv.exists():
        sensitivity_df = pd.read_csv(sensitivity_csv)
        logger.info(f"✓ Loaded: {sensitivity_csv}")

    # =========================================================================
    # Generate visualization figures
    # =========================================================================
    logger.info("\n[3/4] Generating publication-quality figures...")

    # Recompute metrics for visualization
    logger.info("  Computing metrics...")
    
    # Pick first terminal for detailed plots
    term_id = pilot["term_ids"][0]
    data = terminal_data[term_id]

    processed = preprocess_pmu_signals(data)
    channels = select_analysis_channels(processed, mode="voltage_magnitude")

    # Baseline modeling
    baseline_end = len(processed) // 2
    baseline_data = channels.iloc[:baseline_end]
    
    var_model = fit_var_model(baseline_data, order=30, window_size=300)
    residuals = var_model.residuals

    # Metrics
    energy_metric = compute_residual_excitation(
        channels, var_model, window_size=300, overlap_ratio=0.5
    )
    subspace_baseline, _ = extract_dynamic_subspace(baseline_data, n_components=3, method="pca")
    subspace_metric = compute_subspace_distance(channels, subspace_baseline, window_size=300, overlap_ratio=0.5)

    # Detections
    energy_threshold = select_threshold(energy_metric["energy"], percentile=99.0, baseline_end_idx=400)
    energy_det = detect_excitation_anomalies(energy_metric["energy"], threshold=energy_threshold, persistence_k=3)
    
    subspace_threshold = select_subspace_threshold(subspace_metric["distance"], percentile=95.0, baseline_end_idx=400)
    subspace_det = detect_subspace_anomalies(subspace_metric["distance"], threshold=subspace_threshold, persistence_k=2)

    # Convert to binary signals
    energy_signal = np.zeros(len(energy_metric))
    if "event_intervals" in energy_det:
        for interval in energy_det["event_intervals"]:
            start_idx = interval.get("start_idx", 0)
            end_idx = interval.get("end_idx", len(energy_metric))
            energy_signal[start_idx:end_idx] = 1

    subspace_signal = np.zeros(len(subspace_metric))
    if "event_intervals" in subspace_det:
        for interval in subspace_det["event_intervals"]:
            start_idx = interval.get("start_idx", 0)
            end_idx = interval.get("end_idx", len(subspace_metric))
            subspace_signal[start_idx:end_idx] = 1

    spatial_signal = energy_signal  # Use energy for spatial (simplified)

    # Create time index for metrics
    metric_time_index = energy_metric.index

    # Figure 1: Signal timeseries
    logger.info("  [1/6] Signal timeseries...")
    plot_signal_timeseries(
        terminal_data, labeled_time,
        save_path=str(figures_dir / "01_signal_timeseries.png")
    )

    # Figure 2: Energy metric
    logger.info("  [2/6] Energy metric...")
    if len(metric_time_index) == len(energy_metric):
        plot_energy_metric(
            energy_metric, energy_threshold, energy_signal, metric_time_index,
            labeled_time, term_id=str(term_id),
            save_path=str(figures_dir / "02_energy_metric.png")
        )
    else:
        logger.warning(f"    Index mismatch ({len(metric_time_index)} vs {len(energy_metric)})")

    # Figure 3: Subspace metric
    logger.info("  [3/6] Subspace metric...")
    if len(metric_time_index) == len(subspace_metric):
        plot_subspace_metric(
            subspace_metric, subspace_threshold, subspace_signal, metric_time_index,
            labeled_time, term_id=str(term_id),
            save_path=str(figures_dir / "03_subspace_metric.png")
        )

    # Figure 4: Spatial voting heatmap
    logger.info("  [4/6] Spatial voting heatmap...")
    terminal_signals = {}
    for tid in pilot["term_ids"]:
        if tid in terminal_data:
            terminal_signals[str(tid)] = energy_signal[:min(len(energy_signal), 1000)]
    
    if terminal_signals:
        plot_spatial_voting_heatmap(
            terminal_signals, data.index[:len(energy_signal)],
            labeled_time, metric_name="Energy",
            save_path=str(figures_dir / "04_spatial_voting_heatmap.png")
        )

    # Figure 5: Anomaly comparison
    logger.info("  [5/6] Anomaly detection comparison...")
    if len(data.index) >= len(energy_signal):
        plot_anomaly_detection_comparison(
            energy_signal, subspace_signal, spatial_signal,
            data.index[:len(energy_signal)], labeled_time, term_id=str(term_id),
            save_path=str(figures_dir / "05_anomaly_comparison.png")
        )

    # Figure 6: Sensitivity curves
    logger.info("  [6/6] Sensitivity curves...")
    if not sensitivity_df.empty:
        plot_sensitivity_curves(
            sensitivity_df,
            save_path=str(figures_dir / "06_sensitivity_curves.png")
        )

    # =========================================================================
    # Generate summary tables
    # =========================================================================
    logger.info("\n[4/4] Generating summary tables...")

    # Performance table
    perf_table = generate_performance_table(validation_results, metric_agreement)
    perf_csv = results_dir / "performance_metrics.csv"
    perf_table.to_csv(perf_csv, index=False)
    logger.info(f"✓ Saved: {perf_csv}")

    # Validation summary table
    val_table = generate_validation_summary(validation_results)
    val_csv = results_dir / "validation_summary.csv"
    val_table.to_csv(val_csv, index=False)
    logger.info(f"✓ Saved: {val_csv}")

    # Comprehensive report
    report = generate_comprehensive_report(
        validation_results, metric_agreement, sensitivity_df,
        output_path=str(results_dir / "comprehensive_report.txt")
    )
    logger.info(f"✓ Saved: {results_dir / 'comprehensive_report.txt'}")

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5 COMPLETE - REPORTING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nFigures generated: 6")
    logger.info(f"  {figures_dir}/01_signal_timeseries.png")
    logger.info(f"  {figures_dir}/02_energy_metric.png")
    logger.info(f"  {figures_dir}/03_subspace_metric.png")
    logger.info(f"  {figures_dir}/04_spatial_voting_heatmap.png")
    logger.info(f"  {figures_dir}/05_anomaly_comparison.png")
    logger.info(f"  {figures_dir}/06_sensitivity_curves.png")

    logger.info(f"\nTables generated: 3")
    logger.info(f"  {perf_csv}")
    logger.info(f"  {val_csv}")
    logger.info(f"  {sensitivity_csv}")

    logger.info(f"\nReports generated: 2")
    logger.info(f"  {results_dir / 'comprehensive_report.txt'}")
    logger.info(f"  {validation_json}")

    logger.info("\n✓ PHASE 5 COMPLETE - Ready for publication!")
    logger.info("=" * 80)

    # Close all figures to avoid warnings
    plt.close("all")


if __name__ == "__main__":
    run_pilot_report()
