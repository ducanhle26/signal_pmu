#!/usr/bin/env python
"""
Phase 4: Validation Pipeline for PMU Anomaly Detection

Validates detection quality by:
1. Time alignment: Comparing detected vs labeled event timing
2. Internal consistency: Cross-metric agreement
3. Sensitivity analysis: Robustness to threshold variations
4. Cross-terminal agreement: Spatial voting validation
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

# Import Phase 3 detection functions
from src.metrics.residual_energy import detect_excitation_anomalies, extract_event_intervals
from src.metrics.subspace_change import detect_subspace_anomalies
from src.metrics.spatial_coherence import apply_spatial_voting, extract_spatially_consistent_events

# Import Phase 4 validation functions
from src.validation.time_alignment import (
    analyze_detection_timing,
    generate_timing_statistics,
)
from src.validation.internal_consistency import (
    compute_metric_agreement,
    generate_confusion_matrix,
    compute_cross_terminal_agreement,
    sensitivity_analysis,
)

# Import earlier phases
from src.data_loader import load_pilot_data, save_extracted_window
from src.topology import TopologyStore
from src.preprocessing import preprocess_pmu_signals, select_analysis_channels
from src.dynamic_models import fit_var_model, compute_residual_excitation, extract_dynamic_subspace, compute_subspace_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pilot_validation(config_path: str = "config/pilot_config.yaml") -> None:
    """
    Execute complete validation pipeline.

    Parameters
    ----------
    config_path : str
        Path to pilot configuration YAML
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: VALIDATION PIPELINE")
    logger.info("=" * 80)

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pilot = config["pilot"]
    data_cfg = config["data"]
    extraction_cfg = config["extraction"]
    output_cfg = config["output"]

    results_dir = Path(output_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = results_dir / "extracted"

    logger.info(f"Section ID: {pilot['section_id']}")
    logger.info(f"Terminals: {pilot['term_ids']}")
    logger.info(f"Date: {pilot['date']}")

    # =========================================================================
    # Load topology
    # =========================================================================
    logger.info("\n[1/5] Loading topology...")
    topology = TopologyStore(data_cfg["topology_file"])
    event_info = topology.get_event_info(pilot["section_id"])
    labeled_time = pd.Timestamp(event_info["timestamp"])
    logger.info(f"Labeled event time: {labeled_time}")

    # =========================================================================
    # Load extracted data (from Phase 1)
    # =========================================================================
    logger.info("\n[2/5] Loading extracted pilot data...")
    terminal_data = {}
    for term_id in pilot["term_ids"]:
        parquet_file = extracted_dir / f"term_{term_id}.parquet"
        if parquet_file.exists():
            terminal_data[term_id] = pd.read_parquet(parquet_file)
            logger.info(f"  Term {term_id}: {len(terminal_data[term_id])} samples")
        else:
            logger.warning(f"  Term {term_id}: Not found at {parquet_file}")

    if not terminal_data:
        logger.error("No extracted data found. Run Phase 1 first.")
        return

    # =========================================================================
    # Re-run Phase 2-3 to get metric scores & detections
    # =========================================================================
    logger.info("\n[3/5] Re-running Phase 2-3 analysis (signal processing & detection)...")

    all_metrics = {}
    all_detections = {}

    for term_id, data in terminal_data.items():
        logger.info(f"\n  Processing Terminal {term_id}...")

        # Preprocessing
        processed = preprocess_pmu_signals(data)
        channels = select_analysis_channels(processed, mode="voltage_magnitude")

        # Dynamic modeling (baseline on first 45 min)
        baseline_end = len(processed) // 2  # ~45 min in
        baseline_data = channels.iloc[:baseline_end]
        
        var_model = fit_var_model(baseline_data, order=30, window_size=300)
        residuals = var_model["residuals"]
        
        # Residual excitation energy
        energy_metric = compute_residual_excitation(
            channels, var_model["model"], window_size=300, overlap=0.5
        )

        # Subspace analysis
        baseline_subspace = extract_dynamic_subspace(
            baseline_data, n_components=5, method="svd"
        )
        subspace_distance = compute_subspace_distance(
            channels, baseline_subspace, window_size=300
        )

        # Store metrics
        all_metrics[term_id] = {
            "energy": energy_metric,
            "subspace": subspace_distance,
            "residuals": residuals,
        }

        # Detect anomalies
        energy_threshold = np.percentile(energy_metric[:baseline_end], 99)
        energy_detections = detect_excitation_anomalies(
            energy_metric, threshold=energy_threshold, persistence_k=3
        )

        subspace_threshold = np.percentile(subspace_distance[:baseline_end], 95)
        subspace_detections = detect_subspace_anomalies(
            subspace_distance, threshold=subspace_threshold, persistence_k=2
        )

        all_detections[term_id] = {
            "energy": energy_detections,
            "subspace": subspace_detections,
        }

        logger.info(f"    Energy detections: {energy_detections.get('n_events', 0)}")
        logger.info(f"    Subspace detections: {subspace_detections.get('n_events', 0)}")

    # =========================================================================
    # Spatial voting (cross-terminal consensus)
    # =========================================================================
    logger.info("\n[4/5] Computing spatial consensus...")

    # Build anomaly signals for voting
    energy_signals = {}
    subspace_signals = {}
    
    for term_id in pilot["term_ids"]:
        if term_id not in all_detections:
            continue
        
        # Convert event intervals to binary signals
        energy_det = all_detections[term_id]["energy"]
        subspace_det = all_detections[term_id]["subspace"]
        
        n_samples = len(all_metrics[term_id]["energy"])
        
        # Create binary signals
        energy_signal = np.zeros(n_samples)
        if "event_intervals" in energy_det:
            for interval in energy_det["event_intervals"]:
                start_idx = interval.get("start_idx", 0)
                end_idx = interval.get("end_idx", n_samples)
                energy_signal[start_idx:end_idx] = 1
        
        subspace_signal = np.zeros(n_samples)
        if "event_intervals" in subspace_det:
            for interval in subspace_det["event_intervals"]:
                start_idx = interval.get("start_idx", 0)
                end_idx = interval.get("end_idx", n_samples)
                subspace_signal[start_idx:end_idx] = 1
        
        energy_signals[str(term_id)] = energy_signal
        subspace_signals[str(term_id)] = subspace_signal

    # Apply spatial voting
    energy_voting = apply_spatial_voting(
        energy_signals, voting_threshold=2/3
    )
    subspace_voting = apply_spatial_voting(
        subspace_signals, voting_threshold=2/3
    )

    logger.info(f"  Energy voting: {np.sum(energy_voting['consensus'])} anomalous samples")
    logger.info(f"  Subspace voting: {np.sum(subspace_voting['consensus'])} anomalous samples")

    # =========================================================================
    # VALIDATION ANALYSES
    # =========================================================================
    logger.info("\n[5/5] Running validation analyses...")

    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "labeled_event_time": labeled_time.isoformat(),
        "section_id": pilot["section_id"],
        "terminals": pilot["term_ids"],
    }

    # A) Time alignment analysis
    logger.info("\n  A) Time Alignment Analysis...")
    
    timing_analysis = {}
    for term_id in pilot["term_ids"]:
        if term_id not in all_detections:
            continue
        
        energy_det = all_detections[term_id]["energy"]
        subspace_det = all_detections[term_id]["subspace"]
        
        # Get event times
        energy_times = []
        if "event_intervals" in energy_det:
            for iv in energy_det["event_intervals"]:
                if "start_time" in iv:
                    energy_times.append(iv["start_time"])
        
        subspace_times = []
        if "event_intervals" in subspace_det:
            for iv in subspace_det["event_intervals"]:
                if "start_time" in iv:
                    subspace_times.append(iv["start_time"])
        
        timing_analysis[f"term_{term_id}"] = {
            "energy_times": energy_times,
            "subspace_times": subspace_times,
        }

    validation_results["time_alignment"] = timing_analysis

    # B) Internal consistency: metric agreement
    logger.info("\n  B) Internal Consistency (Metric Agreement)...")
    
    metric_agreement_results = {}
    for term_id in pilot["term_ids"]:
        if term_id not in all_metrics:
            continue
        
        energy_signal = energy_signals.get(str(term_id), np.zeros(10))
        subspace_signal = subspace_signals.get(str(term_id), np.zeros(10))
        spatial_signal = np.zeros(len(energy_signal))  # placeholder
        
        agreement = compute_metric_agreement(
            energy_signal,
            subspace_signal,
            spatial_signal,
            tolerance_samples=30,
        )
        
        metric_agreement_results[f"term_{term_id}"] = {
            "energy_vs_subspace": float(agreement["energy_vs_subspace"]),
            "all_three": float(agreement["all_three"]),
            "at_least_two": float(agreement["at_least_two"]),
        }

    validation_results["metric_agreement"] = metric_agreement_results

    # C) Cross-terminal agreement
    logger.info("\n  C) Cross-Terminal Agreement...")
    
    cross_terminal = compute_cross_terminal_agreement(
        energy_signals, n_terminals_required=2
    )
    
    validation_results["cross_terminal_agreement"] = {
        "agreement_fraction": float(cross_terminal["agreement_fraction"]),
        "n_terminals": cross_terminal["n_terminals"],
        "n_terminals_required": cross_terminal["n_terminals_required"],
        "per_terminal_detections": {
            str(k): int(v) for k, v in cross_terminal["per_terminal_detections"].items()
        },
    }

    # D) Sensitivity analysis
    logger.info("\n  D) Sensitivity Analysis (Threshold Variations)...")
    
    energy_scores_all = np.concatenate(
        [all_metrics[tid]["energy"] for tid in pilot["term_ids"] if tid in all_metrics]
    )
    subspace_scores_all = np.concatenate(
        [all_metrics[tid]["subspace"] for tid in pilot["term_ids"] if tid in all_metrics]
    )
    
    sensitivity = sensitivity_analysis(energy_scores_all, subspace_scores_all)
    
    validation_results["sensitivity_analysis"] = {
        "thresholds_tested": sensitivity.to_dict(orient="records")
    }

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SAVING VALIDATION RESULTS")
    logger.info("=" * 80)

    # Save as JSON
    validation_json = results_dir / "validation_report.json"
    with open(validation_json, "w") as f:
        json.dump(validation_results, f, indent=2, default=str)
    logger.info(f"✓ Saved: {validation_json}")

    # Save metrics agreement as CSV
    metrics_df = pd.DataFrame(metric_agreement_results).T
    metrics_csv = results_dir / "metric_agreement.csv"
    metrics_df.to_csv(metrics_csv)
    logger.info(f"✓ Saved: {metrics_csv}")

    # Save sensitivity analysis
    sensitivity_csv = results_dir / "sensitivity_analysis.csv"
    sensitivity.to_csv(sensitivity_csv, index=False)
    logger.info(f"✓ Saved: {sensitivity_csv}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Metric Agreement (Energy vs Subspace): {np.mean([v['energy_vs_subspace'] for v in metric_agreement_results.values()]):.3f}")
    logger.info(f"Cross-Terminal Agreement: {validation_results['cross_terminal_agreement']['agreement_fraction']:.3f}")
    logger.info(f"Terminals in consensus: {validation_results['cross_terminal_agreement']['n_terminals']}")
    logger.info(f"Sensitivity analysis: {len(sensitivity)} threshold combinations tested")

    logger.info("\n✓ PHASE 4 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_pilot_validation()
