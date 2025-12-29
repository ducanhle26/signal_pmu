#!/usr/bin/env python
"""Run Phase 3 anomaly detection on pilot data."""

import sys
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_extracted_window
from src.preprocessing import preprocess_pmu_signals, select_analysis_channels
from src.dynamic_models import fit_var_model, compute_residual_excitation, extract_dynamic_subspace, compute_subspace_distance
from src.metrics.residual_energy import select_threshold as select_energy_threshold, detect_excitation_anomalies, extract_event_intervals, compute_false_alarm_rate
from src.metrics.subspace_change import select_subspace_threshold, detect_subspace_anomalies, compute_combined_metric
from src.metrics.spatial_coherence import apply_spatial_voting, compute_spatial_agreement_metrics, extract_spatially_consistent_events
from src.topology import load_topology, get_event_info


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("=== PMU Pilot Detection (Phase 3) ===")
    
    root = Path(__file__).parent
    extracted_dir = root / "results" / "pilot_section80" / "extracted"
    output_dir = root / "results" / "pilot_section80"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load topology for reference
    topology_file = root / "data" / "topology.csv"
    store = load_topology(topology_file)
    event_info = get_event_info(store, 80)
    labeled_event_time = event_info['event_time']
    logger.info(f"Labeled event time: {labeled_event_time}")
    
    # Load and process data
    term_ids = [249, 252, 372]
    analysis_results = {}
    
    logger.info("\nProcessing terminals and computing metrics...")
    
    for term_id in term_ids:
        logger.info(f"\n--- Terminal {term_id} ---")
        
        # Load and preprocess
        df_raw = load_extracted_window(term_id, extracted_dir)
        df_clean = preprocess_pmu_signals(df_raw, interpolate_missing=True)
        df_voltage = select_analysis_channels(df_clean, mode='voltage_magnitude')
        
        # Model and residuals
        model = fit_var_model(df_voltage, order=30)
        
        # Baseline subspace (first 45 min)
        baseline_end_idx = int(45 * 60 * 30)
        df_baseline = df_voltage.iloc[:baseline_end_idx]
        baseline_basis, baseline_sv = extract_dynamic_subspace(df_baseline, n_components=3)
        
        # Metrics
        excitation = compute_residual_excitation(df_voltage, model, window_size=300, overlap_ratio=0.5)
        subspace_dist = compute_subspace_distance(df_voltage, baseline_basis, window_size=300, overlap_ratio=0.5)
        
        # Store for cross-terminal analysis
        analysis_results[term_id] = {
            'voltage': df_voltage,
            'excitation': excitation,
            'subspace_dist': subspace_dist,
            'baseline_basis': baseline_basis,
            'baseline_sv': baseline_sv
        }
    
    # Phase 3: Anomaly Detection
    logger.info("\n=== PHASE 3: ANOMALY DETECTION ===\n")
    
    # 3.1 Residual Energy Detection
    logger.info("3.1 RESIDUAL EXCITATION ENERGY")
    logger.info("-" * 50)
    
    energy_anomalies = {}
    for term_id, results in analysis_results.items():
        excitation = results['excitation']
        
        # Threshold selection: 99th percentile of baseline
        threshold = select_energy_threshold(excitation['energy'], percentile=99.0, baseline_end_idx=400)
        
        # Detection
        anomalies = detect_excitation_anomalies(
            excitation['energy'],
            threshold,
            persistence_k=3,
            min_gap_windows=10
        )
        
        energy_anomalies[term_id] = anomalies
        
        # Events and false alarm rate
        events = extract_event_intervals(anomalies)
        far = compute_false_alarm_rate(anomalies, baseline_end=400)
        
        logger.info(f"Term {term_id}:")
        logger.info(f"  Threshold: {threshold:.2f}")
        logger.info(f"  Detected {len(events)} events")
        if events:
            logger.info(f"  First event: {events[0]['start_time']}")
            logger.info(f"  Peak energy: {events[0]['peak_energy']:.2f}")
        logger.info(f"  False alarm rate: {far['false_alarm_rate_per_hour']:.2f} per hour")
    
    # 3.2 Subspace Change Detection
    logger.info("\n3.2 SUBSPACE CHANGE DETECTION")
    logger.info("-" * 50)
    
    subspace_anomalies = {}
    for term_id, results in analysis_results.items():
        subspace_dist = results['subspace_dist']
        
        threshold = select_subspace_threshold(subspace_dist['distance'], percentile=95.0, baseline_end_idx=400)
        
        anomalies = detect_subspace_anomalies(
            subspace_dist['distance'],
            threshold,
            persistence_k=2,
            min_gap_windows=10
        )
        
        subspace_anomalies[term_id] = anomalies
        
        event_count = anomalies['event_id'].max() if anomalies['event_id'].notna().any() else 0
        logger.info(f"Term {term_id}: threshold={threshold:.4f}, {int(event_count)} events")
    
    # 3.3 Spatial Coherence Voting
    logger.info("\n3.3 SPATIAL COHERENCE VOTING")
    logger.info("-" * 50)
    
    # Spatial voting for energy
    spatial_votes_energy = apply_spatial_voting(energy_anomalies, vote_threshold=0.66)
    energy_events = extract_spatially_consistent_events(spatial_votes_energy, min_duration_windows=3)
    
    logger.info(f"Energy-based voting: {len(energy_events)} spatially consistent events")
    
    # Spatial voting for subspace
    spatial_votes_subspace = apply_spatial_voting(subspace_anomalies, vote_threshold=0.66)
    subspace_events = extract_spatially_consistent_events(spatial_votes_subspace, min_duration_windows=3)
    
    logger.info(f"Subspace-based voting: {len(subspace_events)} spatially consistent events")
    
    # Agreement metrics
    agreement_energy = compute_spatial_agreement_metrics(energy_anomalies)
    agreement_subspace = compute_spatial_agreement_metrics(subspace_anomalies)
    
    logger.info(f"\nAgreement metrics:")
    logger.info(f"  Energy: {agreement_energy['mean_pairwise_agreement']:.2%}")
    logger.info(f"  Subspace: {agreement_subspace['mean_pairwise_agreement']:.2%}")
    
    # Combined metric
    combined = compute_combined_metric(
        energy_anomalies[249],  # Use first terminal as base
        subspace_anomalies[249],
        weight_energy=0.6,
        weight_subspace=0.4
    )
    
    # Performance metrics
    logger.info("\n=== DETECTION PERFORMANCE ===")
    logger.info("-" * 50)
    
    # Time to first detection from labeled event
    label_time = pd.Timestamp(labeled_event_time)
    
    for name, events in [("Energy", energy_events), ("Subspace", subspace_events)]:
        if events:
            first_detection = min(e['start_time'] for e in events)
            time_offset = (first_detection - label_time).total_seconds() / 60.0
            logger.info(f"{name}: First detection at {first_detection}")
            logger.info(f"         {time_offset:+.1f} min from labeled event")
        else:
            logger.info(f"{name}: No detections")
    
    # Save results
    logger.info("\n=== SAVING RESULTS ===")
    
    # Save detections
    for term_id, anom_df in energy_anomalies.items():
        anom_df.to_csv(output_dir / f"energy_anomalies_term_{term_id}.csv")
    
    for term_id, anom_df in subspace_anomalies.items():
        anom_df.to_csv(output_dir / f"subspace_anomalies_term_{term_id}.csv")
    
    spatial_votes_energy.to_csv(output_dir / "spatial_votes_energy.csv")
    spatial_votes_subspace.to_csv(output_dir / "spatial_votes_subspace.csv")
    
    # Detection report
    detection_report = {
        'labeled_event_time': str(labeled_event_time),
        'energy_detection': {
            'n_events': len(energy_events),
            'first_event': str(energy_events[0]['start_time']) if energy_events else None,
            'first_event_minutes_from_label': float((energy_events[0]['start_time'] - label_time).total_seconds() / 60.0) if energy_events else None,
            'mean_agreement': float(agreement_energy['mean_pairwise_agreement'])
        },
        'subspace_detection': {
            'n_events': len(subspace_events),
            'first_event': str(subspace_events[0]['start_time']) if subspace_events else None,
            'first_event_minutes_from_label': float((subspace_events[0]['start_time'] - label_time).total_seconds() / 60.0) if subspace_events else None,
            'mean_agreement': float(agreement_subspace['mean_pairwise_agreement'])
        },
        'analysis_time': datetime.now().isoformat()
    }
    
    with open(output_dir / "detection_report.json", 'w') as f:
        json.dump(detection_report, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
