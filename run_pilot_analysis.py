#!/usr/bin/env python
"""Run full Phase 1-2 analysis on pilot data."""

import sys
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_extracted_window
from src.preprocessing import preprocess_pmu_signals, select_analysis_channels, compute_signal_statistics
from src.dynamic_models import fit_var_model, compute_residual_excitation, extract_dynamic_subspace, compute_subspace_distance


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    logger.info("=== PMU Pilot Analysis (Phase 2) ===")
    
    root = Path(__file__).parent
    extracted_dir = root / "results" / "pilot_section80" / "extracted"
    output_dir = root / "results" / "pilot_section80"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load extracted data
    term_ids = [249, 252, 372]
    data = {}
    
    logger.info("Loading extracted windows...")
    for term_id in term_ids:
        df = load_extracted_window(term_id, extracted_dir)
        data[term_id] = df
        logger.info(f"  Term {term_id}: {len(df)} samples")
    
    # Preprocessing and channel selection
    logger.info("\nPreprocessing signals...")
    analysis_results = {}
    
    for term_id, df_raw in data.items():
        logger.info(f"\n--- Terminal {term_id} ---")
        
        # Preprocess
        df_clean = preprocess_pmu_signals(df_raw, interpolate_missing=True)
        
        # Select voltage channels
        df_voltage = select_analysis_channels(df_clean, mode='voltage_magnitude')
        
        # Compute statistics
        stats = compute_signal_statistics(df_voltage)
        logger.info(f"Signal stats:")
        for ch, mean_val in stats['mean'].items():
            logger.info(f"  {ch}: mean={mean_val:.1f}, std={stats['std'][ch]:.1f}")
        
        # Dynamic modeling
        logger.info("Fitting VAR model...")
        model = fit_var_model(df_voltage, order=30)
        
        # Baseline subspace (first 45 min: 22:00-22:45)
        baseline_end_idx = int(45 * 60 * 30)  # 45 minutes @ 30 Hz
        df_baseline = df_voltage.iloc[:baseline_end_idx]
        baseline_basis, baseline_sv = extract_dynamic_subspace(df_baseline, n_components=3)
        
        logger.info(f"Baseline subspace: {baseline_basis.shape}")
        logger.info(f"Baseline singular values: {baseline_sv}")
        
        # Compute metrics
        logger.info("Computing excitation energy...")
        excitation = compute_residual_excitation(df_voltage, model, window_size=300, overlap_ratio=0.5)
        
        logger.info("Computing subspace distance...")
        subspace_dist = compute_subspace_distance(df_voltage, baseline_basis, window_size=300, overlap_ratio=0.5)
        
        # Store results
        analysis_results[term_id] = {
            'df_voltage': df_voltage,
            'model': model,
            'baseline_basis': baseline_basis,
            'baseline_sv': baseline_sv,
            'excitation': excitation,
            'subspace_distance': subspace_dist,
            'stats': stats
        }
    
    # Cross-terminal analysis
    logger.info("\n=== CROSS-TERMINAL ANALYSIS ===")
    
    # Combine excitation energies
    combined_excitation = pd.DataFrame(index=pd.DatetimeIndex([]))
    for term_id, result in analysis_results.items():
        col = f'term_{term_id}'
        combined_excitation[col] = result['excitation']['energy']
    
    # Compute mean energy
    combined_excitation['mean_energy'] = combined_excitation.mean(axis=1)
    combined_excitation['std_energy'] = combined_excitation.iloc[:, :-1].std(axis=1)
    
    logger.info(f"Mean excitation energy: {combined_excitation['mean_energy'].mean():.4f}")
    logger.info(f"Peak excitation energy: {combined_excitation['mean_energy'].max():.4f} at {combined_excitation['mean_energy'].idxmax()}")
    
    # Find event window (around expected 22:57)
    event_time = pd.Timestamp('2020-08-31 22:57:00', tz='UTC')
    window_start = event_time - pd.Timedelta(minutes=5)
    window_end = event_time + pd.Timedelta(minutes=10)
    
    event_window = combined_excitation.loc[window_start:window_end]
    if len(event_window) > 0:
        logger.info(f"\nEvent window [{window_start} to {window_end}]:")
        logger.info(f"  Mean energy in window: {event_window['mean_energy'].mean():.4f}")
        logger.info(f"  Peak in window: {event_window['mean_energy'].max():.4f} at {event_window['mean_energy'].idxmax()}")
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save combined metrics
    combined_excitation.to_csv(output_dir / "combined_excitation.csv")
    
    # Save summary
    summary = {
        'analysis_time': datetime.now().isoformat(),
        'terminals': term_ids,
        'peak_excitation': {
            'timestamp': str(combined_excitation['mean_energy'].idxmax()),
            'energy': float(combined_excitation['mean_energy'].max())
        },
        'event_time_label': '2020-08-31 22:57:00 (unreliable)',
        'baseline_period': '22:00-22:45'
    }
    
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("Pilot analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
