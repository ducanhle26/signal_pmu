#!/usr/bin/env python
"""Diagnostic script to investigate detection failures and tune parameters."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_extracted_window
from src.dynamic_models import compute_residual_excitation, fit_var_model
from src.metrics.residual_energy import select_threshold as select_energy_threshold
from src.preprocessing import preprocess_pmu_signals, select_analysis_channels


def diagnose_section(section_id: int, results_dir: Path):
    """Diagnose why energy detection failed for a section."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING SECTION {section_id}")
    print("=" * 60)
    
    section_dir = results_dir / f"section_{section_id}"
    extracted_dir = section_dir / "extracted"
    
    # Load detection report
    with open(section_dir / "detection_report.json") as f:
        report = json.load(f)
    
    print(f"\nLabeled time: {report['labeled_time']}")
    print(f"Terminals: {report['terminals_processed']}")
    print(f"Energy events: {report['n_energy_events']}")
    print(f"Subspace events: {report['n_subspace_events']}")
    
    # Analyze each terminal
    for term_id in report["terminals_processed"]:
        print(f"\n--- Terminal {term_id} ---")
        
        df_raw = load_extracted_window(term_id, extracted_dir)
        print(f"Samples: {len(df_raw)}")
        
        df_clean = preprocess_pmu_signals(df_raw, interpolate_missing=True)
        df_voltage = select_analysis_channels(df_clean, mode="voltage_magnitude")
        
        model = fit_var_model(df_voltage, order=30)
        
        excitation = compute_residual_excitation(
            df_voltage, model, window_size=300, overlap_ratio=0.5
        )
        
        energy = excitation["energy"].values
        
        # Analyze energy distribution
        print(f"\nEnergy statistics:")
        print(f"  Min: {energy.min():.2f}")
        print(f"  Max: {energy.max():.2f}")
        print(f"  Mean: {energy.mean():.2f}")
        print(f"  Std: {energy.std():.2f}")
        print(f"  95th percentile: {np.percentile(energy, 95):.2f}")
        print(f"  99th percentile: {np.percentile(energy, 99):.2f}")
        
        # Check different thresholds
        baseline_end = min(400, len(energy) // 2)
        baseline_energy = energy[:baseline_end]
        
        print(f"\nBaseline (first {baseline_end} windows):")
        print(f"  Mean: {baseline_energy.mean():.2f}")
        print(f"  99th percentile: {np.percentile(baseline_energy, 99):.2f}")
        
        # Find peak in event window
        labeled_time = pd.Timestamp(report["labeled_time"])
        
        # Get timestamps from excitation dataframe
        if "window_start" in excitation.columns:
            times = excitation["window_start"]
        else:
            times = excitation.index
        
        event_idx = None
        for i, t in enumerate(times):
            if pd.Timestamp(t) >= labeled_time:
                event_idx = i
                break
        
        if event_idx:
            window_start = max(0, event_idx - 50)
            window_end = min(len(energy), event_idx + 50)
            event_energy = energy[window_start:window_end]
            
            print(f"\nEvent window ({window_start}:{window_end}):")
            print(f"  Max: {event_energy.max():.2f}")
            print(f"  Mean: {event_energy.mean():.2f}")
            
            # Compare to threshold
            threshold_99 = np.percentile(baseline_energy, 99)
            threshold_95 = np.percentile(baseline_energy, 95)
            threshold_90 = np.percentile(baseline_energy, 90)
            
            print(f"\nThreshold comparison:")
            print(f"  99th threshold: {threshold_99:.2f} -> Peak above? {event_energy.max() > threshold_99}")
            print(f"  95th threshold: {threshold_95:.2f} -> Peak above? {event_energy.max() > threshold_95}")
            print(f"  90th threshold: {threshold_90:.2f} -> Peak above? {event_energy.max() > threshold_90}")
            
            # Peak to baseline ratio
            ratio = event_energy.max() / baseline_energy.mean()
            print(f"\nPeak/Baseline ratio: {ratio:.2f}x")


def main():
    root = Path(__file__).parent
    results_dir = root / "results" / "multi_event"
    
    # Diagnose Section 1476 (energy detection failed)
    diagnose_section(1476, results_dir)
    
    # Also check 1035 (successful 138kV) for comparison
    print("\n\n" + "="*60)
    print("COMPARISON: SUCCESSFUL 138kV EVENT (Section 1035)")
    print("="*60)
    diagnose_section(1035, results_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
