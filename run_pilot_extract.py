#!/usr/bin/env python
"""Pilot data extraction for SectionID 80."""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import PMUFileSpec, load_pilot_data, save_extracted_window
from src.topology import load_topology, get_event_info


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    # Load config
    config_path = Path(__file__).parent / "config" / "pilot_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info("=== PMU Pilot Extraction ===")
    logger.info(f"Config: {config}")
    
    # Prepare paths
    root = Path(__file__).parent
    raw_pmu_dir = root / config['data']['raw_pmu_dir']
    topology_file = root / config['data']['topology_file']
    results_dir = root / config['output']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load topology and validate SectionID
    logger.info(f"Loading topology from {topology_file}")
    store = load_topology(topology_file)
    
    section_id = config['pilot']['section_id']
    event_info = get_event_info(store, section_id)
    logger.info(f"SectionID {section_id}:")
    logger.info(f"  Terminals: {event_info['term_ids']}")
    logger.info(f"  Event time: {event_info['event_time']}")
    logger.info(f"  Voltage: {event_info['voltage_kv']} kV")
    logger.info(f"  Location: {event_info['event_location']}")
    
    # Build time window
    date_str = config['pilot']['date']
    start_hour = config['pilot']['start_hour']
    end_hour = config['pilot']['end_hour']
    
    start_dt = pd.Timestamp(f"{date_str} {start_hour:02d}:00:00", tz="UTC")
    
    if end_hour == 0:
        # Next day
        start_date = datetime.strptime(date_str, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)
        end_dt = pd.Timestamp(f"{end_date.strftime('%Y-%m-%d')} 00:00:00", tz="UTC")
    else:
        end_dt = pd.Timestamp(f"{date_str} {end_hour:02d}:00:00", tz="UTC")
    
    logger.info(f"Time window: {start_dt} to {end_dt}")
    
    # Build PMUFileSpec for each terminal
    term_ids = config['pilot']['term_ids']
    file_specs = []
    
    for term_id in term_ids:
        pmu_file = raw_pmu_dir / f"{term_id} {date_str}.csv"
        
        if not pmu_file.exists():
            logger.error(f"PMU file not found: {pmu_file}")
            continue
        
        spec = PMUFileSpec(
            path=pmu_file,
            term_id=term_id,
            timestamp_col="UTC",
            tz="UTC",
            sample_rate_hz=30.0,
            quality_cols=("STAT",)
        )
        file_specs.append(spec)
    
    if not file_specs:
        logger.error("No valid PMU files found!")
        return 1
    
    # Extract time windows
    logger.info(f"Extracting {len(file_specs)} terminals...")
    results = load_pilot_data(file_specs, start_dt, end_dt, require_exact_rate=False)
    
    # Save extracted windows and generate quality report
    extracted_dir = results_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    quality_report = {
        'section_id': section_id,
        'date': date_str,
        'time_window': {
            'start': str(start_dt),
            'end': str(end_dt)
        },
        'terminals': {}
    }
    
    for term_id, result in results.items():
        if config['output']['save_extracted']:
            save_extracted_window(result, extracted_dir)
        
        quality_report['terminals'][term_id] = {
            'n_rows': result.quality.n_rows,
            'start_ts': str(result.quality.start_ts),
            'end_ts': str(result.quality.end_ts),
            'expected_samples': result.quality.expected_samples,
            'actual_samples': result.quality.actual_samples,
            'missing_count': result.quality.missing_count,
            'duplicate_timestamps': result.quality.duplicate_timestamps,
            'max_gap_seconds': result.quality.max_gap_seconds,
            'rate_estimate_hz': result.quality.rate_estimate_hz,
            'is_healthy': result.quality.is_healthy,
            'stat_summary': result.quality.stat_summary
        }
    
    # Save quality report
    report_path = results_dir / "quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    logger.info(f"Quality report saved to {report_path}")
    
    # Summary
    logger.info("\n=== EXTRACTION SUMMARY ===")
    for term_id, report in quality_report['terminals'].items():
        logger.info(f"Term {term_id}: {report['n_rows']} rows, "
                   f"rate={report['rate_estimate_hz']:.2f} Hz, "
                   f"healthy={report['is_healthy']}")
    
    logger.info("\nPilot extraction complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
