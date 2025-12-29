#!/usr/bin/env python
"""Phase 6: Multi-event analysis pipeline.

Processes all events from topology.csv and aggregates results
for latency distribution analysis and methodology assessment.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.aggregation import (
    EventResult,
    compute_latency_stats,
    compute_voltage_breakdown,
    generate_summary_report,
    results_to_dataframe,
)
from src.data_loader import extract_time_window, load_extracted_window, save_extracted_window, PMUFileSpec
from src.dynamic_models import (
    compute_residual_excitation,
    compute_subspace_distance,
    extract_dynamic_subspace,
    fit_var_model,
)
from src.metrics.residual_energy import (
    compute_false_alarm_rate,
    detect_excitation_anomalies,
    extract_event_intervals,
    select_threshold as select_energy_threshold,
)
from src.metrics.spatial_coherence import (
    apply_spatial_voting,
    compute_spatial_agreement_metrics,
    extract_spatially_consistent_events,
)
from src.metrics.subspace_change import (
    detect_subspace_anomalies,
    select_subspace_threshold,
)
from src.preprocessing import preprocess_pmu_signals, select_analysis_channels
from src.topology import get_event_info, load_topology


def get_unique_sections(store) -> list[int]:
    """Get all unique SectionIDs from topology."""
    return store.sections["SectionID"].unique().tolist()


def determine_time_window(event_time: pd.Timestamp) -> tuple[str, int, int]:
    """
    Determine extraction time window around event.

    Returns:
        (date_str, start_hour, end_hour)
    """
    event_hour = event_time.hour

    if event_hour >= 22:
        start_hour = event_hour - 1
        end_hour = 0
        date_str = event_time.strftime("%Y-%m-%d")
    elif event_hour <= 2:
        start_hour = event_hour - 1 if event_hour > 0 else 23
        end_hour = event_hour + 2
        date_str = (event_time - timedelta(days=1)).strftime("%Y-%m-%d") if event_hour == 0 else event_time.strftime("%Y-%m-%d")
    else:
        start_hour = max(0, event_hour - 1)
        end_hour = min(23, event_hour + 1)
        date_str = event_time.strftime("%Y-%m-%d")

    return date_str, start_hour, end_hour


def check_data_availability(term_ids: list[int], date: str, raw_pmu_dir: Path) -> list[int]:
    """Check which terminals have data files available."""
    available = []
    for term_id in term_ids:
        filename = f"{term_id} {date}.csv"
        if (raw_pmu_dir / filename).exists():
            available.append(term_id)
    return available


def process_single_event(
    section_id: int,
    store,
    raw_pmu_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    use_cache: bool = True,
) -> EventResult:
    """
    Process a single event through the full detection pipeline.

    Returns:
        EventResult with detection outcomes
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing SectionID {section_id}")
    logger.info("=" * 60)

    try:
        event_info = get_event_info(store, section_id)
        term_ids = event_info["term_ids"]
        labeled_time = pd.Timestamp(event_info["event_time"])
        voltage_kv = event_info["voltage_kv"]
        cause = event_info["cause"]

        logger.info(f"  Event: {event_info['event_location']}")
        logger.info(f"  Time: {labeled_time}")
        logger.info(f"  Voltage: {voltage_kv/1000:.0f}kV")
        logger.info(f"  Terminals: {term_ids}")

        date_str, start_hour, end_hour = determine_time_window(labeled_time)
        available_terms = check_data_availability(term_ids, date_str, raw_pmu_dir)

        if len(available_terms) < 1:
            logger.warning(f"  No terminals available")
            return EventResult(
                section_id=section_id,
                term_ids=term_ids,
                voltage_kv=voltage_kv,
                cause=cause,
                labeled_time=labeled_time,
                n_energy_events=0,
                n_subspace_events=0,
                first_detection_time=None,
                latency_minutes=None,
                spatial_agreement=0.0,
                false_alarm_rate=0.0,
                processing_status="no_data",
                error_message=f"No terminals available",
            )
        
        # Flag for single-terminal mode (skip spatial voting)
        single_terminal_mode = len(available_terms) == 1
        if single_terminal_mode:
            logger.info(f"  Single-terminal mode: spatial voting disabled")

        logger.info(f"  Available terminals: {available_terms}")

        section_output_dir = output_dir / f"section_{section_id}"
        extracted_dir = section_output_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        analysis_results = {}

        for term_id in available_terms:
            logger.info(f"  Processing terminal {term_id}...")

            cache_path = extracted_dir / f"term_{term_id}.parquet"

            if use_cache and cache_path.exists():
                df_raw = load_extracted_window(term_id, extracted_dir)
                logger.info(f"    Loaded from cache: {len(df_raw)} rows")
            else:
                pmu_file = raw_pmu_dir / f"{term_id} {date_str}.csv"

                window_start = labeled_time - pd.Timedelta(hours=1)
                window_end = labeled_time + pd.Timedelta(hours=1)

                spec = PMUFileSpec(path=pmu_file, term_id=term_id)
                result = extract_time_window(spec, window_start, window_end)
                df_raw = result.df

                if df_raw.empty:
                    logger.warning(f"    No data in time window for term {term_id}")
                    continue

                save_extracted_window(result, extracted_dir)
                logger.info(f"    Extracted: {len(df_raw)} rows")

            if len(df_raw) < 1000:
                logger.warning(f"    Insufficient samples: {len(df_raw)}")
                continue

            df_clean = preprocess_pmu_signals(df_raw, interpolate_missing=True)
            df_voltage = select_analysis_channels(df_clean, mode="voltage_magnitude")

            if df_voltage.empty or len(df_voltage.columns) == 0:
                logger.warning(f"    No valid voltage channels for term {term_id}")
                continue

            model = fit_var_model(df_voltage, order=30)

            baseline_end_idx = min(int(30 * 60 * 30), len(df_voltage) // 2)
            df_baseline = df_voltage.iloc[:baseline_end_idx]

            if len(df_baseline) < 100:
                logger.warning(f"    Insufficient baseline samples for term {term_id}")
                continue

            baseline_basis, baseline_sv = extract_dynamic_subspace(df_baseline, n_components=3)

            excitation = compute_residual_excitation(df_voltage, model, window_size=300, overlap_ratio=0.5)
            subspace_dist = compute_subspace_distance(df_voltage, baseline_basis, window_size=300, overlap_ratio=0.5)

            analysis_results[term_id] = {
                "voltage": df_voltage,
                "excitation": excitation,
                "subspace_dist": subspace_dist,
            }

        if len(analysis_results) < 1:
            return EventResult(
                section_id=section_id,
                term_ids=term_ids,
                voltage_kv=voltage_kv,
                cause=cause,
                labeled_time=labeled_time,
                n_energy_events=0,
                n_subspace_events=0,
                first_detection_time=None,
                latency_minutes=None,
                spatial_agreement=0.0,
                false_alarm_rate=0.0,
                processing_status="processing_failed",
                error_message=f"No terminals processed successfully",
            )

        logger.info("  Running anomaly detection...")

        energy_anomalies = {}
        total_far = 0.0

        for term_id, results in analysis_results.items():
            excitation = results["excitation"]

            baseline_end = min(400, len(excitation) // 2)
            baseline_energy = excitation["energy"].iloc[:baseline_end]
            
            # Adaptive threshold: use 95th percentile for noisy baselines
            # Check if baseline has high variance (noisy signal)
            baseline_cv = baseline_energy.std() / baseline_energy.mean() if baseline_energy.mean() > 0 else 0
            if baseline_cv > 1.0:  # High variance baseline
                percentile = 95.0
                logger.info(f"    Term {term_id}: High noise (CV={baseline_cv:.2f}), using 95th percentile")
            else:
                percentile = 99.0
            
            threshold = select_energy_threshold(excitation["energy"], percentile=percentile, baseline_end_idx=baseline_end)

            anomalies = detect_excitation_anomalies(
                excitation["energy"],
                threshold,
                persistence_k=3,
                min_gap_windows=10,
            )

            energy_anomalies[term_id] = anomalies

            far = compute_false_alarm_rate(anomalies, baseline_end=baseline_end)
            total_far += far["false_alarm_rate_per_hour"]

        avg_far = total_far / len(energy_anomalies) if energy_anomalies else 0.0

        subspace_anomalies = {}
        for term_id, results in analysis_results.items():
            subspace_dist = results["subspace_dist"]

            baseline_end = min(400, len(subspace_dist) // 2)
            threshold = select_subspace_threshold(subspace_dist["distance"], percentile=95.0, baseline_end_idx=baseline_end)

            anomalies = detect_subspace_anomalies(
                subspace_dist["distance"],
                threshold,
                persistence_k=2,
                min_gap_windows=10,
            )

            subspace_anomalies[term_id] = anomalies

        # Handle single-terminal vs multi-terminal detection
        # Recompute single_terminal_mode based on actual processed terminals
        single_terminal_mode = len(analysis_results) == 1
        
        if single_terminal_mode:
            # Single terminal: use direct detection without spatial voting
            term_id = list(analysis_results.keys())[0]
            energy_flags = energy_anomalies[term_id]
            subspace_flags = subspace_anomalies[term_id]
            
            # Get timestamps from the processed terminal's results
            term_results = analysis_results[term_id]
            excitation_df = term_results["excitation"]
            subspace_df = term_results["subspace_dist"]
            
            # Extract events from energy anomalies
            energy_events = extract_event_intervals(
                energy_flags, 
                excitation_df["window_start"] if "window_start" in excitation_df.columns else excitation_df.index,
                min_gap_windows=10
            )
            
            # Extract events from subspace anomalies  
            subspace_events = extract_event_intervals(
                subspace_flags,
                subspace_df["window_start"] if "window_start" in subspace_df.columns else subspace_df.index,
                min_gap_windows=10
            )
            
            spatial_agreement = 1.0  # Single terminal = 100% "agreement"
            logger.info(f"    Single-terminal detection: {len(energy_events)} energy, {len(subspace_events)} subspace events")
        else:
            # Multi-terminal: use spatial voting
            vote_threshold = 0.5 if len(analysis_results) == 2 else 0.66
            spatial_votes = apply_spatial_voting(energy_anomalies, vote_threshold=vote_threshold)
            energy_events = extract_spatially_consistent_events(spatial_votes, min_duration_windows=3)

            subspace_votes = apply_spatial_voting(subspace_anomalies, vote_threshold=vote_threshold)
            subspace_events = extract_spatially_consistent_events(subspace_votes, min_duration_windows=3)

            agreement = compute_spatial_agreement_metrics(energy_anomalies)
            spatial_agreement = agreement["mean_pairwise_agreement"]

        # Combined detection: use energy OR subspace events
        first_detection = None
        latency_min = None

        # Prefer energy events, fall back to subspace
        if energy_events:
            first_detection = min(e["start_time"] for e in energy_events)
            latency_min = (first_detection - labeled_time).total_seconds() / 60.0
        elif subspace_events:
            # Fallback to subspace detection
            first_detection = min(e["start_time"] for e in subspace_events)
            latency_min = (first_detection - labeled_time).total_seconds() / 60.0
            logger.info(f"    Using subspace detection (energy failed)")

        logger.info(f"  Results:")
        logger.info(f"    Energy events: {len(energy_events)}")
        logger.info(f"    Subspace events: {len(subspace_events)}")
        logger.info(f"    Spatial agreement: {spatial_agreement:.1%}")
        logger.info(f"    Latency: {latency_min:+.1f} min" if latency_min else "    Latency: N/A")

        detection_report = {
            "section_id": section_id,
            "labeled_time": str(labeled_time),
            "first_detection": str(first_detection) if first_detection else None,
            "latency_minutes": latency_min,
            "n_energy_events": len(energy_events),
            "n_subspace_events": len(subspace_events),
            "spatial_agreement": spatial_agreement,
            "false_alarm_rate": avg_far,
            "terminals_processed": list(analysis_results.keys()),
        }

        with open(section_output_dir / "detection_report.json", "w") as f:
            json.dump(detection_report, f, indent=2, default=str)

        return EventResult(
            section_id=section_id,
            term_ids=term_ids,
            voltage_kv=voltage_kv,
            cause=cause,
            labeled_time=labeled_time,
            n_energy_events=len(energy_events),
            n_subspace_events=len(subspace_events),
            first_detection_time=first_detection,
            latency_minutes=latency_min,
            spatial_agreement=spatial_agreement,
            false_alarm_rate=avg_far,
            processing_status="success",
        )

    except Exception as e:
        logger.error(f"  Error processing section {section_id}: {e}")
        import traceback

        traceback.print_exc()

        return EventResult(
            section_id=section_id,
            term_ids=[],
            voltage_kv=0,
            cause="",
            labeled_time=pd.Timestamp.now(tz="UTC"),
            n_energy_events=0,
            n_subspace_events=0,
            first_detection_time=None,
            latency_minutes=None,
            spatial_agreement=0.0,
            false_alarm_rate=0.0,
            processing_status="error",
            error_message=str(e),
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("PHASE 6: MULTI-EVENT ANALYSIS PIPELINE")
    logger.info("=" * 70)

    root = Path(__file__).parent
    raw_pmu_dir = root / "data" / "raw_pmu"
    topology_file = root / "data" / "topology.csv"
    output_dir = root / "results" / "multi_event"
    output_dir.mkdir(parents=True, exist_ok=True)

    store = load_topology(topology_file)
    section_ids = get_unique_sections(store)

    logger.info(f"\nFound {len(section_ids)} unique events in topology")
    logger.info(f"Section IDs: {section_ids}")

    results: list[EventResult] = []

    for i, section_id in enumerate(section_ids):
        logger.info(f"\n[{i+1}/{len(section_ids)}] Processing Section {section_id}")

        result = process_single_event(
            section_id=section_id,
            store=store,
            raw_pmu_dir=raw_pmu_dir,
            output_dir=output_dir,
            logger=logger,
            use_cache=True,
        )

        results.append(result)

    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATING RESULTS")
    logger.info("=" * 70)

    report = generate_summary_report(results, output_dir / "summary_report.txt")
    print("\n" + report)

    results_df = results_to_dataframe(results)
    results_df.to_csv(output_dir / "all_events_results.csv", index=False)

    latency_stats = compute_latency_stats(results)
    voltage_breakdown = compute_voltage_breakdown(results)

    summary = {
        "analysis_time": datetime.now().isoformat(),
        "n_events_total": len(results),
        "n_events_detected": latency_stats["n_detected"],
        "detection_rate": latency_stats["detection_rate"],
        "latency_stats": latency_stats,
        "voltage_breakdown": voltage_breakdown.to_dict("records") if not voltage_breakdown.empty else [],
    }

    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("\nPhase 6 complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
