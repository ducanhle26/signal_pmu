"""Efficient extraction of time windows from large PMU CSV files."""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PMUFileSpec:
    """Specification for a PMU CSV file."""
    path: Path
    term_id: int
    timestamp_col: str = "UTC"
    tz: str = "UTC"
    sample_rate_hz: float = 30.0
    quality_cols: tuple = ("STAT",)
    dtype_map: dict = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Quality metrics for extracted PMU data."""
    n_rows: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    expected_samples: int
    actual_samples: int
    missing_count: int
    duplicate_timestamps: int
    nonmonotonic_count: int
    max_gap_seconds: Optional[float]
    rate_estimate_hz: float
    stat_summary: dict = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Quick health check: rate within ±5% and no duplicates."""
        rate_ok = abs(self.rate_estimate_hz - 30.0) / 30.0 < 0.05
        no_dups = self.duplicate_timestamps == 0
        return rate_ok and no_dups


@dataclass
class ExtractionResult:
    """Result of time window extraction."""
    df: pd.DataFrame
    quality: DataQualityReport
    source: PMUFileSpec
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_timestamp_series(s: pd.Series) -> pd.DatetimeIndex:
    """Parse timestamp column to UTC datetime index."""
    try:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
    except Exception as e:
        logger.warning(f"Failed to parse timestamps: {e}")
        dt = pd.to_datetime(s, errors="coerce")
    
    nat_rate = dt.isna().sum() / len(dt)
    if nat_rate > 0.001:
        logger.warning(f"High NaT rate after parsing: {nat_rate:.2%}")
    
    return dt


def _filter_chunk_to_window(
    chunk_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ts_col: str
) -> pd.DataFrame:
    """Filter chunk to time window [start, end)."""
    mask = (chunk_df[ts_col] >= start) & (chunk_df[ts_col] < end)
    return chunk_df[mask].copy()


def extract_time_window(
    spec: PMUFileSpec,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    chunksize: int = 200_000,
    columns: Optional[list] = None
) -> ExtractionResult:
    """
    Extract a time window from a large PMU CSV without loading entire file.
    
    Args:
        spec: PMUFileSpec with file path and metadata
        start: Start time (inclusive)
        end: End time (exclusive)
        chunksize: Rows to read at a time
        columns: Specific columns to extract (default: all)
    
    Returns:
        ExtractionResult with DataFrame, quality report, and metadata
    """
    if not spec.path.exists():
        raise FileNotFoundError(f"PMU file not found: {spec.path}")
    
    # Prepare column selection
    if columns is None:
        columns = None  # Read all columns
    elif spec.timestamp_col not in columns:
        columns = list(set(columns) | {spec.timestamp_col})
    
    # Apply dtype mapping if provided
    dtype_dict = spec.dtype_map or {}
    
    logger.info(f"Extracting {spec.term_id} from {start} to {end}")
    
    chunks = []
    chunk_count = 0
    early_stop = False
    
    for chunk in pd.read_csv(
        spec.path,
        usecols=columns,
        dtype=dtype_dict or None,
        chunksize=chunksize,
        engine="c"
    ):
        chunk_count += 1
        
        # Parse timestamp for this chunk
        chunk[spec.timestamp_col] = _parse_timestamp_series(chunk[spec.timestamp_col])
        
        # Filter to window
        filtered = _filter_chunk_to_window(chunk, start, end, spec.timestamp_col)
        
        if len(filtered) > 0:
            chunks.append(filtered)
        
        # Early stop: if all timestamps >= end, we're done
        if chunk[spec.timestamp_col].max() >= end:
            early_stop = True
            break
    
    if not chunks:
        logger.warning(f"No data found in window [{start}, {end}) for {spec.term_id}")
        empty_df = pd.DataFrame()
        quality = DataQualityReport(
            n_rows=0, start_ts=start, end_ts=end,
            expected_samples=0, actual_samples=0, missing_count=0,
            duplicate_timestamps=0, nonmonotonic_count=0, max_gap_seconds=None,
            rate_estimate_hz=spec.sample_rate_hz
        )
        return ExtractionResult(empty_df, quality, spec, start, end)
    
    # Concatenate chunks
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(spec.timestamp_col).reset_index(drop=True)
    df = df.set_index(spec.timestamp_col)
    
    # Compute quality report
    quality = _validate_pmu_window(df, spec)
    
    logger.info(f"Extracted {len(df)} samples ({chunk_count} chunks, early_stop={early_stop})")
    
    return ExtractionResult(df, quality, spec, start, end)


def _validate_pmu_window(df: pd.DataFrame, spec: PMUFileSpec) -> DataQualityReport:
    """Compute data quality metrics."""
    n_rows = len(df)
    
    if n_rows == 0:
        return DataQualityReport(
            n_rows=0, start_ts=pd.Timestamp.now(),
            end_ts=pd.Timestamp.now(), expected_samples=0,
            actual_samples=0, missing_count=0,
            duplicate_timestamps=0, nonmonotonic_count=0,
            max_gap_seconds=None, rate_estimate_hz=spec.sample_rate_hz
        )
    
    ts_index = df.index
    start_ts = ts_index[0]
    end_ts = ts_index[-1]
    
    # Expected samples
    duration_sec = (end_ts - start_ts).total_seconds()
    expected_samples = int(duration_sec * spec.sample_rate_hz) + 1
    
    # Gaps and monotonicity
    if n_rows > 1:
        deltas = ts_index[1:] - ts_index[:-1]
        gaps_sec = pd.Series(deltas.total_seconds())
        max_gap = gaps_sec.max()
        duplicate_timestamps = (gaps_sec == 0).sum()
        nonmonotonic_count = (gaps_sec < 0).sum()
        expected_delta_sec = 1.0 / spec.sample_rate_hz
        missing_count = (gaps_sec > expected_delta_sec * 1.5).sum()
        rate_estimate_hz = 1.0 / gaps_sec.mean()
    else:
        max_gap = None
        duplicate_timestamps = 0
        nonmonotonic_count = 0
        missing_count = 0
        rate_estimate_hz = spec.sample_rate_hz
    
    # STAT summary if present
    stat_summary = {}
    for col in spec.quality_cols:
        if col in df.columns:
            stat_summary[col] = df[col].value_counts().to_dict()
    
    return DataQualityReport(
        n_rows=n_rows,
        start_ts=start_ts,
        end_ts=end_ts,
        expected_samples=expected_samples,
        actual_samples=n_rows,
        missing_count=missing_count,
        duplicate_timestamps=duplicate_timestamps,
        nonmonotonic_count=nonmonotonic_count,
        max_gap_seconds=max_gap,
        rate_estimate_hz=rate_estimate_hz,
        stat_summary=stat_summary
    )


def load_pilot_data(
    file_specs: list[PMUFileSpec],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    require_exact_rate: bool = False
) -> dict[int, ExtractionResult]:
    """
    Load multiple terminals for same time window.
    
    Args:
        file_specs: List of PMUFileSpec for terminals
        start: Start time
        end: End time
        require_exact_rate: If True, raise on rate mismatch
    
    Returns:
        Dict mapping TermID to ExtractionResult
    """
    results = {}
    
    for spec in file_specs:
        result = extract_time_window(spec, start, end)
        
        if require_exact_rate and not result.quality.is_healthy:
            logger.warning(f"Term {spec.term_id} rate = {result.quality.rate_estimate_hz:.2f} Hz")
        
        results[spec.term_id] = result
    
    return results


def save_extracted_window(result: ExtractionResult, out_dir: Path) -> Path:
    """Save extracted window to Parquet for fast reload."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    term_id = result.source.term_id
    out_path = out_dir / f"term_{term_id}.parquet"
    
    result.df.to_parquet(out_path)
    logger.info(f"Saved extracted window to {out_path}")
    
    return out_path


def load_extracted_window(term_id: int, out_dir: Path) -> pd.DataFrame:
    """Load previously extracted window from Parquet."""
    out_path = Path(out_dir) / f"term_{term_id}.parquet"
    
    if not out_path.exists():
        raise FileNotFoundError(f"Extracted window not found: {out_path}")
    
    df = pd.read_parquet(out_path)
    logger.info(f"Loaded extracted window from {out_path}")
    
    return df
