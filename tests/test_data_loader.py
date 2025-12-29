"""Unit tests for data_loader.py."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import csv

from src.data_loader import (
    PMUFileSpec,
    extract_time_window,
    _parse_timestamp_series,
    DataQualityReport
)


@pytest.fixture
def temp_pmu_file():
    """Generate a temporary synthetic PMU CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        path = Path(f.name)
        
        # Write header
        writer = csv.writer(f)
        writer.writerow(['UTC', 'VP_M', 'VA_M', 'VB_M', 'VC_M', 'STAT'])
        
        # Write 3600 rows (2 hours @ 30 Hz)
        start = datetime(2020, 8, 31, 22, 0, 0)
        interval = timedelta(seconds=1/30)
        
        for i in range(3600):
            ts = start + i * interval
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # millisecond precision
            writer.writerow([ts_str, 40000 + np.random.randn()*100, 40000, 40000, 40000, 0])
    
    yield path
    path.unlink()


def test_parse_timestamp_series():
    """Test timestamp parsing."""
    ts = pd.Series([
        '2020-08-31 22:00:00.000',
        '2020-08-31 22:00:00.033',
        '2020-08-31 22:00:00.067'
    ])
    
    result = _parse_timestamp_series(ts)
    
    assert len(result) == 3
    assert result.isna().sum() == 0
    assert 'UTC' in str(result.dtype)  # UTC timezone


def test_extract_time_window_basic(temp_pmu_file):
    """Test basic extraction."""
    spec = PMUFileSpec(
        path=temp_pmu_file,
        term_id=249,
        timestamp_col='UTC',
        sample_rate_hz=30.0
    )
    
    start = pd.Timestamp('2020-08-31 22:00:00', tz='UTC')
    end = pd.Timestamp('2020-08-31 22:30:00', tz='UTC')
    
    result = extract_time_window(spec, start, end, chunksize=100)
    
    assert len(result.df) > 0
    assert result.df.index[0] >= start
    assert result.df.index[-1] < end
    assert result.quality.rate_estimate_hz > 29.0  # Close to 30 Hz


def test_extract_time_window_inclusive_exclusive(temp_pmu_file):
    """Test that window is [start, end)."""
    spec = PMUFileSpec(
        path=temp_pmu_file,
        term_id=249,
        timestamp_col='UTC',
        sample_rate_hz=30.0
    )
    
    start = pd.Timestamp('2020-08-31 22:00:00', tz='UTC')
    end = pd.Timestamp('2020-08-31 22:01:00', tz='UTC')
    
    result = extract_time_window(spec, start, end, chunksize=100)
    
    assert result.df.index[0] >= start
    assert result.df.index[-1] < end
    assert all(result.df.index >= start)
    assert all(result.df.index < end)


def test_extract_empty_window(temp_pmu_file):
    """Test extraction with no data in window."""
    spec = PMUFileSpec(
        path=temp_pmu_file,
        term_id=249,
        timestamp_col='UTC',
        sample_rate_hz=30.0
    )
    
    start = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
    end = pd.Timestamp('2025-01-01 01:00:00', tz='UTC')
    
    result = extract_time_window(spec, start, end, chunksize=100)
    
    assert len(result.df) == 0
    assert result.quality.n_rows == 0


def test_quality_report_rate_healthy(temp_pmu_file):
    """Test quality report for healthy data."""
    spec = PMUFileSpec(
        path=temp_pmu_file,
        term_id=249,
        timestamp_col='UTC',
        sample_rate_hz=30.0
    )
    
    start = pd.Timestamp('2020-08-31 22:00:00', tz='UTC')
    end = pd.Timestamp('2020-08-31 22:10:00', tz='UTC')
    
    result = extract_time_window(spec, start, end, chunksize=100)
    
    assert result.quality.is_healthy
    assert result.quality.duplicate_timestamps == 0


def test_file_not_found():
    """Test error handling for missing file."""
    spec = PMUFileSpec(
        path=Path('/nonexistent/file.csv'),
        term_id=999,
        timestamp_col='UTC'
    )
    
    start = pd.Timestamp('2020-08-31 22:00:00', tz='UTC')
    end = pd.Timestamp('2020-08-31 22:01:00', tz='UTC')
    
    with pytest.raises(FileNotFoundError):
        extract_time_window(spec, start, end)


def test_quality_report_fields():
    """Test DataQualityReport dataclass."""
    report = DataQualityReport(
        n_rows=900,
        start_ts=pd.Timestamp('2020-08-31 22:00:00', tz='UTC'),
        end_ts=pd.Timestamp('2020-08-31 22:30:00', tz='UTC'),
        expected_samples=900,
        actual_samples=900,
        missing_count=0,
        duplicate_timestamps=0,
        nonmonotonic_count=0,
        max_gap_seconds=0.033,
        rate_estimate_hz=30.0
    )
    
    assert report.is_healthy
    assert report.n_rows == 900
