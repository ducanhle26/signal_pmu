"""Unit tests for topology.py."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.topology import (
    TopologyStore,
    load_topology,
    get_terminals_for_section,
    haversine_km,
    pairwise_distance_matrix
)


@pytest.fixture
def temp_topology_file():
    """Generate a temporary synthetic topology CSV."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        path = Path(f.name)
        f.write(
            "Timestamp,SectionID,TermID,Terminal,Substation,Voltage,Event_Location,Cause,Latitude,Longitude\n"
            "2020-08-31 22:57:00,80,249,T249,Sub1,69,Line1-Line2,Lightning,40.0,-100.0\n"
            "2020-08-31 22:57:00,80,252,T252,Sub2,69,Line1-Line2,Lightning,40.1,-100.0\n"
            "2020-08-31 22:57:00,80,372,T372,Sub3,69,Line1-Line2,Lightning,40.0,-99.9\n"
        )
    
    yield path
    path.unlink()


def test_load_topology_basic(temp_topology_file):
    """Test topology loading."""
    store = load_topology(temp_topology_file)
    
    assert len(store.terminals) == 3
    assert len(store.sections) == 1
    assert len(store.events) == 3


def test_get_terminals_for_section(temp_topology_file):
    """Test retrieving terminals for a section."""
    store = load_topology(temp_topology_file)
    
    term_ids = get_terminals_for_section(store, 80)
    
    assert set(term_ids) == {249, 252, 372}


def test_get_terminals_for_missing_section(temp_topology_file):
    """Test error for missing section."""
    store = load_topology(temp_topology_file)
    
    with pytest.raises(ValueError):
        get_terminals_for_section(store, 999)


def test_haversine_distance_zero():
    """Test Haversine for same point."""
    d = haversine_km(40.0, -100.0, 40.0, -100.0)
    
    assert d < 0.001


def test_haversine_distance_nonzero():
    """Test Haversine for different points."""
    # Two points ~11 km apart
    d = haversine_km(40.0, -100.0, 40.1, -100.0)
    
    assert d > 10
    assert d < 12


def test_pairwise_distance_matrix_symmetric(temp_topology_file):
    """Test that distance matrix is symmetric."""
    store = load_topology(temp_topology_file)
    
    term_ids = [249, 252, 372]
    matrix = pairwise_distance_matrix(store, term_ids)
    
    assert matrix.shape == (3, 3)
    # Check symmetry
    assert (matrix == matrix.T).all().all()
    # Check diagonal is zero
    assert (matrix.values.diagonal() == 0).all()


def test_file_not_found():
    """Test error handling for missing topology file."""
    with pytest.raises(FileNotFoundError):
        load_topology(Path('/nonexistent/topology.csv'))
