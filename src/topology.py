"""Event metadata and spatial analysis integration."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TopologyStore:
    """In-memory repository for topology and event data."""

    terminals: pd.DataFrame
    sections: pd.DataFrame
    events: pd.DataFrame


def load_topology(
    topology_path: Path, events_path: Optional[Path] = None
) -> TopologyStore:
    """
    Load topology and event metadata.

    Args:
        topology_path: Path to topology.csv
        events_path: Optional path to separate events file

    Returns:
        TopologyStore with terminals, sections, and events DataFrames
    """
    topology_path = Path(topology_path)

    if not topology_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    # Load main topology
    df = pd.read_csv(topology_path)
    logger.info(f"Loaded topology: {len(df)} rows")

    # Parse timestamps
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)

    # Build sections dataframe (unique by SectionID)
    sections = (
        df.groupby("SectionID")
        .agg(
            {
                "Event_Location": "first",
                "Cause": "first",
                "Voltage": "first",
                "Timestamp": "first",
            }
        )
        .reset_index()
    )

    # Build terminals dataframe (unique by TermID with spatial info)
    terminals = (
        df.groupby("TermID")
        .agg(
            {
                "Terminal": "first",
                "Substation": "first",
                "Latitude": "first",
                "Longitude": "first",
            }
        )
        .reset_index()
    )

    # Events dataframe: all rows with timestamp info
    events = df[
        [
            "Timestamp",
            "SectionID",
            "TermID",
            "Event_Location",
            "Cause",
            "Voltage",
        ]
    ].copy()

    logger.info(
        f"Parsed: {len(terminals)} terminals, {len(sections)} sections, {len(events)} events"
    )

    return TopologyStore(terminals=terminals, sections=sections, events=events)


def get_event_info(store: TopologyStore, section_id: int) -> dict:
    """
    Get event metadata for a SectionID.

    Returns:
        Dict with terminals, coordinates, voltage, cause, event_time
    """
    section_events = store.events[store.events["SectionID"] == section_id]

    if len(section_events) == 0:
        raise ValueError(f"SectionID {section_id} not found")

    term_ids = section_events["TermID"].unique().tolist()

    # Get terminal details
    term_details = store.terminals[store.terminals["TermID"].isin(term_ids)]

    return {
        "section_id": section_id,
        "term_ids": term_ids,
        "terminals": term_details.to_dict("records"),
        "voltage_kv": section_events["Voltage"].iloc[0],
        "event_location": section_events["Event_Location"].iloc[0],
        "cause": section_events["Cause"].iloc[0],
        "event_time": section_events["Timestamp"].iloc[
            0
        ],  # First occurrence (weak label)
    }


def get_terminals_for_section(
    store: TopologyStore, section_id: int
) -> list[int]:
    """Get all TermIDs affected by a SectionID."""
    result = (
        store.events[store.events["SectionID"] == section_id]["TermID"]
        .unique()
        .tolist()
    )
    if not result:
        raise ValueError(f"SectionID {section_id} not found")
    return result


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in km."""
    from math import atan2, cos, radians, sin, sqrt

    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def get_spatial_neighbors(
    store: TopologyStore, term_id: int, max_distance_km: float = 50.0
) -> pd.DataFrame:
    """
    Get terminals within max_distance of term_id.

    Returns:
        DataFrame with neighbor TermIDs and distances
    """
    term = store.terminals[store.terminals["TermID"] == term_id]

    if len(term) == 0:
        raise ValueError(f"TermID {term_id} not found")

    lat, lon = term.iloc[0][["Latitude", "Longitude"]]

    if pd.isna(lat) or pd.isna(lon):
        logger.warning(f"TermID {term_id} has missing coordinates")
        return pd.DataFrame()

    # Compute distances to all other terminals
    distances = []
    for _, row in store.terminals.iterrows():
        if pd.isna(row["Latitude"]) or pd.isna(row["Longitude"]):
            continue

        d = haversine_km(lat, lon, row["Latitude"], row["Longitude"])
        if d <= max_distance_km:
            distances.append(
                {
                    "TermID": row["TermID"],
                    "Terminal": row["Terminal"],
                    "Substation": row["Substation"],
                    "Distance_km": d,
                }
            )

    if not distances:
        return pd.DataFrame()

    return pd.DataFrame(distances).sort_values("Distance_km")


def pairwise_distance_matrix(
    store: TopologyStore, term_ids: list[int]
) -> pd.DataFrame:
    """
    Compute pairwise distances for a set of TermIDs.

    Returns:
        Symmetric matrix (as DataFrame)
    """
    n = len(term_ids)
    distances = [[0.0] * n for _ in range(n)]

    term_map = {tid: i for i, tid in enumerate(term_ids)}

    for term_id in term_ids:
        neighbors = get_spatial_neighbors(store, term_id, max_distance_km=1000)
        for _, row in neighbors.iterrows():
            nid = row["TermID"]
            if nid in term_map:
                i = term_map[term_id]
                j = term_map[nid]
                distances[i][j] = row["Distance_km"]
                distances[j][i] = row["Distance_km"]

    return pd.DataFrame(distances, index=term_ids, columns=term_ids)
