"""Tertiary metric: Spatial coherence voting across terminals."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_spatial_voting(
    anomaly_dict: dict,
    vote_threshold: float = 0.66,
    weight_by_proximity: bool = False,
    distance_matrix: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Apply spatial voting across terminals (Principle C).
    
    Requires vote_threshold fraction of terminals to agree on anomaly.
    
    Args:
        anomaly_dict: Dict mapping term_id -> anomaly DataFrame
        vote_threshold: Fraction of terminals that must agree (e.g., 0.66 = 2/3)
        weight_by_proximity: If True, weight votes by spatial proximity
        distance_matrix: Distance matrix for terminals (used if weight_by_proximity=True)
    
    Returns:
        DataFrame with spatially consistent detections
    """
    # Align all timestamps
    all_timestamps = set()
    for anom_df in anomaly_dict.values():
        all_timestamps.update(anom_df.index)
    
    all_timestamps = sorted(all_timestamps)
    term_ids = sorted(anomaly_dict.keys())
    n_terminals = len(term_ids)
    
    # Build voting matrix
    voting_matrix = {}
    
    for ts in all_timestamps:
        votes = []
        weights = []
        
        for i, term_id in enumerate(term_ids):
            anom_df = anomaly_dict[term_id]
            
            if ts in anom_df.index:
                is_anomaly = anom_df.loc[ts, 'is_anomaly']
                votes.append(1.0 if is_anomaly else 0.0)
            else:
                votes.append(0.0)
            
            # Weight by proximity if available
            if weight_by_proximity and distance_matrix is not None:
                # Closer = higher weight
                ref_term = term_ids[0]
                dist = distance_matrix.loc[ref_term, term_id]
                weight = 1.0 / (1.0 + dist)  # Soft weighting
                weights.append(weight)
            else:
                weights.append(1.0)
        
        # Compute weighted vote
        if weight_by_proximity:
            weighted_votes = np.array(votes) * np.array(weights)
            total_weight = np.sum(weights)
            vote_fraction = np.sum(weighted_votes) / total_weight if total_weight > 0 else 0
        else:
            vote_fraction = np.mean(votes)
        
        # Determine if spatially consistent
        is_consistent = vote_fraction >= vote_threshold
        n_votes = int(np.sum(votes))
        
        voting_matrix[ts] = {
            'vote_fraction': vote_fraction,
            'n_votes': n_votes,
            'is_consistent': is_consistent
        }
    
    result = pd.DataFrame(voting_matrix).T
    result.index = pd.DatetimeIndex(result.index)
    
    consistent_count = result['is_consistent'].sum()
    logger.info(f"Spatial voting: {consistent_count} windows with {vote_threshold:.1%} agreement")
    
    return result


def compute_spatial_agreement_metrics(
    anomaly_dict: dict
) -> dict:
    """
    Compute agreement statistics across terminals.
    
    Args:
        anomaly_dict: Dict mapping term_id -> anomaly DataFrame
    
    Returns:
        Dict with pairwise agreement, event overlap stats
    """
    term_ids = sorted(anomaly_dict.keys())
    n_terminals = len(term_ids)
    
    # Pairwise agreement
    pairwise_agreement = np.zeros((n_terminals, n_terminals))
    
    for i, term_a in enumerate(term_ids):
        for j, term_b in enumerate(term_ids):
            if i == j:
                pairwise_agreement[i, j] = 1.0
            else:
                anom_a = anomaly_dict[term_a]
                anom_b = anomaly_dict[term_b]
                
                # Find overlapping timestamps
                overlap_ts = sorted(set(anom_a.index) & set(anom_b.index))
                
                if len(overlap_ts) > 0:
                    a_events = anom_a.loc[overlap_ts, 'is_anomaly']
                    b_events = anom_b.loc[overlap_ts, 'is_anomaly']
                    agreement = (a_events == b_events).sum() / len(overlap_ts)
                else:
                    agreement = 0.0
                
                pairwise_agreement[i, j] = agreement
    
    # Event overlap
    event_overlaps = []
    
    for i, term_a in enumerate(term_ids):
        events_a = set(anomaly_dict[term_a][anomaly_dict[term_a]['is_anomaly']].index)
        
        for j, term_b in enumerate(term_ids):
            if i < j:
                events_b = set(anomaly_dict[term_b][anomaly_dict[term_b]['is_anomaly']].index)
                overlap = len(events_a & events_b) / max(len(events_a | events_b), 1)
                event_overlaps.append({
                    'term_pair': f"({term_a}, {term_b})",
                    'overlap_fraction': overlap
                })
    
    return {
        'pairwise_agreement': pairwise_agreement,
        'term_ids': term_ids,
        'event_overlaps': event_overlaps,
        'mean_pairwise_agreement': np.mean(pairwise_agreement[np.triu_indices_from(pairwise_agreement, k=1)])
    }


def extract_spatially_consistent_events(
    spatial_votes: pd.DataFrame,
    min_duration_windows: int = 5
) -> list:
    """
    Extract events with sufficient spatial agreement.
    
    Args:
        spatial_votes: Output from apply_spatial_voting
        min_duration_windows: Minimum event duration
    
    Returns:
        List of dicts with event metadata
    """
    consistent_mask = spatial_votes['is_consistent']
    transitions = consistent_mask.astype(int).diff().fillna(0)
    group_id = (transitions != 0).cumsum()
    
    events = []
    event_count = 0
    
    for gid in group_id.unique():
        group = spatial_votes[group_id == gid]
        
        if consistent_mask[group.index].any():
            if len(group) >= min_duration_windows:
                event_count += 1
                
                events.append({
                    'event_id': event_count,
                    'start_time': group.index[0],
                    'end_time': group.index[-1],
                    'duration_windows': len(group),
                    'min_vote_fraction': group['vote_fraction'].min(),
                    'mean_vote_fraction': group['vote_fraction'].mean(),
                    'max_votes': int(group['n_votes'].max())
                })
    
    logger.info(f"Extracted {event_count} spatially consistent events")
    
    return events
