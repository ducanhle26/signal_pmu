# Signal PMU Implementation Plan

## IEEE Transactions on Smart Grid Submission

---

## Executive Summary

**Objective**: Implement defensible PMU anomaly detection for grid disturbances under extreme class imbalance, unreliable event logs, and temporal misalignment.

**Pilot Study**: SectionID 80 (Lightning event on 2020-08-31)

- **TermIDs**: 249, 252, 372 (69kV Fixico-Forest Hill-Maud Tap line)
- **Time Window**: 22:00:00 to 00:00:00 (2 hours)
- **Event Label**: ~22:57:00 (unreliable)
- **Data**: 3 PMU files × 2 hours @ 30Hz ≈ 216,000 samples

---

## Phase 1: Data Infrastructure & Extraction

### 1.1 Data Extraction Module

**File**: `src/data_loader.py`

**Purpose**: Efficient extraction of 2-hour windows from large PMU files without loading entire datasets.

**Key Functions**:

```python
extract_time_window(pmu_file, start_time, end_time)
    - Uses chunked reading (pandas chunksize=10000)
    - Converts UTC timestamps to datetime
    - Filters to time window
    - Returns DataFrame with datetime index

load_pilot_data(term_ids, date, start_hour, end_hour)
    - Loads multiple terminals for same time window
    - Returns dict: {term_id: DataFrame}
    - Validates sampling rate and data completeness
```

**Design Decisions**:

- Memory-efficient: Never load full 480MB files
- Robust time parsing: Handle millisecond precision UTC
- Data quality checks: Detect gaps, check STAT flags
- Export option: Save extracted windows for faster reloading

**Deliverables**:

- [ ] `data_loader.py` with chunked reading
- [ ] Unit tests for time window extraction
- [ ] Pilot data extraction: TermIDs [249, 252, 372], 22:00-00:00
- [ ] Data quality report (gaps, sampling irregularities)

---

### 1.2 Topology Integration Module

**File**: `src/topology.py`

**Purpose**: Link event metadata to PMU terminals and enable spatial analysis.

**Key Functions**:

```python
load_topology()
    - Parses topology.csv
    - Handles timestamp formats
    - Returns indexed DataFrame

get_event_info(section_id)
    - Returns all terminals for a section
    - GPS coordinates, voltage levels
    - Event cause and timing

get_spatial_neighbors(term_id, max_distance_km)
    - Calculate distances using GPS coordinates
    - Return neighboring terminals
    - For spatial consistency rules
```

**Design Decisions**:

- Separate event logs from ground truth (logs are weak evidence only)
- Enable spatial analysis for coherence metrics
- Prepare for multi-event expansion

**Deliverables**:

- [ ] `topology.py` with event and spatial queries
- [ ] SectionID 80 event summary
- [ ] Spatial proximity matrix for pilot terminals

---

## Phase 2: Signal Processing & Feature Engineering

### 2.1 Preprocessing Pipeline

**File**: `src/preprocessing.py`

**Purpose**: Clean and prepare PMU signals for dynamic analysis.

**Key Functions**:

```python
preprocess_pmu_signals(df, channels)
    - Handle missing values (interpolation or flagging)
    - Check STAT flags for data quality
    - Optional: Remove DC offset
    - Optional: Bandpass filtering (0.1-10 Hz for electromechanical modes)

select_analysis_channels(df, mode='voltage_magnitude')
    - Extract relevant channels (VP_M, VA_M, VB_M, VC_M)
    - Or current magnitude, frequency, angles
    - Return clean signal matrix
```

**Design Decisions**:

- Minimal preprocessing (avoid over-smoothing)
- Preserve transient signatures
- Document all filtering choices
- Flag questionable data instead of removing

**Deliverables**:

- [ ] `preprocessing.py` with channel selection
- [ ] Preprocessed pilot signals (3 terminals × selected channels)
- [ ] Quality flags for each terminal

---

### 2.2 Dynamic Modeling Module

**File**: `src/dynamic_models.py`

**Purpose**: Estimate baseline dynamics and unexplained excitation using subspace methods.

**Key Functions**:

```python
fit_var_model(data, order, window_size)
    - Vector Autoregression (VAR) for multi-PMU dynamics
    - Rolling window estimation
    - Returns model parameters and residuals

compute_residual_excitation(data, model)
    - Calculate unexplained forcing
    - Energy metric: ||residual||^2
    - Per-window aggregation

extract_dynamic_subspace(data, n_components, method='svd')
    - Singular Value Decomposition or PCA
    - Extract dominant dynamic modes
    - Return basis vectors and singular values
```

**Design Decisions**:

- Use VAR (Vector Autoregression) for baseline dynamics
  - Captures temporal and spatial dependencies
  - Order selection: AIC/BIC or fixed (e.g., 10-30 samples @ 30Hz)
- Residuals represent unexplained excitation
- Subspace method: SVD for robustness

**Deliverables**:

- [ ] `dynamic_models.py` with VAR and subspace extraction
- [ ] Model order selection analysis for pilot data
- [ ] Baseline dynamics for normal period (22:00-22:45)

---

## Phase 3: Anomaly Detection Metrics

### 3.1 Primary Metric: Residual Excitation Energy

**File**: `src/metrics/residual_energy.py`

**Purpose**: Core anomaly metric based on unexplained forcing.

**Key Functions**:

```python
compute_residual_energy(residuals, window_size, overlap)
    - Rolling window energy: sum(residual^2)
    - Normalize by signal variance
    - Returns time series of energy metric

detect_excitation_anomalies(energy, threshold, persistence_k)
    - Flag windows exceeding threshold
    - Require K consecutive windows (Principle A)
    - Return anomaly timestamps and scores
```

**Design Decisions**:

- Window size: 3-10 seconds (90-300 samples @ 30Hz)
- Overlap: 50% for smooth detection
- Persistence K: 2-5 windows (test sensitivity)
- Threshold: Data-driven (99th percentile of normal period)

**Deliverables**:

- [ ] `residual_energy.py` with rolling window computation
- [ ] Threshold selection for pilot data
- [ ] Residual energy time series for 22:00-00:00

---

### 3.2 Secondary Metric: Subspace Change

**File**: `src/metrics/subspace_change.py`

**Purpose**: Detect structural changes in dynamic modes (Principle B).

**Key Functions**:

```python
compute_principal_angles(subspace1, subspace2)
    - Canonical angles between subspaces
    - Measure of subspace rotation

compute_subspace_distance(data, baseline_subspace, window_size)
    - Rolling window subspace extraction
    - Compare to baseline (normal period)
    - Return distance metric time series

detect_subspace_anomalies(distance, threshold, persistence_k)
    - Flag significant subspace changes
    - Persistence requirement
    - Return anomaly timestamps
```

**Design Decisions**:

- Baseline subspace: Estimate from 22:00-22:45 (pre-event)
- Subspace dimension: 3-5 components (capture 80-90% variance)
- Distance metric: Principal angles or projection error
- Combine with spatial coherence

**Deliverables**:

- [ ] `subspace_change.py` with principal angle computation
- [ ] Baseline subspace from normal period
- [ ] Subspace distance time series for 22:00-00:00

---

### 3.3 Spatial Coherence Module

**File**: `src/metrics/spatial_coherence.py`

**Purpose**: Enforce spatial consistency (Principle B & Section 3 of design notes).

**Key Functions**:

```python
compute_cross_pmu_coherence(pmu_data_dict, freq_bands)
    - Coherence function between PMU pairs
    - Frequency-domain correlation
    - Identify coherent vs incoherent modes

apply_spatial_voting(anomaly_flags, neighbor_matrix, min_fraction)
    - Local consistency: m% of neighbors must agree
    - Remove isolated false alarms
    - Return spatially-validated anomalies

system_level_detection(anomaly_flags_all_pmus, top_percent)
    - System rule: top 5% PMUs exceed threshold
    - Or: anomalies in ≥2 distinct regions
    - Suppress sensor-specific noise
```

**Design Decisions**:

- Local rule: 10-20% of neighbors, minimum 2
  - For pilot: 3 terminals = use 2/3 agreement (66%)
- System rule: Require 2/3 terminals for event declaration
- Coherence analysis: Cross-correlation and frequency coherence

**Deliverables**:

- [ ] `spatial_coherence.py` with voting logic
- [ ] Cross-terminal coherence analysis for pilot
- [ ] Spatially-validated anomaly detections

---

## Phase 4: Validation & Alignment

### 4.1 Time Alignment Module

**File**: `src/validation/time_alignment.py`

**Purpose**: Handle temporal misalignment between PMUs and event logs.

**Key Functions**:

```python
estimate_pmu_lag_distribution(pmu_data_dict)
    - Cross-correlation between PMU pairs
    - Estimate typical delays
    - Return lag statistics (median, 95th percentile)

create_tolerance_window(event_time, lag_stats, margin_sec)
    - Detection tolerance: ±(P95_lag + margin)
    - Example: ±30 seconds for alignment
    - Data-driven, not arbitrary

match_detection_to_log(detections, event_log, tolerance)
    - Check temporal overlap (not exact match)
    - Qualitative consistency only
    - Return overlap analysis
```

**Design Decisions**:

- Tolerance window: 95th percentile lag + 10 sec margin
- Event logs as weak evidence, not labels
- Report both aligned and unaligned detections
- Explicitly state unreliability in results

**Deliverables**:

- [ ] `time_alignment.py` with lag estimation
- [ ] PMU lag distribution for pilot terminals
- [ ] Alignment analysis: detection vs 22:57:00 label

---

### 4.2 Internal Consistency Validation

**File**: `src/validation/internal_consistency.py`

**Purpose**: Validate without relying on noisy labels (Section 4.D of design notes).

**Key Functions**:

```python
compute_false_alarm_rate(detections, normal_period_indices)
    - False alarms during known quiet period
    - Baseline: 22:00-22:45 (pre-event)
    - Report FAR per hour

test_repeatability(detector, data, n_runs, parameter_variations)
    - Sensitivity to window length
    - Sensitivity to threshold choices
    - Assess robustness

cross_pmu_consistency_score(detections_by_pmu)
    - Do multiple PMUs detect simultaneously?
    - Temporal clustering of detections
    - Statistical significance test
```

**Design Decisions**:

- Normal period: 22:00-22:45 (45 minutes before event)
- Test period: 22:45-00:00 (includes event + post-event)
- Consistency checks stronger than noisy labels
- Report detection stability

**Deliverables**:

- [ ] `internal_consistency.py` with FAR and robustness tests
- [ ] False alarm rate during 22:00-22:45
- [ ] Cross-PMU consistency scores
- [ ] Parameter sensitivity analysis

---

## Phase 5: Visualization & Reporting

### 5.1 Visualization Module

**File**: `src/visualization/plots.py`

**Purpose**: Publication-quality figures for TSG submission.

**Key Plots**:

```python
plot_raw_signals(pmu_data, event_time, tolerance_window)
    - Multi-panel: 3 terminals × voltage magnitude
    - Highlight event window
    - Show full 2-hour context

plot_residual_energy_timeline(energy_series, threshold, detections)
    - Time series with detection markers
    - Threshold line
    - Event log overlay (with uncertainty band)

plot_subspace_distance_timeline(distance_series, threshold, detections)
    - Secondary metric visualization
    - Consistency with primary metric

plot_spatial_map(topology_df, detections_by_terminal)
    - GPS map with terminals
    - Color-code by detection timing
    - Show transmission line geometry

plot_coherence_matrix(coherence_results, freq_bands)
    - Cross-PMU coherence heatmap
    - Time-frequency analysis

plot_detection_summary(all_metrics, event_log, consistency_scores)
    - Combined view: primary + secondary + spatial
    - Decision fusion visualization
```

**Deliverables**:

- [ ] `plots.py` with all visualization functions
- [ ] Figure 1: Raw signals for SectionID 80
- [ ] Figure 2: Residual energy detection
- [ ] Figure 3: Subspace change analysis
- [ ] Figure 4: Spatial coherence and map
- [ ] Figure 5: Validation summary

---

### 5.2 Results Reporting Module

**File**: `src/reporting/results.py`

**Purpose**: Structured output for paper and reproducibility.

**Key Functions**:

```python
generate_detection_report(detections, metrics, validation)
    - Detection timestamps (all terminals)
    - Metric values at detection times
    - Spatial consistency scores
    - Alignment with event log
    - False alarm rate

generate_methodology_summary(config)
    - Document all parameter choices
    - Window sizes, thresholds, persistence K
    - Justification for each choice

export_results(results_dict, format='csv')
    - Save detections and metrics
    - Enable reproducibility
    - Format for supplementary materials
```

**Deliverables**:

- [ ] `results.py` with report generation
- [ ] Detection report for SectionID 80
- [ ] Methodology summary document
- [ ] CSV exports for supplementary materials

---

## Phase 6: Pipeline Integration & Automation

### 6.1 End-to-End Pipeline

**File**: `run_pilot_analysis.py`

**Purpose**: Automated execution of full analysis workflow.

**Workflow**:

```python
# 1. Data Loading
pilot_data = load_pilot_data(
    term_ids=[249, 252, 372],
    date='2020-08-31',
    start_time='22:00:00',
    end_time='00:00:00'
)

# 2. Preprocessing
signals = {tid: preprocess_pmu_signals(data) for tid, data in pilot_data.items()}

# 3. Baseline Modeling (normal period: 22:00-22:45)
baseline_period = slice_time_range(signals, '22:00:00', '22:45:00')
var_model = fit_var_model(baseline_period)
baseline_subspace = extract_dynamic_subspace(baseline_period)

# 4. Anomaly Detection
residuals = compute_residual_excitation(signals, var_model)
energy_metric = compute_residual_energy(residuals)
subspace_metric = compute_subspace_distance(signals, baseline_subspace)

# 5. Spatial Validation
detections_primary = detect_excitation_anomalies(energy_metric, threshold, k=3)
detections_secondary = detect_subspace_anomalies(subspace_metric, threshold, k=3)
detections_validated = apply_spatial_voting(detections_primary, neighbor_matrix)

# 6. Internal Consistency
far = compute_false_alarm_rate(detections_validated, normal_period_indices)
consistency = cross_pmu_consistency_score(detections_validated)

# 7. Log Alignment (weak evidence)
alignment = match_detection_to_log(detections_validated, event_log='22:57:00', tolerance=30)

# 8. Reporting
generate_detection_report(detections_validated, metrics, validation)
generate_all_plots(signals, metrics, detections_validated, topology)
```

**Deliverables**:

- [ ] `run_pilot_analysis.py` with full pipeline
- [ ] Configuration file: `config/pilot_config.yaml`
- [ ] Execution log with timestamps and intermediate results

---

### 6.2 Configuration Management

**File**: `config/pilot_config.yaml`

**Purpose**: Centralized parameter management for reproducibility.

**Configuration**:

```yaml
data:
  term_ids: [249, 252, 372]
  section_id: 80
  date: "2020-08-31"
  time_window:
    start: "22:00:00"
    end: "00:00:00"
  pmu_files:
    - "data/raw_pmu/249 2020-08-31.csv"
    - "data/raw_pmu/252 2020-08-31.csv"
    - "data/raw_pmu/372 2020-08-31.csv"

preprocessing:
  channels: ["VP_M", "VA_M", "VB_M", "VC_M"] # Voltage magnitudes
  interpolation: "linear"
  filter:
    enabled: false # Start without filtering
    type: "bandpass"
    freq_range: [0.1, 10] # Hz

modeling:
  var_order: 20 # ~0.67 seconds at 30 Hz
  window_size: 150 # 5 seconds at 30 Hz
  overlap: 0.5
  subspace_components: 5

detection:
  primary_metric: "residual_energy"
  secondary_metric: "subspace_change"
  threshold_method: "percentile"
  threshold_percentile: 99 # Based on normal period
  persistence_k: 3 # Consecutive windows

spatial:
  local_voting:
    min_neighbors: 2
    min_fraction: 0.66 # 2 out of 3 for pilot
  system_rule:
    min_terminals: 2 # At least 2/3 terminals

validation:
  normal_period:
    start: "22:00:00"
    end: "22:45:00"
  event_log:
    time: "22:57:00"
    tolerance_sec: 30
    reliability: "low" # Explicitly state
  internal_consistency:
    n_bootstrap: 100
    confidence_level: 0.95

output:
  results_dir: "results/pilot_section80/"
  figures_dir: "results/pilot_section80/figures/"
  export_format: "csv"
```

**Deliverables**:

- [ ] `config/pilot_config.yaml` with all parameters
- [ ] Configuration documentation explaining each choice

---

## Phase 7: Extension & Scalability

### 7.1 Multi-Event Analysis

**File**: `run_all_events.py`

**Purpose**: Extend pilot to all 14 events in topology.csv.

**Approach**:

- Reuse pilot pipeline
- Loop over all SectionIDs
- Extract appropriate time windows for each event
- Aggregate results for statistical analysis

**Deliverables**:

- [ ] `run_all_events.py` for batch processing
- [ ] Comparative analysis across events
- [ ] Event detection success rate

---

### 7.2 Real-Time / Streaming Mode

**File**: `src/streaming/online_detector.py`

**Purpose**: Adapt offline analysis to real-time detection.

**Design**:

- Incremental VAR model updates
- Streaming subspace tracking
- Online threshold adaptation
- Computational efficiency

**Deliverables**:

- [ ] Streaming detection prototype
- [ ] Latency analysis
- [ ] Resource usage profiling

---

## Implementation Timeline & Milestones

### Milestone 1: Data & Infrastructure (Week 1)

- [ ] Phase 1.1: Data extraction module
- [ ] Phase 1.2: Topology integration
- [ ] Pilot data extracted and validated

### Milestone 2: Signal Processing (Week 2)

- [ ] Phase 2.1: Preprocessing pipeline
- [ ] Phase 2.2: Dynamic modeling (VAR + subspace)
- [ ] Baseline dynamics established

### Milestone 3: Detection Metrics (Week 3)

- [ ] Phase 3.1: Residual energy metric
- [ ] Phase 3.2: Subspace change metric
- [ ] Phase 3.3: Spatial coherence
- [ ] Initial detections for pilot

### Milestone 4: Validation (Week 4)

- [ ] Phase 4.1: Time alignment analysis
- [ ] Phase 4.2: Internal consistency tests
- [ ] Detection validated for pilot

### Milestone 5: Reporting (Week 5)

- [ ] Phase 5.1: All visualizations
- [ ] Phase 5.2: Results reports
- [ ] Pilot analysis complete

### Milestone 6: Pipeline & Extension (Week 6)

- [ ] Phase 6.1: End-to-end pipeline
- [ ] Phase 6.2: Configuration management
- [ ] Ready for multi-event expansion

---

## Code Organization

```
Sig_pmu/
├── config/
│   └── pilot_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── topology.py
│   ├── preprocessing.py
│   ├── dynamic_models.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── residual_energy.py
│   │   ├── subspace_change.py
│   │   └── spatial_coherence.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── time_alignment.py
│   │   └── internal_consistency.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── results.py
│   └── streaming/
│       ├── __init__.py
│       └── online_detector.py
├── run_pilot_analysis.py
├── run_all_events.py
├── results/
│   └── pilot_section80/
│       ├── figures/
│       ├── detections.csv
│       ├── metrics.csv
│       └── report.txt
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_dynamic_models.py
│   └── test_metrics.py
├── requirements.txt
├── README.md
├── CLAUDE.md
├── PMU_TSG_Method_Design_Notes.md
└── IMPLEMENTATION_PLAN.md (this file)
```

---

## Dependencies & Environment

**Core Libraries**:

```
pandas>=2.0.0          # Data handling
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Signal processing, statistics
scikit-learn>=1.3.0    # PCA, preprocessing
statsmodels>=0.14.0    # VAR modeling
matplotlib>=3.7.0      # Visualization
seaborn>=0.12.0        # Statistical plots
pyyaml>=6.0            # Configuration files
pytest>=7.4.0          # Testing
```

**Optional**:

```
geopandas>=0.13.0      # Spatial visualization
folium>=0.14.0         # Interactive maps
numba>=0.57.0          # JIT compilation for speed
h5py>=3.9.0            # Efficient data storage
```

**Deliverables**:

- [ ] `requirements.txt` with pinned versions
- [ ] Environment setup instructions
- [ ] Docker container (optional, for reproducibility)

---

## Testing Strategy

### Unit Tests

- [ ] Data loader: Time window extraction accuracy
- [ ] Preprocessing: Signal quality validation
- [ ] VAR model: Residual properties
- [ ] Metrics: Edge cases and numerical stability

### Integration Tests

- [ ] End-to-end pipeline on pilot data
- [ ] Configuration loading and validation
- [ ] Output format consistency

### Validation Tests

- [ ] Known event detection (SectionID 80)
- [ ] False alarm rate on normal period
- [ ] Reproducibility: Same results with same config

---

## Success Criteria for Pilot

### Technical Success

1. **Detection Performance**:

   - Detect event around 22:57:00 ±30 seconds
   - False alarm rate <1 per hour during normal period
   - Spatial consistency: 2/3 terminals agree

2. **Methodological Rigor**:

   - All design choices documented and justified
   - Internal consistency validated
   - Sensitivity analysis complete

3. **Reproducibility**:
   - Pipeline runs end-to-end without manual intervention
   - Results stable across runs
   - Configuration-driven (easy to modify)

### Publication Readiness

1. **Figures**: 5 publication-quality plots
2. **Results**: Detection report with statistics
3. **Methods**: Clear documentation of defensible choices
4. **Code**: Clean, tested, documented

---

## Key Risks & Mitigation

### Risk 1: Event log timing unreliable

**Mitigation**:

- Use tolerance windows (±30 sec)
- Emphasize internal consistency validation
- Report both aligned and unaligned detections

### Risk 2: High false alarm rate

**Mitigation**:

- Strong spatial voting (2/3 terminals)
- Persistence requirement (K=3 windows)
- Adaptive thresholding on normal period

### Risk 3: Weak signal for 69kV events

**Mitigation**:

- Analyze multiple metrics (primary + secondary)
- Test various window sizes
- Consider frequency-based features

### Risk 4: Computational scalability

**Mitigation**:

- Pilot with 2-hour window first
- Optimize before expanding to all events
- Consider parallel processing for multi-event analysis

---

## Final Deliverable Checklist

### Code

- [ ] All modules implemented and tested
- [ ] `run_pilot_analysis.py` executes successfully
- [ ] Configuration file complete
- [ ] Documentation in README.md

### Results

- [ ] Detection report for SectionID 80
- [ ] 5 publication-quality figures
- [ ] Methodology summary
- [ ] CSV exports of detections and metrics

### Validation

- [ ] False alarm rate computed
- [ ] Cross-PMU consistency scores
- [ ] Parameter sensitivity analysis
- [ ] Comparison to event log (with caveats)

### Documentation

- [ ] Implementation plan (this document)
- [ ] Code comments and docstrings
- [ ] User guide for running analysis
- [ ] Parameter selection justification

---

## Anchor Statement for Paper

> "We detect grid anomalies by identifying sustained, spatially coherent excitation that cannot be explained by dominant oscillatory PMU dynamics, without relying on disturbance labels."

This implementation plan operationalizes this statement for IEEE TSG submission with defensible, reproducible methods.
