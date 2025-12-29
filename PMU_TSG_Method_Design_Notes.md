# PMU Anomaly Detection for IEEE Transactions on Smart Grid  
## Method Design Notes and Defensible Choices

This document clarifies methodological choices for PMU anomaly detection under extreme class imbalance, complex normal dynamics, unreliable event logs, and temporal misalignment.  
The focus is on **defensible design decisions**, not algorithms or code.

---

## 1. Rare anomalies and many normal modes  
### Why anomalies should not be defined as "new modes"

In large scale power systems, normal operation already contains many oscillatory modes, including low energy and intermittent ones.  
Therefore, anomaly detection **must not rely on the appearance of new or weak modes alone**.

### Key principle  
An anomaly is defined as a **structural change that is statistically and physically inconsistent with baseline variability**, not merely a spectral novelty.

### Three robustness principles

#### Principle A. Persistence over time  
Weak normal modes may appear in a single window due to noise or transient behavior.  
True anomalies persist across multiple windows or show organized temporal evolution.

Defensible rule:
- An anomaly must persist for *K* consecutive windows
- Typical range: 2 to 5 windows, depending on window length

#### Principle B. Coherence over amplitude  
Normal low energy modes are often incoherent across PMUs.  
Grid events alter **how PMUs move together**, not just signal energy.

Track:
- Changes in spatial mode shapes
- Loss or gain of inter PMU coherence
- Drift of dominant dynamic subspaces

These effects are difficult to explain by random weak modes.

#### Principle C. Excitation over explanation  
In a linear plus residual representation:
- Normal modes are explained by dominant dynamics
- Disturbances generate **unexplained excitation**

Thus, anomalies are not defined as new modes, but as **persistent unexplained excitation that is spatially coherent**.

---

## 2. Simple anomaly metric  
### How to know whether a metric is good

Metrics are chosen by **requirements**, not popularity.

A valid PMU anomaly metric must satisfy:

1. Stability during normal operation  
2. Sensitivity during real disturbances  
3. Robustness to noise and window selection  
4. Interpretability in grid operation terms  

### Metric families aligned with this framework

#### Option 1. Residual excitation energy  
Examples:
- Norm or energy of unexplained forcing
- Aggregated over short windows

Why suitable:
- Directly matches anomaly definition
- Interpretable as external grid excitation
- Naturally sparse in time

#### Option 2. Dynamic subspace change  
Examples:
- Principal angles between subspaces
- Projection error onto baseline subspace

Why suitable:
- Robust to many weak modes
- Effective in high dimensional PMU settings

#### Option 3. Mode stability shift  
Examples:
- Eigenvalue movement
- Reduced damping or growing oscillations

Why suitable:
- Strong physical interpretation
- Aligned with power system stability analysis

### Recommended practice
- Select **one primary metric**
- Add **one secondary metric** for consistency checks

A defensible combination:
- Primary: residual excitation energy  
- Secondary: subspace change metric  

---

## 3. Spatial consistency  
### How many votes and how many hops

Avoid fixed numbers such as "2 or 3 PMUs".  
Use **scalable and adaptive rules**.

### Level 1. Local consistency  
A PMU is considered anomalous only if:
- Its local metric is high
- And at least *m* neighboring PMUs also show high values

How to select *m*:
- Use a fraction of neighbors, not a constant
- Example: 10 to 20 percent of adjacent PMUs
- Enforce a minimum of 2 neighbors

This adapts to network density.

### Level 2. System or regional consistency  
A system level event is declared only if:
- A minimum fraction of PMUs show anomalous behavior simultaneously

Defensible choices:
- Top 5 percent of PMUs exceed threshold
- Or anomalies appear in at least two distinct regions

This suppresses sensor specific false alarms.

### 1 hop vs 2 hops
- Use 1 hop for detection
- Use 2 hops only to study propagation or confirmation

Detection should remain simple.

---

## 4. Validation without reliable labels  
### Handling incomplete logs and time misalignment

#### A. Event logs as weak evidence  
Event logs are used for:
- Temporal overlap analysis
- Qualitative consistency checks

They are **not used as training labels**.

Explicitly state:
- Logs are incomplete
- Timing may be inaccurate

This honesty is critical for TSG.

---

#### B. Time misalignment handling  
PMU channels and logs may have delays.

Use a **tolerance window** instead of exact alignment.

Defensible approach:
- Estimate lag distributions between PMUs using cross correlation
- Set tolerance as the 95th percentile lag plus a margin

This makes alignment data driven.

---

#### C. Detection window vs persistence window

Use two time scales:

1. Detection tolerance window  
   - Seconds level  
   - For alignment across channels and logs  

2. Persistence window  
   - Longer duration  
   - Used to reject noise and confirm events  

Both must be reported clearly.

---

#### D. Validation without labels at all

Even without usable logs, validate using:
- Cross PMU consistency
- Repeatability across calm periods
- False alarm rate during known quiet intervals
- Sensitivity to window length and thresholds

These internal consistency checks are often stronger than noisy labels.

---

## 5. A defensible baseline configuration

One example configuration that is realistic and publishable:

- Anomaly definition: persistent, spatially coherent unexplained excitation  
- Primary metric: residual excitation energy  
- Secondary metric: dynamic subspace change  
- Local spatial rule: 10 to 20 percent of neighbors, minimum 2  
- System rule: top 5 percent PMUs exceed threshold  
- Time alignment: data driven lag tolerance window  
- Validation: weak log overlap plus internal consistency tests  

---

## Final anchor statement

We detect grid anomalies by identifying sustained, spatially coherent excitation that cannot be explained by dominant oscillatory PMU dynamics, without relying on disturbance labels.
