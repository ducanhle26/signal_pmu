# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for analyzing PMU (Phasor Measurement Unit) data from power grid transmission lines. The project focuses on analyzing electrical events (primarily lightning strikes) on the power grid using high-frequency measurements from PMU sensors.

## Data Architecture

### Data Structure

The repository contains two main data components:

1. **Raw PMU Data** (`data/raw_pmu/`)
   - 24 CSV files, each ~460-480MB
   - Named by TermID and date (e.g., "12 2020-08-31.csv", "249 2020-08-31.csv")
   - Each file contains time-series measurements from a specific terminal in the power grid
   - Sampling rate: ~30 Hz (33ms intervals)

2. **Topology Data** (`topology.csv`)
   - Maps event metadata to physical grid infrastructure
   - Links events to specific terminals, substations, and transmission lines
   - Contains GPS coordinates for spatial analysis

### PMU Data Format

Each PMU CSV file has the following columns:
- **UTC**: Timestamp (millisecond precision)
- **Voltage Measurements**: VP_M, VA_M, VB_M, VC_M (magnitude), VP_A, VA_A, VB_A, VC_A (angle)
- **Current Measurements**: IP_M, IA_M, IB_M, IC_M (magnitude), IP_A, IA_A, IB_A, IC_A (angle)
- **Frequency**: F (Hz), DF (rate of change)
- **STAT**: Status flag

Subscript notation:
- P = Positive sequence
- A, B, C = Three-phase components
- M = Magnitude
- A = Angle (degrees)

### Topology Data Format

The `topology.csv` file contains:
- **Timestamp**: Event occurrence time
- **SectionID**: Unique identifier for the transmission line section
- **Event_Location**: Description of affected transmission line
- **Cause**: Event cause (predominantly lightning)
- **TermID**: Terminal identifier (links to raw PMU filenames)
- **Terminal**: Terminal name and connected substations
- **Substation**: Primary substation name
- **Voltage**: Transmission line voltage level (kV)
- **Latitude/Longitude**: GPS coordinates

### Data Relationships

- Each row in `topology.csv` represents one terminal affected by an event
- Multiple terminals can be affected by the same event (same SectionID and timestamp)
- TermID in `topology.csv` corresponds to the filename prefix in `data/raw_pmu/`
- Events typically involve 2-3 terminals (endpoints of transmission line sections)

## Key Considerations

### Working with PMU Data

- **Large files**: Each PMU CSV is ~460-480MB. Use chunked reading strategies (pandas `chunksize`, generators) to avoid memory issues
- **Time alignment**: Events in topology.csv may not align exactly with UTC timestamps in PMU data due to sampling intervals
- **Missing data**: Check for gaps in time series or status flags indicating data quality issues
- **Three-phase systems**: A, B, C phases should be analyzed together for balanced system analysis

### Event Analysis

- Most events are lightning-related transmission line faults
- Events occur simultaneously across multiple terminals on the same transmission line
- Event time in topology.csv is approximate; actual fault signature may appear slightly before/after in PMU data
- Voltage levels (69kV, 138kV, 161kV) indicate transmission system hierarchy

### Spatial Analysis

- GPS coordinates enable geospatial event mapping
- Lightning events cluster geographically during storm systems
- Terminal proximity can be calculated using Latitude/Longitude fields
