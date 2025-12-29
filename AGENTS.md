# AGENTS.md

## Commands
- **Install**: `pip install -r requirements.txt`
- **Run all tests**: `pytest tests/`
- **Run single test file**: `pytest tests/test_<module>.py`
- **Run single test**: `pytest tests/test_<module>.py::test_function_name -v`
- **Pilot scripts**: `python run_pilot_<step>.py` (extract, detection, analysis, validation, report)

## Architecture
- `src/` - Core modules: data_loader.py, preprocessing.py, topology.py, dynamic_models.py
- `src/metrics/` - Signal analysis metrics
- `src/validation/` - Data validation logic
- `src/visualization/` - Plotting utilities
- `src/reporting/` - Report generation
- `config/pilot_config.yaml` - Configuration settings
- `data/raw_pmu/` - Large CSV files (~460MB each), use chunked reading with pandas

## Code Style
- Python 3.x with type hints encouraged
- Use pandas for data manipulation with chunked reading for large files
- NumPy/SciPy for numerical operations
- Follow existing module patterns in `src/`
- Tests mirror source structure: `src/foo.py` → `tests/test_foo.py`

See CLAUDE.md for detailed PMU data format and domain context.
