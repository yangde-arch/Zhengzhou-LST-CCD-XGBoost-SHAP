# Data description

## Public sample data

The `sample_data/` folder contains reduced sample CSV files for demonstrating the code workflow and required input structure. These public CSV files currently contain a subset of the 2023 data and are intended for code testing and workflow demonstration only.

The sample files are **not** the full processed multi-year dataset used to generate all numerical results reported in the manuscript.

Exact reproduction of the manuscript's full numerical results requires the complete processed multi-year dataset, which is available from the corresponding author upon reasonable request because several original remote-sensing, socioeconomic, and meteorological datasets are subject to third-party data-use policies.

## Files in `sample_data/`

### `01_main_xgboost_shap_analysis.csv`

This file is used by:

```bash
python scripts/01_main_xgboost_shap_analysis.py
