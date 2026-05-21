# Zhengzhou-LST-CCD-XGBoost-SHAP

This repository provides the Python code used for the machine-learning and explainable-AI analyses in the manuscript revision:

**Nonlinear Driving Mechanisms of Land Surface Temperature Under the Coupling of Blue-Green Space and Urbanization: An Analysis Based on Explainable Machine Learning**

## Overview

The repository contains Python scripts for:

1. XGBoost model training, model comparison, and SHAP-based interpretation;
2. SHAP response breakpoint validation using segmented regression and bootstrap confidence intervals;
3. Sensitivity analysis of CCD classification reconstruction;
4. Meteorological-control sensitivity analysis.

Remote-sensing preprocessing and landscape metric calculation were conducted outside this repository using GIS and landscape-analysis software. The AI-related analyses were conducted in Python.

## Repository structure

```text
Zhengzhou-LST-CCD-XGBoost-SHAP/
│
├── README.md
├── LICENSE
├── requirements.txt
├── data_description.md
├── .gitignore
│
├── scripts/
│   ├── 01_main_xgboost_shap_analysis.py
│   ├── 02_ccd_reconstruction_sensitivity.py
│   └── 03_meteorological_control_sensitivity.py
│
├── sample_data/
│   ├── 01_main_xgboost_shap_analysis.csv
│   ├── 02_ccd_reconstruction_sensitivity.csv
│   └── 03_meteorological_control_sensitivity.csv
│
├── example_outputs/
│   └── 01_main_xgboost_shap_analysis_results/
│
└── archive/
    └── README.md
