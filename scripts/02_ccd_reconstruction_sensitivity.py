# -*- coding: utf-8 -*-
"""CCD reconstruction sensitivity analysis for Zhengzhou LST XGBoost-SHAP study.

Purpose
-------
This script evaluates whether reconstructing the original six CCD levels
(Level 1--Level 6) into three adjacent gradients (Level 1-2, Level 3-4,
Level 5-6) materially changes the XGBoost-SHAP attribution structure.

The workflow is designed for manuscript-level reproducibility and reviewer inspection:

1. Train XGBoost models for each original CCD level and each reconstructed
   CCD group by year.
2. Repeat train/test splitting with inner cross-validated hyperparameter
   tuning. This is a repeated holdout evaluation with inner CV tuning, not
   a full nested cross-validation claim.
3. Compute model performance (R2, RMSE, MAE) and SHAP feature importance.
4. Compare merged groups with weighted original sub-levels using:
   - Top-5 feature overlap
   - Top-5 Jaccard similarity
   - Spearman rank correlation
   - Kendall rank correlation
5. Export clean CSV and Excel tables suitable for supplementary materials.

Recommended repository layout
-----------------------------

    Zhengzhou-LST-CCD-XGBoost-SHAP/
    ├── scripts/
    │   └── 02_ccd_reconstruction_sensitivity.py
    ├── sample_data/
    │   └── 02_ccd_reconstruction_sensitivity.csv
    └── example_outputs/

Default execution
-----------------

    python scripts/02_ccd_reconstruction_sensitivity.py

For a fast smoke test:

    python scripts/02_ccd_reconstruction_sensitivity.py --quick

The default input path is resolved relative to the repository root, so the
script can be run directly after placing the CSV in ``sample_data/``.

If the public CSV is a reduced sample, the script will still run and will export
a data-coverage audit. Exact manuscript-number reproduction requires the full
processed multi-year dataset with original CCD levels 1--6.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Repository paths
# =============================================================================


def get_repository_root() -> Path:
    """Return project root when this file is stored in ``scripts/``."""
    script_path = Path(__file__).resolve()
    if script_path.parent.name.lower() == "scripts":
        return script_path.parent.parent
    return script_path.parent


REPO_ROOT = get_repository_root()
DEFAULT_INPUT_PATH = REPO_ROOT / "sample_data" / "02_ccd_reconstruction_sensitivity.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "example_outputs" / "02_ccd_reconstruction_sensitivity_results"


# =============================================================================
# Configuration
# =============================================================================


FEATURE_COLS = [
    "PD", "ED", "LSI", "AWMSI", "AI", "CONTAG", "SHDI",
    "FP", "GP", "WP", "FVC", "ECV",
    "BP", "POP", "GDP", "UEI",
]

CCD_GROUPS = {
    "Level 1-2": [1, 2],
    "Level 3-4": [3, 4],
    "Level 5-6": [5, 6],
}

EXPECTED_LEVELS = [1, 2, 3, 4, 5, 6]
DEFAULT_EXPECTED_YEARS = [2003, 2008, 2013, 2018, 2023]
OPTIONAL_IDENTIFIER_COLUMNS = ["LID", "lon", "lat", "Longitude", "Latitude", "X", "Y"]
DEFAULT_N_REPEATS = 20
DEFAULT_MIN_N_FOR_MODEL = 50
DEFAULT_LOW_N_WARNING = 100
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RANDOM_SEED_BASE = 2026
DEFAULT_SHAP_MAX_SAMPLES = 2500


@dataclass
class ModelRunResult:
    importance_avg: Optional[pd.DataFrame]
    seed_importances: Dict[int, pd.DataFrame]
    params_rows: List[dict]
    performance_row: dict


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CCD reconstruction sensitivity analysis using repeated "
            "XGBoost-SHAP attribution comparisons."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV path. Default: sample_data/02_ccd_reconstruction_sensitivity.csv",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for all result tables.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Number of repeated train/test splits. Default: 20.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=DEFAULT_MIN_N_FOR_MODEL,
        help="Minimum sample size required to train a level-specific model.",
    )
    parser.add_argument(
        "--low-n-warning",
        type=int,
        default=DEFAULT_LOW_N_WARNING,
        help="Sample-size threshold used to flag unstable original sub-levels.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test-set fraction for repeated holdout evaluation.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_RANDOM_SEED_BASE,
        help="Base random seed. Seeds are seed_base, seed_base+1, ...",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for XGBoost/GridSearchCV. Default: 1 for stability.",
    )
    parser.add_argument(
        "--tuning",
        choices=["grid", "light", "fixed"],
        default="grid",
        help=(
            "Hyperparameter strategy. 'grid' matches manuscript-level tuning; "
            "'light' is faster; 'fixed' uses one predefined parameter set."
        ),
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=DEFAULT_SHAP_MAX_SAMPLES,
        help="Maximum test samples used to compute SHAP per repeat. 0 means no sampling.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Fast smoke-test mode: n_repeats=3, light tuning, and SHAP sample cap=800. "
            "Use the default mode for manuscript-level results."
        ),
    )
    parser.add_argument(
        "--allow-merged-level-labels",
        action="store_true",
        help=(
            "Allow range-like CCD labels such as 'Level 1-2'. This is not recommended "
            "for the CCD reconstruction sensitivity analysis because the script is "
            "intended to compare original raw levels 1--6 against reconstructed groups."
        ),
    )
    parser.add_argument(
        "--strict-full-data",
        action="store_true",
        help=(
            "Require all expected manuscript years and CCD levels to be present. "
            "Leave disabled for public sample-data testing."
        ),
    )
    return parser.parse_args()


# =============================================================================
# Robust input handling
# =============================================================================


def read_csv_smart(path: Path | str) -> pd.DataFrame:
    """Read CSV input robustly and detect Excel files renamed as CSV.

    Recommended input format for GitHub reproducibility:
        CSV UTF-8 (Comma delimited) (*.csv)
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path_obj}")

    with open(path_obj, "rb") as f:
        magic = f.read(8)

    excel_like = (
        magic.startswith(b"\xD0\xCF\x11\xE0")
        or magic.startswith(b"PK\x03\x04")
        or path_obj.suffix.lower() in [".xls", ".xlsx", ".xlsm"]
    )

    if excel_like:
        try:
            df = pd.read_excel(path_obj)
            print(f"Excel workbook loaded: {path_obj}")
            print(
                "Note: For GitHub reproducibility, consider saving this input as "
                "CSV UTF-8 (Comma delimited) (*.csv)."
            )
            return df
        except Exception as exc:
            raise ValueError(
                "The input appears to be an Excel workbook rather than a plain-text "
                "CSV file. Please save it as CSV UTF-8 (Comma delimited) (*.csv), "
                "or keep the correct Excel suffix and install the required reader. "
                f"Original error: {exc}"
            ) from exc

    encodings = [
        "utf-8-sig", "utf-8", "gb18030", "gbk", "cp936",
        "utf-16", "utf-16le", "utf-16be", "latin1",
    ]

    errors = []
    for enc in encodings:
        try:
            df = pd.read_csv(path_obj, encoding=enc, low_memory=False)
            if df.shape[1] == 1:
                try:
                    df_auto = pd.read_csv(
                        path_obj,
                        encoding=enc,
                        sep=None,
                        engine="python",
                    )
                    if df_auto.shape[1] > df.shape[1]:
                        print(f"CSV loaded with delimiter auto-detection: {path_obj} ({enc})")
                        return df_auto
                except Exception:
                    pass
            print(f"CSV loaded: {path_obj} (encoding: {enc})")
            return df
        except Exception as exc:
            errors.append(f"{enc}: {exc}")

    raise UnicodeError(
        "Unable to decode the input file as CSV. Please save it as "
        "CSV UTF-8 (Comma delimited) (*.csv). Tried encodings: "
        + "; ".join(errors)
    )


def clean_column_name(col: object) -> str:
    col = str(col).replace("\ufeff", "").strip()
    col = col.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    col = re.sub(r"\s+", " ", col)
    return col


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common column-name variants used across user exports."""
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    rename_map = {
        "year": "Year",
        "YEAR": "Year",
        "Temperature": "Temperature",
        "Temp": "Temperature",
        "LST": "Temperature",
        "lst": "Temperature",
        "temperature": "Temperature",
        "Coupling Coordination Level": "CCD_Level",
        "Coupling_Coordination_Level": "CCD_Level",
        "CouplingCoordinationLevel": "CCD_Level",
        "ccd_level": "CCD_Level",
        "CCD": "CCD_Level",
        "Level": "CCD_Level",
        "Longitude": "lon",
        "Latitude": "lat",
        "Lon": "lon",
        "Lat": "lat",
        "X": "lon",
        "Y": "lat",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    compact_map = {}
    for c in df.columns:
        compact = re.sub(r"[\s_]+", "", str(c)).lower()
        if compact == "couplingcoordinationlevel":
            compact_map[c] = "CCD_Level"
        elif compact in ["temperature", "lst"]:
            compact_map[c] = "Temperature"
        elif compact == "year":
            compact_map[c] = "Year"
    if compact_map:
        df.rename(columns=compact_map, inplace=True)

    return df


def parse_ccd_level(value: object) -> float:
    """Parse CCD level as a numeric original level.

    The sensitivity analysis expects original CCD levels 1--6. If a merged label
    such as 'Level 1-2' is provided, this function returns the first number so
    the row is not silently dropped; however, a warning is issued later because
    merged labels are not ideal for this analysis.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isfinite(value):
            return int(round(float(value)))
        return np.nan
    numbers = re.findall(r"\d+", str(value).strip())
    if not numbers:
        return np.nan
    return int(numbers[0])


def detect_merged_level_labels(series: pd.Series) -> bool:
    """Detect range-like merged labels in the original CCD field."""
    sample = series.dropna().astype(str).head(5000)
    pattern = re.compile(r"\d+\s*[-–—_]\s*\d+")
    return any(bool(pattern.search(x)) for x in sample)


def prepare_data(df: pd.DataFrame, allow_merged_level_labels: bool = False) -> pd.DataFrame:
    """Clean and validate model-ready dataframe."""
    df = standardize_columns(df)

    required = ["Year", "Temperature", "CCD_Level"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nExpected columns include Year, Temperature/LST, "
            "Coupling Coordination Level, and all landscape/urbanization predictors."
        )

    if detect_merged_level_labels(df["CCD_Level"]):
        msg = (
            "The CCD level field appears to contain merged labels such as "
            "'Level 1-2'. This sensitivity analysis must use the original raw "
            "single CCD levels 1--6 so that the script can compare original "
            "sub-levels with reconstructed groups. Please replace merged labels "
            "with raw levels, or rerun with --allow-merged-level-labels only for "
            "a non-manuscript smoke test."
        )
        if not allow_merged_level_labels:
            raise ValueError(msg)
        print("WARNING: " + msg)

    for col in ["Year", "Temperature"] + FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["CCD_level_numeric"] = df["CCD_Level"].apply(parse_ccd_level)
    df["CCD_level_numeric"] = pd.to_numeric(df["CCD_level_numeric"], errors="coerce")

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Year", "Temperature", "CCD_level_numeric"] + FEATURE_COLS).copy()
    after = len(df)

    df["Year"] = df["Year"].round().astype(int)
    df["CCD_level_numeric"] = df["CCD_level_numeric"].round().astype(int)

    invalid_levels = sorted(set(df["CCD_level_numeric"].unique()) - set(EXPECTED_LEVELS))
    if invalid_levels:
        print(
            "WARNING: Unexpected CCD levels were found and will be excluded: "
            f"{invalid_levels}. Expected levels are 1--6."
        )
        df = df[df["CCD_level_numeric"].isin(EXPECTED_LEVELS)].copy()

    print(f"Rows before cleaning: {before:,}")
    print(f"Rows after cleaning : {len(df):,} (dropped {before - after:,} rows before level filtering)")

    if df.empty:
        raise ValueError("No valid rows remain after data cleaning.")

    return df



# =============================================================================
# Public-data and coverage audit
# =============================================================================


def build_input_audit(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize data coverage and public-release checks.

    This audit is especially useful when the public repository contains a reduced
    sample dataset rather than the full manuscript dataset.
    """
    rows = []

    raw_rows = int(len(raw_df))
    valid_rows = int(len(cleaned_df))
    mostly_empty_rows = int(raw_df.isna().all(axis=1).sum())
    optional_present = [c for c in OPTIONAL_IDENTIFIER_COLUMNS if c in raw_df.columns]

    years = sorted(cleaned_df["Year"].dropna().unique().astype(int).tolist())
    levels = sorted(cleaned_df["CCD_level_numeric"].dropna().unique().astype(int).tolist())

    rows.extend([
        {
            "Check": "raw_row_count",
            "Value": raw_rows,
            "Status": "INFO",
            "Recommendation": "Rows before model-data cleaning.",
        },
        {
            "Check": "valid_row_count_after_cleaning",
            "Value": valid_rows,
            "Status": "INFO",
            "Recommendation": "Rows used after removing blank or invalid records.",
        },
        {
            "Check": "fully_empty_row_count",
            "Value": mostly_empty_rows,
            "Status": "WARN" if mostly_empty_rows > 0 else "OK",
            "Recommendation": (
                "Remove trailing blank rows from the public CSV for a cleaner GitHub repository."
                if mostly_empty_rows > 0 else "No fully empty rows detected."
            ),
        },
        {
            "Check": "years_detected",
            "Value": ",".join(map(str, years)),
            "Status": "WARN" if len(years) < len(DEFAULT_EXPECTED_YEARS) else "OK",
            "Recommendation": (
                "Public sample data may contain fewer years. Full manuscript-number "
                "reproduction requires the full processed multi-year dataset."
                if len(years) < len(DEFAULT_EXPECTED_YEARS) else "All expected manuscript years appear to be present."
            ),
        },
        {
            "Check": "ccd_levels_detected",
            "Value": ",".join(map(str, levels)),
            "Status": "WARN" if set(levels) != set(EXPECTED_LEVELS) else "OK",
            "Recommendation": (
                "Some original CCD levels are absent in this public sample. This is acceptable "
                "for code testing but should be documented if exact manuscript reproduction is expected."
                if set(levels) != set(EXPECTED_LEVELS) else "All original CCD levels 1--6 are present."
            ),
        },
        {
            "Check": "optional_identifier_columns_present",
            "Value": ",".join(optional_present) if optional_present else "None",
            "Status": "WARN" if optional_present else "OK",
            "Recommendation": (
                "These columns are not required by this script. For a public repository, consider "
                "removing LID/lon/lat or other spatial identifiers unless you intentionally want "
                "to release them."
                if optional_present else "No optional spatial or row identifiers detected."
            ),
        },
    ])

    for year in years:
        sub = cleaned_df[cleaned_df["Year"] == year]
        present_levels = sorted(sub["CCD_level_numeric"].dropna().unique().astype(int).tolist())
        missing_levels = [lvl for lvl in EXPECTED_LEVELS if lvl not in present_levels]
        rows.append({
            "Check": f"missing_levels_in_{year}",
            "Value": ",".join(map(str, missing_levels)) if missing_levels else "None",
            "Status": "WARN" if missing_levels else "OK",
            "Recommendation": (
                "Missing original CCD levels will be skipped and flagged as not independently estimable."
                if missing_levels else "No missing original CCD level in this year."
            ),
        })

    return pd.DataFrame(rows)


def validate_full_data_scope(df: pd.DataFrame, strict: bool = False) -> None:
    """Warn or fail when a reduced sample is used instead of full manuscript data."""
    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    missing_years = [y for y in DEFAULT_EXPECTED_YEARS if y not in years]

    missing_by_year = {}
    for year in years:
        levels = set(df.loc[df["Year"] == year, "CCD_level_numeric"].dropna().astype(int).tolist())
        missing = [lvl for lvl in EXPECTED_LEVELS if lvl not in levels]
        if missing:
            missing_by_year[year] = missing

    messages = []
    if missing_years:
        messages.append(f"Missing expected manuscript years: {missing_years}")
    if missing_by_year:
        messages.append(f"Missing original CCD levels by year: {missing_by_year}")

    if messages:
        msg = (
            "Dataset coverage note: "
            + " | ".join(messages)
            + ". This can be acceptable for a public sample dataset, but exact manuscript-number "
              "reproduction requires the full processed multi-year dataset."
        )
        if strict:
            raise ValueError(msg)
        print("WARNING: " + msg)



# =============================================================================
# Modeling helpers
# =============================================================================


def get_param_grid(strategy: str) -> List[dict] | dict:
    """Return hyperparameter grid for the selected tuning strategy."""
    if strategy == "fixed":
        return {
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

    if strategy == "light":
        return {
            "max_depth": [3, 4],
            "learning_rate": [0.05],
            "n_estimators": [300, 400],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "reg_alpha": [0.1],
            "reg_lambda": [1.0],
        }

    # Manuscript-level grid. Still intentionally compact to avoid memory errors.
    return {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05],
        "n_estimators": [300, 500],
        "subsample": [0.7],
        "colsample_bytree": [0.7],
        "reg_alpha": [0.1],
        "reg_lambda": [1.0],
    }


def fit_xgb_with_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    tuning: str,
    n_jobs: int,
) -> Tuple[XGBRegressor, dict]:
    """Fit XGBoost with fixed, light, or grid hyperparameter strategy."""
    base_params = {
        "objective": "reg:squarederror",
        "random_state": seed,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }

    if tuning == "fixed":
        best_params = get_param_grid("fixed")
        model = XGBRegressor(**base_params, **best_params)
        model.fit(X_train, y_train)
        return model, dict(best_params)

    param_grid = get_param_grid(tuning)
    cv_folds = min(3, max(2, len(X_train) // 20))
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    model = XGBRegressor(**base_params)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        verbose=0,
        n_jobs=1,  # Keep the outer workflow stable on ordinary computers.
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = dict(grid_search.best_params_)
    return best_model, best_params


def compute_model_and_shap_repeated(
    data: pd.DataFrame,
    model_type: str,
    year: int,
    group_label: str,
    n_repeats: int,
    min_n_for_model: int,
    test_size: float,
    seed_base: int,
    tuning: str,
    n_jobs: int,
    shap_max_samples: int,
) -> ModelRunResult:
    """Run repeated XGBoost-SHAP evaluation for one year-level subset."""
    n = len(data)
    base_perf = {
        "model_type": model_type,
        "Year": year,
        "CCD_group": group_label,
        "n": n,
    }

    if n < min_n_for_model:
        perf_row = {
            **base_perf,
            "Successful_Repeats": 0,
            "R2_Mean": np.nan,
            "R2_SD": np.nan,
            "MAE_Mean": np.nan,
            "MAE_SD": np.nan,
            "RMSE_Mean": np.nan,
            "RMSE_SD": np.nan,
            "status": f"Skipped: n < {min_n_for_model}",
        }
        return ModelRunResult(None, {}, [], perf_row)

    X = data[FEATURE_COLS]
    y = data["Temperature"]

    r2_list, mae_list, rmse_list = [], [], []
    params_rows: List[dict] = []
    seed_importances: Dict[int, pd.DataFrame] = {}
    mean_abs_arrays: List[np.ndarray] = []
    failed_seeds: List[int] = []

    for repeat_idx in range(n_repeats):
        seed = seed_base + repeat_idx
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=seed,
            )

            model, best_params = fit_xgb_with_tuning(
                X_train=X_train,
                y_train=y_train,
                seed=seed,
                tuning=tuning,
                n_jobs=n_jobs,
            )

            y_pred = model.predict(X_test)
            r2_list.append(r2_score(y_test, y_pred))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            rmse_list.append(float(np.sqrt(mean_squared_error(y_test, y_pred))))

            params_rows.append({
                "model_type": model_type,
                "Year": year,
                "CCD_group": group_label,
                "Repeat": repeat_idx + 1,
                "Seed": seed,
                **best_params,
            })

            X_shap = X_test
            if shap_max_samples and shap_max_samples > 0 and len(X_shap) > shap_max_samples:
                X_shap = X_shap.sample(n=shap_max_samples, random_state=seed)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):
                shap_arr = shap_values[0]
            elif hasattr(shap_values, "values"):
                shap_arr = shap_values.values
            else:
                shap_arr = shap_values

            mean_abs = np.abs(np.asarray(shap_arr)).mean(axis=0)
            total = float(mean_abs.sum())
            norm_importance = mean_abs / total if total > 0 else np.zeros_like(mean_abs)

            seed_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "MeanAbsSHAP": mean_abs,
                "NormImportance": norm_importance,
            })
            seed_df["Rank"] = seed_df["NormImportance"].rank(
                ascending=False,
                method="average",
            )
            seed_df["Seed"] = seed
            seed_df["Repeat"] = repeat_idx + 1
            seed_importances[seed] = seed_df
            mean_abs_arrays.append(mean_abs)

        except Exception as exc:
            failed_seeds.append(seed)
            print(f"  WARNING: {model_type} {year} {group_label} seed={seed} failed: {exc}")
            continue

    successful = len(r2_list)
    if successful == 0:
        perf_row = {
            **base_perf,
            "Successful_Repeats": 0,
            "R2_Mean": np.nan,
            "R2_SD": np.nan,
            "MAE_Mean": np.nan,
            "MAE_SD": np.nan,
            "RMSE_Mean": np.nan,
            "RMSE_SD": np.nan,
            "status": "Failed: no successful repeats",
        }
        return ModelRunResult(None, seed_importances, params_rows, perf_row)

    status = "OK" if successful == n_repeats else f"Partial: {successful}/{n_repeats} repeats succeeded"
    if failed_seeds:
        status += f"; failed seeds={failed_seeds}"

    perf_row = {
        **base_perf,
        "Successful_Repeats": successful,
        "R2_Mean": float(np.mean(r2_list)),
        "R2_SD": float(np.std(r2_list, ddof=0)),
        "MAE_Mean": float(np.mean(mae_list)),
        "MAE_SD": float(np.std(mae_list, ddof=0)),
        "RMSE_Mean": float(np.mean(rmse_list)),
        "RMSE_SD": float(np.std(rmse_list, ddof=0)),
        "status": status,
    }

    mean_abs_shap = np.mean(mean_abs_arrays, axis=0)
    total = float(mean_abs_shap.sum())
    norm_importance = mean_abs_shap / total if total > 0 else np.zeros_like(mean_abs_shap)

    importance_avg = pd.DataFrame({
        "model_type": model_type,
        "Year": year,
        "CCD_group": group_label,
        "Feature": FEATURE_COLS,
        "MeanAbsSHAP": mean_abs_shap,
        "NormImportance": norm_importance,
    })
    importance_avg["Rank"] = importance_avg["NormImportance"].rank(
        ascending=False,
        method="average",
    )
    importance_avg = importance_avg.sort_values(["Year", "CCD_group", "Rank"])

    return ModelRunResult(importance_avg, seed_importances, params_rows, perf_row)


# =============================================================================
# Robustness metrics
# =============================================================================


def safe_spearman(rank_a: Sequence[float], rank_b: Sequence[float]) -> float:
    rho, _ = spearmanr(rank_a, rank_b)
    return float(rho) if np.isfinite(rho) else np.nan


def safe_kendall(rank_a: Sequence[float], rank_b: Sequence[float]) -> float:
    tau, _ = kendalltau(rank_a, rank_b)
    return float(tau) if np.isfinite(tau) else np.nan


def topk_overlap_and_jaccard(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int = 5) -> Tuple[int, float]:
    top_a = set(df_a.sort_values("Rank").head(k)["Feature"])
    top_b = set(df_b.sort_values("Rank").head(k)["Feature"])
    inter = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))
    jaccard = inter / union if union else np.nan
    return inter, float(jaccard)


def classify_stability(
    overlap_mean: float,
    jaccard_mean: float,
    rho_mean: float,
    tau_mean: float,
    skipped_levels: Optional[List[int]] = None,
    min_sublevel_n: Optional[int] = None,
    low_n_warning: int = DEFAULT_LOW_N_WARNING,
) -> str:
    """Classify attribution stability using transparent rule-based criteria."""
    skipped_levels = skipped_levels or []

    if skipped_levels:
        conclusion = (
            "Merging necessary: at least one original sub-level was not independently "
            "estimable"
        )
    elif (
        np.isfinite(overlap_mean) and np.isfinite(rho_mean) and np.isfinite(tau_mean)
        and overlap_mean >= 4 and rho_mean >= 0.70 and tau_mean >= 0.50
    ):
        conclusion = "Highly stable"
    elif (
        np.isfinite(overlap_mean) and np.isfinite(rho_mean)
        and overlap_mean >= 3 and rho_mean >= 0.60
    ):
        conclusion = "Stable"
    elif (
        (np.isfinite(overlap_mean) and overlap_mean >= 3)
        or (np.isfinite(rho_mean) and rho_mean >= 0.40)
        or (np.isfinite(jaccard_mean) and jaccard_mean >= 0.40)
    ):
        conclusion = "Partly stable"
    else:
        conclusion = "Unstable"

    if min_sublevel_n is not None and 0 < min_sublevel_n < low_n_warning:
        conclusion += f" | Low-N caution: minimum original sub-level n < {low_n_warning}"

    return conclusion


def build_weighted_original_importance(
    seed: int,
    year: int,
    levels: Sequence[int],
    seed_imp_dict: Dict[Tuple[int, str], Dict[int, pd.DataFrame]],
    n_dict: Dict[Tuple[int, str], int],
) -> Tuple[Optional[pd.DataFrame], List[int], List[int]]:
    """Build sample-size-weighted original-level importance for one seed."""
    weighted = np.zeros(len(FEATURE_COLS), dtype=float)
    total_n = 0
    used_levels: List[int] = []
    skipped_levels: List[int] = []

    for lvl in levels:
        label = f"Level {lvl}"
        key = (year, label)
        if key in seed_imp_dict and seed in seed_imp_dict[key]:
            sub_df = seed_imp_dict[key][seed]
            sub_df = sub_df.set_index("Feature").loc[FEATURE_COLS]
            n = n_dict.get(key, 0)
            weighted += sub_df["NormImportance"].values * n
            total_n += n
            used_levels.append(lvl)
        else:
            skipped_levels.append(lvl)

    if total_n <= 0:
        return None, used_levels, skipped_levels

    weighted /= total_n
    w_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "NormImportance": weighted,
    })
    w_df["Rank"] = w_df["NormImportance"].rank(
        ascending=False,
        method="average",
    )
    return w_df, used_levels, skipped_levels


def compare_rank_tables(df_a: pd.DataFrame, df_b: pd.DataFrame, top_k: int = 5) -> dict:
    """Compare two feature-importance rank tables."""
    a = df_a.set_index("Feature").loc[FEATURE_COLS]
    b = df_b.set_index("Feature").loc[FEATURE_COLS]

    overlap, jaccard = topk_overlap_and_jaccard(a.reset_index(), b.reset_index(), k=top_k)
    return {
        "Top5_Overlap": overlap,
        "Top5_Jaccard": jaccard,
        "Spearman_Rho": safe_spearman(a["Rank"], b["Rank"]),
        "Kendall_Tau": safe_kendall(a["Rank"], b["Rank"]),
    }


def summarize_metric(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def create_robustness_tables(
    years: Sequence[int],
    seed_imp_dict: Dict[Tuple[int, str], Dict[int, pd.DataFrame]],
    n_dict: Dict[Tuple[int, str], int],
    n_repeats: int,
    seed_base: int,
    low_n_warning: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create weighted and individual robustness-comparison tables."""
    weighted_rows = []
    individual_rows = []
    seeds = [seed_base + i for i in range(n_repeats)]

    for year in years:
        for group, levels in CCD_GROUPS.items():
            merged_key = (year, group)
            if merged_key not in seed_imp_dict:
                continue

            weighted_metrics = {
                "Top5_Overlap": [],
                "Top5_Jaccard": [],
                "Spearman_Rho": [],
                "Kendall_Tau": [],
            }
            used_levels_all: List[int] = []
            skipped_levels_all: List[int] = []

            for seed in seeds:
                if seed not in seed_imp_dict[merged_key]:
                    continue

                merged_df = seed_imp_dict[merged_key][seed]
                weighted_df, used_levels, skipped_levels = build_weighted_original_importance(
                    seed=seed,
                    year=year,
                    levels=levels,
                    seed_imp_dict=seed_imp_dict,
                    n_dict=n_dict,
                )
                used_levels_all.extend(used_levels)
                skipped_levels_all.extend(skipped_levels)

                if weighted_df is None:
                    continue

                metrics = compare_rank_tables(merged_df, weighted_df, top_k=5)
                for k, v in metrics.items():
                    weighted_metrics[k].append(v)

            if weighted_metrics["Top5_Overlap"]:
                used_unique = sorted(set(used_levels_all))
                skipped_unique = sorted(set(skipped_levels_all))
                min_sublevel_n = (
                    min([n_dict[(year, f"Level {lvl}")] for lvl in used_unique])
                    if used_unique else np.nan
                )

                overlap_mean, overlap_sd = summarize_metric(weighted_metrics["Top5_Overlap"])
                jaccard_mean, jaccard_sd = summarize_metric(weighted_metrics["Top5_Jaccard"])
                rho_mean, rho_sd = summarize_metric(weighted_metrics["Spearman_Rho"])
                tau_mean, tau_sd = summarize_metric(weighted_metrics["Kendall_Tau"])

                weighted_rows.append({
                    "Year": year,
                    "Merged_Group": group,
                    "Comparison": "Merged vs sample-size-weighted original sub-levels",
                    "Used_Levels": ",".join(map(str, used_unique)),
                    "Skipped_Levels": ",".join(map(str, skipped_unique)),
                    "Min_SubLevel_N": min_sublevel_n,
                    "Successful_Seed_Comparisons": len(weighted_metrics["Top5_Overlap"]),
                    "Top5_Overlap_Mean": overlap_mean,
                    "Top5_Overlap_SD": overlap_sd,
                    "Top5_Jaccard_Mean": jaccard_mean,
                    "Top5_Jaccard_SD": jaccard_sd,
                    "Spearman_Rho_Mean": rho_mean,
                    "Spearman_Rho_SD": rho_sd,
                    "Kendall_Tau_Mean": tau_mean,
                    "Kendall_Tau_SD": tau_sd,
                    "Conclusion": classify_stability(
                        overlap_mean=overlap_mean,
                        jaccard_mean=jaccard_mean,
                        rho_mean=rho_mean,
                        tau_mean=tau_mean,
                        skipped_levels=skipped_unique,
                        min_sublevel_n=int(min_sublevel_n) if np.isfinite(min_sublevel_n) else None,
                        low_n_warning=low_n_warning,
                    ),
                })

            # Individual comparisons: merged group vs each original sub-level.
            for lvl in levels:
                original_key = (year, f"Level {lvl}")
                if original_key not in seed_imp_dict:
                    individual_rows.append({
                        "Year": year,
                        "Merged_Group": group,
                        "Target_Level": f"Level {lvl}",
                        "Target_N": n_dict.get(original_key, 0),
                        "Successful_Seed_Comparisons": 0,
                        "Top5_Overlap_Mean": np.nan,
                        "Top5_Overlap_SD": np.nan,
                        "Top5_Jaccard_Mean": np.nan,
                        "Top5_Jaccard_SD": np.nan,
                        "Spearman_Rho_Mean": np.nan,
                        "Spearman_Rho_SD": np.nan,
                        "Kendall_Tau_Mean": np.nan,
                        "Kendall_Tau_SD": np.nan,
                        "Conclusion": "Not independently estimable; merging necessary",
                    })
                    continue

                indiv_metrics = {
                    "Top5_Overlap": [],
                    "Top5_Jaccard": [],
                    "Spearman_Rho": [],
                    "Kendall_Tau": [],
                }

                for seed in seeds:
                    if seed not in seed_imp_dict[merged_key] or seed not in seed_imp_dict[original_key]:
                        continue
                    metrics = compare_rank_tables(
                        seed_imp_dict[merged_key][seed],
                        seed_imp_dict[original_key][seed],
                        top_k=5,
                    )
                    for k, v in metrics.items():
                        indiv_metrics[k].append(v)

                overlap_mean, overlap_sd = summarize_metric(indiv_metrics["Top5_Overlap"])
                jaccard_mean, jaccard_sd = summarize_metric(indiv_metrics["Top5_Jaccard"])
                rho_mean, rho_sd = summarize_metric(indiv_metrics["Spearman_Rho"])
                tau_mean, tau_sd = summarize_metric(indiv_metrics["Kendall_Tau"])
                target_n = n_dict.get(original_key, 0)

                individual_rows.append({
                    "Year": year,
                    "Merged_Group": group,
                    "Target_Level": f"Level {lvl}",
                    "Target_N": target_n,
                    "Successful_Seed_Comparisons": len(indiv_metrics["Top5_Overlap"]),
                    "Top5_Overlap_Mean": overlap_mean,
                    "Top5_Overlap_SD": overlap_sd,
                    "Top5_Jaccard_Mean": jaccard_mean,
                    "Top5_Jaccard_SD": jaccard_sd,
                    "Spearman_Rho_Mean": rho_mean,
                    "Spearman_Rho_SD": rho_sd,
                    "Kendall_Tau_Mean": tau_mean,
                    "Kendall_Tau_SD": tau_sd,
                    "Conclusion": classify_stability(
                        overlap_mean=overlap_mean,
                        jaccard_mean=jaccard_mean,
                        rho_mean=rho_mean,
                        tau_mean=tau_mean,
                        skipped_levels=[],
                        min_sublevel_n=target_n,
                        low_n_warning=low_n_warning,
                    ),
                })

    return pd.DataFrame(weighted_rows), pd.DataFrame(individual_rows)


# =============================================================================
# Data-quality diagnostics
# =============================================================================


def build_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact data-quality report for reviewer-facing reproducibility.

    The report is useful for checking whether each predictor has enough
    variation within each year and CCD level before model training.
    """
    rows = []

    def add_scope(scope: str, sub: pd.DataFrame, year: object = "ALL", ccd: object = "ALL") -> None:
        for feature in FEATURE_COLS:
            x = pd.to_numeric(sub[feature], errors="coerce")
            rows.append({
                "Scope": scope,
                "Year": year,
                "CCD_level": ccd,
                "Feature": feature,
                "n_rows": int(len(sub)),
                "n_valid": int(x.notna().sum()),
                "missing_rate": float(x.isna().mean()) if len(sub) else np.nan,
                "zero_rate": float((x == 0).mean()) if len(sub) else np.nan,
                "n_unique": int(x.nunique(dropna=True)),
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "std": float(x.std()) if x.notna().sum() > 1 else np.nan,
                "min": float(x.min()) if x.notna().any() else np.nan,
                "p25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
                "median": float(x.median()) if x.notna().any() else np.nan,
                "p75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
                "max": float(x.max()) if x.notna().any() else np.nan,
            })

    add_scope("All samples", df)
    for year, sub in df.groupby("Year", dropna=False):
        add_scope("By year", sub, year=year)
    for (year, level), sub in df.groupby(["Year", "CCD_level_numeric"], dropna=False):
        add_scope("By year and original CCD level", sub, year=year, ccd=level)

    return pd.DataFrame(rows)



# =============================================================================
# Output helpers
# =============================================================================


def build_sample_size_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_rows = []
    years = sorted(df["Year"].dropna().unique().astype(int).tolist())

    for year in years:
        df_year = df[df["Year"] == year]
        for lvl in EXPECTED_LEVELS:
            n = int((df_year["CCD_level_numeric"] == lvl).sum())
            sample_rows.append({
                "Year": year,
                "CCD_scheme": "Original six levels",
                "CCD_group": f"Level {lvl}",
                "n": n,
            })
        for group, levels in CCD_GROUPS.items():
            n = int(df_year["CCD_level_numeric"].isin(levels).sum())
            sample_rows.append({
                "Year": year,
                "CCD_scheme": "Reconstructed three gradients",
                "CCD_group": group,
                "n": n,
            })

    sample_size_df = pd.DataFrame(sample_rows)

    original = sample_size_df[sample_size_df["CCD_scheme"] == "Original six levels"]
    pivot = original.pivot(index="Year", columns="CCD_group", values="n")
    expected_cols = [f"Level {i}" for i in EXPECTED_LEVELS]
    pivot = pivot.reindex(columns=expected_cols).fillna(0).astype(int)
    pivot["Missing_Level_Count"] = (pivot[expected_cols] == 0).sum(axis=1)
    nonzero = pivot[expected_cols].replace(0, np.nan)
    pivot["Min_nonzero_n"] = nonzero.min(axis=1)
    pivot["Max_n"] = pivot[expected_cols].max(axis=1)
    pivot["Max_Min_Ratio"] = pivot["Max_n"] / pivot["Min_nonzero_n"]
    pivot = pivot.reset_index()

    return sample_size_df, pivot


def save_outputs(
    output_dir: Path,
    sample_size_df: pd.DataFrame,
    sample_pivot_df: pd.DataFrame,
    input_audit_df: pd.DataFrame,
    data_quality_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    params_df: pd.DataFrame,
    weighted_df: pd.DataFrame,
    individual_df: pd.DataFrame,
    unmerged_imp_df: pd.DataFrame,
    merged_imp_df: pd.DataFrame,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "Sample_Size_Raw": sample_size_df,
        "Sample_Size_Pivot": sample_pivot_df,
        "Input_Coverage_Audit": input_audit_df,
        "Data_Quality_Report": data_quality_df,
        "Model_Performance": perf_df,
        "All_BestParams": params_df,
        "Robustness_Weighted": weighted_df,
        "Robustness_Individual": individual_df,
        "SHAP_Unmerged_Avg": unmerged_imp_df,
        "SHAP_Merged_Avg": merged_imp_df,
    }

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    excel_path = output_dir / "SHAP_Robustness_Results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, table in tables.items():
            sheet_name = name[:31]
            table.to_excel(writer, sheet_name=sheet_name, index=False)

    with open(output_dir / "Run_Metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Excel workbook : {excel_path}")


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    start_time = time.time()
    args = parse_args()

    if args.quick:
        args.n_repeats = 3
        args.tuning = "light"
        args.shap_max_samples = 800
        print("Quick mode enabled: n_repeats=3, tuning='light', shap_max_samples=800")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1.")
    seeds = [args.seed_base + i for i in range(args.n_repeats)]

    print("=" * 80)
    print("CCD reconstruction sensitivity analysis for XGBoost-SHAP")
    print("=" * 80)
    print(f"Repository root : {REPO_ROOT}")
    print(f"Input file      : {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Repeats         : {args.n_repeats}")
    print(f"Tuning strategy : {args.tuning}")
    print(f"n_jobs          : {args.n_jobs}")
    print(f"Seeds           : {seeds[0]} to {seeds[-1] if seeds else seeds[0]}")
    print("=" * 80)

    raw_df = read_csv_smart(input_path)
    df = prepare_data(raw_df, allow_merged_level_labels=args.allow_merged_level_labels)
    validate_full_data_scope(df, strict=args.strict_full_data)

    years = sorted(df["Year"].dropna().unique().astype(int).tolist())
    print(f"Years detected  : {years}")
    print("CCD level counts:")
    print(df.groupby(["Year", "CCD_level_numeric"]).size().unstack(fill_value=0))

    sample_size_df, sample_pivot_df = build_sample_size_tables(df)
    input_audit_df = build_input_audit(raw_df, df)
    data_quality_df = build_data_quality_report(df)

    perf_rows: List[dict] = []
    params_rows: List[dict] = []
    unmerged_importance_avg: List[pd.DataFrame] = []
    merged_importance_avg: List[pd.DataFrame] = []
    seed_imp_dict: Dict[Tuple[int, str], Dict[int, pd.DataFrame]] = {}
    n_dict: Dict[Tuple[int, str], int] = {}

    # Original six-level models.
    for year in years:
        df_year = df[df["Year"] == year]
        for lvl in EXPECTED_LEVELS:
            subset = df_year[df_year["CCD_level_numeric"] == lvl].copy()
            label = f"Level {lvl}"
            print(f"\nTraining original model: Year={year}, {label}, n={len(subset):,}")
            result = compute_model_and_shap_repeated(
                data=subset,
                model_type="Original six levels",
                year=year,
                group_label=label,
                n_repeats=args.n_repeats,
                min_n_for_model=args.min_n,
                test_size=args.test_size,
                seed_base=args.seed_base,
                tuning=args.tuning,
                n_jobs=args.n_jobs,
                shap_max_samples=args.shap_max_samples,
            )
            perf_rows.append(result.performance_row)
            params_rows.extend(result.params_rows)
            if result.importance_avg is not None:
                unmerged_importance_avg.append(result.importance_avg)
                seed_imp_dict[(year, label)] = result.seed_importances
                n_dict[(year, label)] = len(subset)

    # Reconstructed three-gradient models.
    for year in years:
        df_year = df[df["Year"] == year]
        for group, levels in CCD_GROUPS.items():
            subset = df_year[df_year["CCD_level_numeric"].isin(levels)].copy()
            print(f"\nTraining reconstructed model: Year={year}, {group}, n={len(subset):,}")
            result = compute_model_and_shap_repeated(
                data=subset,
                model_type="Reconstructed three gradients",
                year=year,
                group_label=group,
                n_repeats=args.n_repeats,
                min_n_for_model=args.min_n,
                test_size=args.test_size,
                seed_base=args.seed_base,
                tuning=args.tuning,
                n_jobs=args.n_jobs,
                shap_max_samples=args.shap_max_samples,
            )
            perf_rows.append(result.performance_row)
            params_rows.extend(result.params_rows)
            if result.importance_avg is not None:
                merged_importance_avg.append(result.importance_avg)
                seed_imp_dict[(year, group)] = result.seed_importances
                n_dict[(year, group)] = len(subset)

    perf_df = pd.DataFrame(perf_rows)
    params_df = pd.DataFrame(params_rows)
    unmerged_imp_df = (
        pd.concat(unmerged_importance_avg, ignore_index=True)
        if unmerged_importance_avg else pd.DataFrame()
    )
    merged_imp_df = (
        pd.concat(merged_importance_avg, ignore_index=True)
        if merged_importance_avg else pd.DataFrame()
    )

    weighted_df, individual_df = create_robustness_tables(
        years=years,
        seed_imp_dict=seed_imp_dict,
        n_dict=n_dict,
        n_repeats=args.n_repeats,
        seed_base=args.seed_base,
        low_n_warning=args.low_n_warning,
    )

    elapsed = time.time() - start_time
    metadata = {
        "script": Path(__file__).name,
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "repository_root": str(REPO_ROOT),
        "rows_after_cleaning": int(len(df)),
        "allow_merged_level_labels": bool(args.allow_merged_level_labels),
        "strict_full_data": bool(args.strict_full_data),
        "expected_manuscript_years": DEFAULT_EXPECTED_YEARS,
        "years": years,
        "feature_columns": FEATURE_COLS,
        "ccd_groups": CCD_GROUPS,
        "n_repeats": int(args.n_repeats),
        "min_n_for_model": int(args.min_n),
        "low_n_warning": int(args.low_n_warning),
        "test_size": float(args.test_size),
        "seed_base": int(args.seed_base),
        "tuning": args.tuning,
        "n_jobs": int(args.n_jobs),
        "shap_max_samples": int(args.shap_max_samples),
        "elapsed_seconds": round(elapsed, 2),
        "python_version": sys.version,
        "note": (
            "This analysis compares reconstructed CCD groups with original "
            "sub-levels using repeated XGBoost-SHAP attribution rankings. "
            "SHAP results describe model attribution structure, not causal effects. The input should contain original raw CCD levels 1--6 for manuscript-level CCD reconstruction sensitivity analysis."
        ),
    }

    save_outputs(
        output_dir=output_dir,
        sample_size_df=sample_size_df,
        sample_pivot_df=sample_pivot_df,
        input_audit_df=input_audit_df,
        data_quality_df=data_quality_df,
        perf_df=perf_df,
        params_df=params_df,
        weighted_df=weighted_df,
        individual_df=individual_df,
        unmerged_imp_df=unmerged_imp_df,
        merged_imp_df=merged_imp_df,
        metadata=metadata,
    )

    print("\nAnalysis finished successfully.")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
