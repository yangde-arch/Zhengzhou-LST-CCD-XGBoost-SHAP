# -*- coding: utf-8 -*-
"""Meteorological-control sensitivity analysis for Zhengzhou LST XGBoost-SHAP study.

Purpose
-------
This script evaluates whether the main landscape/urbanization attribution
patterns remain stable after adding short-term meteorological control variables.

The workflow compares three model scenarios:

    M0: landscape + urbanization predictors
    M1: M0 + WS + Tair + SRAD + RH + PRE
    M2: M0 + WS + Tair + SRAD + RH, excluding PRE

M2 is included because precipitation may be zero or nearly invariant on some
satellite acquisition dates, which can reduce its explanatory value. If PRE has
no effective variation within a subset, M1 and M2 may produce identical results;
the summary diagnostics explicitly records this situation.

The script is designed for manuscript-level reproducibility and reviewer
inspection. It uses repository-relative paths by default and exports clean CSV,
Excel, and JSON metadata outputs.

Recommended repository layout
-----------------------------

    Zhengzhou-LST-CCD-XGBoost-SHAP/
    ├── scripts/
    │   └── 03_meteorological_control_sensitivity.py
    ├── sample_data/
    │   └── 03_meteorological_control_sensitivity.csv
    └── example_outputs/

Default execution
-----------------

Run all model scenarios and then generate summary tables:

    python scripts/03_meteorological_control_sensitivity.py

Run a single stage for lower-memory machines:

    python scripts/03_meteorological_control_sensitivity.py --stage M0
    python scripts/03_meteorological_control_sensitivity.py --stage M1
    python scripts/03_meteorological_control_sensitivity.py --stage M2
    python scripts/03_meteorological_control_sensitivity.py --stage SUMMARY

Fast smoke test:

    python scripts/03_meteorological_control_sensitivity.py --quick

Important interpretation note
-----------------------------
This analysis is a sensitivity test, not a causal analysis. SHAP values describe
model attribution structure under different predictor sets. They should not be
interpreted as causal effects of meteorological variables on LST.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

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
DEFAULT_INPUT_PATH = REPO_ROOT / "sample_data" / "03_meteorological_control_sensitivity.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "example_outputs" / "03_meteorological_control_sensitivity_results"


# =============================================================================
# Model configuration
# =============================================================================


TARGET_COL = "LST"
YEAR_COL = "Year"
LEVEL_COL = "Coupling_Coordination_Level"

BASE_FEATURES = [
    "PD", "ED", "LSI", "AWMSI", "AI", "CONTAG", "SHDI",
    "FP", "GP", "WP", "FVC", "ECV",
    "BP", "POP", "GDP", "UEI",
]

CLIMATE_WITH_PRE = ["WS", "Tair", "SRAD", "RH", "PRE"]
CLIMATE_NO_PRE = ["WS", "Tair", "SRAD", "RH"]

SCENARIOS: Dict[str, Dict[str, object]] = {
    "M0": {
        "full_name": "M0_original_landscape_urban",
        "description": "Landscape and urbanization predictors only",
        "features": BASE_FEATURES,
    },
    "M1": {
        "full_name": "M1_control_WS_Tair_SRAD_RH_PRE",
        "description": "M0 plus wind speed, air temperature, shortwave radiation, humidity, and precipitation",
        "features": BASE_FEATURES + CLIMATE_WITH_PRE,
    },
    "M2": {
        "full_name": "M2_control_WS_Tair_SRAD_RH_no_PRE",
        "description": "M0 plus wind speed, air temperature, shortwave radiation, and humidity; precipitation excluded",
        "features": BASE_FEATURES + CLIMATE_NO_PRE,
    },
}

DEFAULT_MIN_SAMPLES = 80
DEFAULT_N_REPEATS = 20
DEFAULT_TEST_SIZE = 0.25
DEFAULT_SEED_BASE = 2026
DEFAULT_SHAP_SAMPLE_MAX = 1200

PARAM_CANDIDATES = [
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 500,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
]

FIXED_PARAMS = PARAM_CANDIDATES[0]


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run meteorological-control sensitivity analysis for Zhengzhou "
            "LST XGBoost-SHAP workflow."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV path. Default: sample_data/03_meteorological_control_sensitivity.csv",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for all result tables.",
    )
    parser.add_argument(
        "--stage",
        choices=["M0", "M1", "M2", "SUMMARY", "ALL"],
        default="ALL",
        help=(
            "Stage to run. Default ALL runs M0, M1, M2, then SUMMARY. "
            "Use individual stages for lower-memory machines."
        ),
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=DEFAULT_N_REPEATS,
        help="Number of repeated train/test splits. Default: 20.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum sample size for a subset-scenario model. Default: 80.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Test-set fraction. Default: 0.25.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
        help="Base random seed. Seeds are seed_base, seed_base+1, ...",
    )
    parser.add_argument(
        "--tuning",
        choices=["light", "fixed"],
        default="light",
        help="Sequential hyperparameter tuning strategy. Default: light.",
    )
    parser.add_argument(
        "--shap-sample-max",
        type=int,
        default=DEFAULT_SHAP_SAMPLE_MAX,
        help="Maximum test samples used for SHAP calculation. 0 means no sampling.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP importance calculation and only evaluate model performance.",
    )
    parser.add_argument(
        "--include-all-samples",
        action="store_true",
        help="Also run one pooled model across all years and CCD groups.",
    )
    parser.add_argument(
        "--no-year-ccd",
        action="store_true",
        help="Do not run year-by-CCD-group models.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of XGBoost jobs. Default 1 for stable, low-memory execution.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke-test mode: n_repeats=3, fixed tuning, SHAP cap=500.",
    )
    return parser.parse_args()


# =============================================================================
# Robust input handling
# =============================================================================


def read_csv_smart(path: Path | str) -> pd.DataFrame:
    """Read tabular input robustly and detect Excel files renamed as CSV.

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
                    df_auto = pd.read_csv(path_obj, encoding=enc, sep=None, engine="python")
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
        "Unable to decode the input file as CSV. Please save it as CSV UTF-8 "
        "(Comma delimited) (*.csv). Tried encodings: " + "; ".join(errors)
    )


def clean_column_name(col: object) -> str:
    col = str(col).replace("\ufeff", "").strip()
    col = col.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    col = re.sub(r"\s+", " ", col)
    return col


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common column-name variants across spreadsheet exports."""
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    rename_map = {
        "Temperature": "LST",
        "Temp": "LST",
        "temperature": "LST",
        "lst": "LST",
        "Land_Surface_Temperature": "LST",
        "year": "Year",
        "YEAR": "Year",
        "Coupling Coordination Level": "Coupling_Coordination_Level",
        "CouplingCoordinationLevel": "Coupling_Coordination_Level",
        "ccd_level": "Coupling_Coordination_Level",
        "CCD": "Coupling_Coordination_Level",
        "Level": "Coupling_Coordination_Level",
        "Wind": "WS",
        "wind": "WS",
        "Tair": "Tair",
        "TAIR": "Tair",
        "Air_Temperature": "Tair",
        "SRAD": "SRAD",
        "srad": "SRAD",
        "RH": "RH",
        "rhum": "RH",
        "RHum": "RH",
        "PRE": "PRE",
        "prec": "PRE",
        "Precipitation": "PRE",
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
        if compact in ["couplingcoordinationlevel", "ccdlevel"]:
            compact_map[c] = "Coupling_Coordination_Level"
        elif compact in ["temperature", "lst", "landsurfacetemperature"]:
            compact_map[c] = "LST"
        elif compact == "year":
            compact_map[c] = "Year"
    if compact_map:
        df.rename(columns=compact_map, inplace=True)

    return df


def parse_ccd_group(value: object) -> Optional[str]:
    """Map original or merged CCD labels to reconstructed groups."""
    if pd.isna(value):
        return None

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        n = int(round(float(numeric)))
        if n <= 2:
            return "L1_2"
        if n <= 4:
            return "L3_4"
        if n <= 6:
            return "L5_6"
        return "L7_10"

    txt = str(value).strip().lower().replace(" ", "")
    txt = txt.replace("–", "-").replace("—", "-").replace("_", "-")

    if any(k in txt for k in ["1-2", "level1-2", "levels1-2", "low", "低"]):
        return "L1_2"
    if any(k in txt for k in ["3-4", "level3-4", "levels3-4", "middle", "medium", "中"]):
        return "L3_4"
    if any(k in txt for k in ["5-6", "level5-6", "levels5-6", "high", "near", "高", "近"]):
        return "L5_6"

    nums = re.findall(r"\d+", txt)
    if nums:
        n = int(nums[0])
        if n <= 2:
            return "L1_2"
        if n <= 4:
            return "L3_4"
        if n <= 6:
            return "L5_6"
        return "L7_10"

    return None


def ensure_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean, validate, and standardize the input dataframe."""
    df = standardize_columns(raw_df)

    required = [TARGET_COL, YEAR_COL, LEVEL_COL] + BASE_FEATURES + CLIMATE_WITH_PRE
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            "\nExpected columns include Year, LST/Temperature, Coupling_Coordination_Level, "
            "landscape/urbanization predictors, and WS, Tair, SRAD, RH, PRE."
        )

    numeric_cols = [YEAR_COL, TARGET_COL] + BASE_FEATURES + CLIMATE_WITH_PRE
    df = ensure_numeric(df, numeric_cols)
    df["CCD_Group"] = df[LEVEL_COL].apply(parse_ccd_group)

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[YEAR_COL, TARGET_COL, "CCD_Group"]).copy()
    df[YEAR_COL] = df[YEAR_COL].round().astype(int)
    after = len(df)

    invalid_ccd = sorted([x for x in df["CCD_Group"].dropna().unique() if x not in ["L1_2", "L3_4", "L5_6"]])
    if invalid_ccd:
        print(f"WARNING: Excluding CCD groups outside L1_2/L3_4/L5_6: {invalid_ccd}")
        df = df[df["CCD_Group"].isin(["L1_2", "L3_4", "L5_6"])].copy()

    print(f"Rows before cleaning: {before:,}")
    print(f"Rows after cleaning : {len(df):,} (dropped {before - after:,} rows before CCD filtering)")

    if df.empty:
        raise ValueError("No valid rows remain after cleaning.")

    return df


# =============================================================================
# Data-quality diagnostics
# =============================================================================


def data_quality_report(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Build data-quality report for all samples, years, and year-CCD groups."""
    rows = []

    def add_rows(scope: str, sub: pd.DataFrame, year: object = "ALL", ccd: object = "ALL") -> None:
        for f in features:
            if f not in sub.columns:
                continue
            x = pd.to_numeric(sub[f], errors="coerce")
            rows.append({
                "Scope": scope,
                "Year": year,
                "CCD_Group": ccd,
                "Variable": f,
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

    add_rows("ALL", df)
    for year, sub in df.groupby(YEAR_COL, dropna=False):
        add_rows("Year", sub, year=year, ccd="ALL")
    for (year, ccd), sub in df.groupby([YEAR_COL, "CCD_Group"], dropna=False):
        add_rows("Year_CCD", sub, year=year, ccd=ccd)

    return pd.DataFrame(rows)


def sample_size_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([YEAR_COL, "CCD_Group"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values([YEAR_COL, "CCD_Group"])
    )


def remove_bad_features(data: pd.DataFrame, features: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Remove missing, near-empty, and zero-variance predictors."""
    kept, removed = [], []
    for f in features:
        if f not in data.columns:
            removed.append((f, "not_in_table"))
            continue
        x = pd.to_numeric(data[f], errors="coerce")
        if x.notna().sum() < 3:
            removed.append((f, "too_many_missing"))
            continue
        if x.nunique(dropna=True) <= 1:
            removed.append((f, "zero_variance"))
            continue
        kept.append(f)
    return kept, removed


# =============================================================================
# Modeling helpers
# =============================================================================


def build_xgb(seed: int, params: dict, n_jobs: int = 1) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
        eval_metric="rmse",
        verbosity=0,
        **params,
    )


def calc_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def tune_xgb_sequential(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    tuning: str,
    n_jobs: int,
) -> Tuple[dict, float]:
    """Low-memory sequential tuning using KFold RMSE."""
    if tuning == "fixed":
        return FIXED_PARAMS.copy(), np.nan

    n = len(X)
    n_splits = 3 if n >= 300 else 2
    n_splits = min(n_splits, max(2, n // 20))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_params = None
    best_rmse = np.inf

    for i, params in enumerate(PARAM_CANDIDATES, start=1):
        rmses = []
        print(f"    Sequential tuning {i}/{len(PARAM_CANDIDATES)}: {params}")

        for train_idx, val_idx in cv.split(X):
            X_train_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train_raw)
            X_val = imputer.transform(X_val_raw)

            model = build_xgb(seed, params, n_jobs=n_jobs)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            rmses.append(np.sqrt(mean_squared_error(y_val, pred)))

            del imputer, model, X_train, X_val, pred
            gc.collect()

        mean_rmse = float(np.mean(rmses))
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params.copy()

    return best_params if best_params is not None else FIXED_PARAMS.copy(), best_rmse


def safe_spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return np.nan
    rho, _ = spearmanr(a[ok], b[ok])
    return float(rho) if np.isfinite(rho) else np.nan


def safe_kendall(a: Sequence[float], b: Sequence[float]) -> float:
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return np.nan
    tau, _ = kendalltau(a[ok], b[ok])
    return float(tau) if np.isfinite(tau) else np.nan


def safe_wilcoxon(x: Sequence[float], y: Sequence[float]) -> float:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return np.nan
    diff = y[ok].values - x[ok].values
    if np.allclose(diff, 0):
        return np.nan
    try:
        return float(wilcoxon(x[ok], y[ok]).pvalue)
    except Exception:
        return np.nan


def topk_overlap_jaccard(list_a: Sequence[str], list_b: Sequence[str], k: int = 5) -> Tuple[int, float]:
    a = set(list(list_a)[:k])
    b = set(list(list_b)[:k])
    inter = len(a & b)
    union = len(a | b)
    return inter, float(inter / union) if union else np.nan


# =============================================================================
# Output helpers
# =============================================================================


def write_csv(path: Path | str, df: pd.DataFrame) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_obj, index=False, encoding="utf-8-sig")


def append_csv(path: Path | str, rows: List[dict]) -> None:
    path_obj = Path(path)
    if not rows:
        return
    df = pd.DataFrame(rows)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    header = not path_obj.exists()
    df.to_csv(path_obj, mode="a", header=header, index=False, encoding="utf-8-sig")


def read_existing_csv(path: Path | str) -> pd.DataFrame:
    path_obj = Path(path)
    if path_obj.exists():
        return pd.read_csv(path_obj, encoding="utf-8-sig")
    return pd.DataFrame()


def remove_stage_outputs(output_dir: Path, stage_key: str) -> None:
    stage_files = [
        output_dir / f"02_removed_features_{stage_key}.csv",
        output_dir / f"03_best_xgb_params_{stage_key}.csv",
        output_dir / f"04_performance_all_runs_{stage_key}.csv",
        output_dir / f"07_shap_importance_{stage_key}.csv",
        output_dir / f"08_group_shap_share_{stage_key}.csv",
    ]
    for p in stage_files:
        if p.exists():
            p.unlink()


# =============================================================================
# Model-stage workflow
# =============================================================================


def run_one_subset_one_scenario(
    sub_df: pd.DataFrame,
    subset_name: str,
    year: object,
    ccd: object,
    stage_key: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    scenario = SCENARIOS[stage_key]
    scenario_name = str(scenario["full_name"])
    raw_features = list(scenario["features"])

    performance_rows: List[dict] = []
    shap_rows: List[dict] = []
    group_contrib_rows: List[dict] = []
    removed_rows: List[dict] = []
    best_param_rows: List[dict] = []

    sub_df = sub_df.copy().replace([np.inf, -np.inf], np.nan)
    sub_df = sub_df.dropna(subset=[TARGET_COL])

    if len(sub_df) < args.min_samples:
        print(f"Skipping {subset_name}: n={len(sub_df)} < {args.min_samples}")
        return

    features, removed = remove_bad_features(sub_df, raw_features)
    for f, reason in removed:
        removed_rows.append({
            "Year": year,
            "CCD_Group": ccd,
            "subset": subset_name,
            "scenario": scenario_name,
            "feature": f,
            "reason": reason,
        })

    if len(features) < 3:
        print(f"Skipping {subset_name} / {scenario_name}: fewer than three valid predictors.")
        append_csv(output_dir / f"02_removed_features_{stage_key}.csv", removed_rows)
        return

    model_data = sub_df[features + [TARGET_COL]].copy()
    y_all = pd.to_numeric(model_data[TARGET_COL], errors="coerce")
    ok = y_all.notna()
    X_all = model_data.loc[ok, features]
    y_all = y_all.loc[ok]

    if len(X_all) < args.min_samples:
        print(f"Skipping {subset_name} / {scenario_name}: insufficient valid samples.")
        append_csv(output_dir / f"02_removed_features_{stage_key}.csv", removed_rows)
        return

    print("\n" + "=" * 90)
    print(f"Running: {subset_name} | {scenario_name} | n={len(X_all):,} | p={len(features)}")

    try:
        best_params, best_cv_rmse = tune_xgb_sequential(
            X=X_all,
            y=y_all,
            seed=123,
            tuning=args.tuning,
            n_jobs=args.n_jobs,
        )
    except Exception as exc:
        print(f"Tuning failed; using fixed parameters. Error: {exc}")
        best_params, best_cv_rmse = FIXED_PARAMS.copy(), np.nan

    best_param_rows.append({
        "Year": year,
        "CCD_Group": ccd,
        "subset": subset_name,
        "scenario": scenario_name,
        "n": len(X_all),
        "n_features": len(features),
        "features": ",".join(features),
        "best_cv_RMSE": best_cv_rmse,
        "best_params_json": json.dumps(best_params, ensure_ascii=False),
    })

    seeds = [args.seed_base + i for i in range(args.n_repeats)]

    for seed in seeds:
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=args.test_size,
                random_state=seed,
            )

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train_raw)
            X_test = imputer.transform(X_test_raw)

            model = build_xgb(seed, best_params, n_jobs=args.n_jobs)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            met = calc_metrics(y_test, pred)

            performance_rows.append({
                "Year": year,
                "CCD_Group": ccd,
                "subset": subset_name,
                "scenario": scenario_name,
                "repeat_seed": seed,
                "n_total": len(X_all),
                "n_train": len(X_train_raw),
                "n_test": len(X_test_raw),
                "n_features": len(features),
                **met,
            })

            del imputer, model, X_train, X_test, pred
            gc.collect()

        except Exception as exc:
            print(f"Performance repeat failed: seed={seed}, error={exc}")
            traceback.print_exc()

    if not args.skip_shap:
        try:
            print(f"Computing SHAP: {subset_name} | {scenario_name}")

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=args.test_size,
                random_state=123,
            )

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train_raw)
            X_test = imputer.transform(X_test_raw)

            model = build_xgb(123, best_params, n_jobs=args.n_jobs)
            model.fit(X_train, y_train)

            X_test_imp = pd.DataFrame(X_test, columns=features, index=X_test_raw.index)
            if args.shap_sample_max and args.shap_sample_max > 0 and len(X_test_imp) > args.shap_sample_max:
                X_shap = X_test_imp.sample(args.shap_sample_max, random_state=123)
            else:
                X_shap = X_test_imp

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            shap_arr = shap_values[0] if isinstance(shap_values, list) else shap_values
            shap_arr = shap_arr.values if hasattr(shap_arr, "values") else shap_arr

            mean_abs = np.abs(np.asarray(shap_arr)).mean(axis=0)
            total = float(mean_abs.sum())
            if total <= 0:
                total = np.nan

            imp_df = pd.DataFrame({
                "feature": features,
                "mean_abs_SHAP": mean_abs,
                "SHAP_share_percent": mean_abs / total * 100,
            }).sort_values("mean_abs_SHAP", ascending=False)
            imp_df["rank"] = np.arange(1, len(imp_df) + 1)
            imp_df["feature_group"] = imp_df["feature"].apply(
                lambda f: "Climate" if f in CLIMATE_WITH_PRE else "Landscape_Urban"
            )

            for _, row in imp_df.iterrows():
                shap_rows.append({
                    "Year": year,
                    "CCD_Group": ccd,
                    "subset": subset_name,
                    "scenario": scenario_name,
                    "feature": row["feature"],
                    "feature_group": row["feature_group"],
                    "mean_abs_SHAP": row["mean_abs_SHAP"],
                    "SHAP_share_percent": row["SHAP_share_percent"],
                    "rank": row["rank"],
                })

            group_df = imp_df.groupby("feature_group", as_index=False)["mean_abs_SHAP"].sum()
            group_total = group_df["mean_abs_SHAP"].sum()
            for _, grow in group_df.iterrows():
                group_contrib_rows.append({
                    "Year": year,
                    "CCD_Group": ccd,
                    "subset": subset_name,
                    "scenario": scenario_name,
                    "feature_group": grow["feature_group"],
                    "group_mean_abs_SHAP": grow["mean_abs_SHAP"],
                    "group_SHAP_share_percent": (
                        grow["mean_abs_SHAP"] / group_total * 100 if group_total else np.nan
                    ),
                })

            del explainer, shap_values, shap_arr, X_shap, X_test_imp, imputer, model, X_train, X_test
            gc.collect()

        except Exception as exc:
            print(f"SHAP failed: {subset_name} | {scenario_name} | {exc}")
            traceback.print_exc()

    append_csv(output_dir / f"02_removed_features_{stage_key}.csv", removed_rows)
    append_csv(output_dir / f"03_best_xgb_params_{stage_key}.csv", best_param_rows)
    append_csv(output_dir / f"04_performance_all_runs_{stage_key}.csv", performance_rows)
    append_csv(output_dir / f"07_shap_importance_{stage_key}.csv", shap_rows)
    append_csv(output_dir / f"08_group_shap_share_{stage_key}.csv", group_contrib_rows)

    print(f"Finished and saved: {subset_name} | {scenario_name}")
    gc.collect()


def run_stage_model(df: pd.DataFrame, stage_key: str, args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stage_outputs(output_dir, stage_key)

    if args.include_all_samples:
        run_one_subset_one_scenario(
            df,
            subset_name="ALL_samples",
            year="ALL",
            ccd="ALL",
            stage_key=stage_key,
            args=args,
            output_dir=output_dir,
        )

    if not args.no_year_ccd:
        for (year, ccd), sub in df.groupby([YEAR_COL, "CCD_Group"], dropna=False):
            if pd.isna(ccd):
                continue
            subset_name = f"Year_{year}_CCD_{ccd}"
            run_one_subset_one_scenario(
                sub,
                subset_name=subset_name,
                year=year,
                ccd=ccd,
                stage_key=stage_key,
                args=args,
                output_dir=output_dir,
            )


# =============================================================================
# Summary-stage workflow
# =============================================================================


def summarize_performance(perf_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if perf_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_cols = ["Year", "CCD_Group", "subset", "scenario"]
    summary = (
        perf_df.groupby(group_cols, dropna=False)
        .agg(
            n_total=("n_total", "first"),
            n_features=("n_features", "first"),
            successful_repeats=("repeat_seed", "nunique"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
        )
        .reset_index()
    )

    rows = []
    key_cols = ["Year", "CCD_Group", "subset"]
    for key, sub in perf_df.groupby(key_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(key_cols, key))

        m0 = sub[sub["scenario"] == "M0_original_landscape_urban"].copy()
        if m0.empty:
            continue

        for ctrl in ["M1_control_WS_Tair_SRAD_RH_PRE", "M2_control_WS_Tair_SRAD_RH_no_PRE"]:
            mc = sub[sub["scenario"] == ctrl].copy()
            if mc.empty:
                continue

            merged = m0[["repeat_seed", "R2", "RMSE", "MAE"]].merge(
                mc[["repeat_seed", "R2", "RMSE", "MAE"]],
                on="repeat_seed",
                suffixes=("_M0", "_control"),
            )
            if merged.empty:
                continue

            rows.append({
                **meta,
                "control_scenario": ctrl,
                "n_repeats": len(merged),
                "M0_R2_mean": merged["R2_M0"].mean(),
                "control_R2_mean": merged["R2_control"].mean(),
                "delta_R2_control_minus_M0": (merged["R2_control"] - merged["R2_M0"]).mean(),
                "p_wilcoxon_R2": safe_wilcoxon(merged["R2_M0"], merged["R2_control"]),
                "M0_RMSE_mean": merged["RMSE_M0"].mean(),
                "control_RMSE_mean": merged["RMSE_control"].mean(),
                "delta_RMSE_control_minus_M0": (merged["RMSE_control"] - merged["RMSE_M0"]).mean(),
                "p_wilcoxon_RMSE": safe_wilcoxon(merged["RMSE_M0"], merged["RMSE_control"]),
                "M0_MAE_mean": merged["MAE_M0"].mean(),
                "control_MAE_mean": merged["MAE_control"].mean(),
                "delta_MAE_control_minus_M0": (merged["MAE_control"] - merged["MAE_M0"]).mean(),
                "p_wilcoxon_MAE": safe_wilcoxon(merged["MAE_M0"], merged["MAE_control"]),
            })

    return summary, pd.DataFrame(rows)


def calc_rank_stability(shap_df: pd.DataFrame) -> pd.DataFrame:
    if shap_df.empty:
        return pd.DataFrame()

    rows = []
    key_cols = ["Year", "CCD_Group", "subset"]
    for key, sub in shap_df.groupby(key_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(key_cols, key))

        base = sub[sub["scenario"] == "M0_original_landscape_urban"].copy()
        base = base[base["feature"].isin(BASE_FEATURES)]
        if base.empty:
            continue

        for ctrl in ["M1_control_WS_Tair_SRAD_RH_PRE", "M2_control_WS_Tair_SRAD_RH_no_PRE"]:
            cdf = sub[sub["scenario"] == ctrl].copy()
            cdf = cdf[cdf["feature"].isin(BASE_FEATURES)]
            if cdf.empty:
                continue

            merged = base.merge(cdf, on="feature", suffixes=("_M0", "_control"))
            if len(merged) < 3:
                continue

            m0_top5 = base.sort_values("rank")["feature"].head(5).tolist()
            ctrl_top5 = cdf.sort_values("rank")["feature"].head(5).tolist()
            overlap, jaccard = topk_overlap_jaccard(m0_top5, ctrl_top5, k=5)

            rows.append({
                **meta,
                "control_scenario": ctrl,
                "n_common_base_features": len(merged),
                "top5_overlap_count": overlap,
                "top5_jaccard": jaccard,
                "top5_overlap_ratio": overlap / 5,
                "spearman_rank_corr": safe_spearman(merged["rank_M0"], merged["rank_control"]),
                "kendall_rank_corr": safe_kendall(merged["rank_M0"], merged["rank_control"]),
                "spearman_importance_corr": safe_spearman(
                    merged["mean_abs_SHAP_M0"], merged["mean_abs_SHAP_control"]
                ),
                "kendall_importance_corr": safe_kendall(
                    merged["mean_abs_SHAP_M0"], merged["mean_abs_SHAP_control"]
                ),
                "M0_top5": ", ".join(m0_top5),
                "control_top5_base_only": ", ".join(ctrl_top5),
            })

    return pd.DataFrame(rows)


def make_climate_effect_summary(
    delta_df: pd.DataFrame,
    group_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()

    result = delta_df.copy()

    if not group_df.empty:
        climate_share = group_df[group_df["feature_group"] == "Climate"].copy()
        climate_share = climate_share.rename(columns={
            "scenario": "control_scenario",
            "group_SHAP_share_percent": "climate_SHAP_share_percent",
        })
        climate_share = climate_share[
            ["Year", "CCD_Group", "subset", "control_scenario", "climate_SHAP_share_percent"]
        ]
        result = result.merge(
            climate_share,
            on=["Year", "CCD_Group", "subset", "control_scenario"],
            how="left",
        )

    if not shap_df.empty:
        tmp = shap_df[
            (shap_df["feature_group"] == "Climate")
            & (shap_df["scenario"].isin([
                "M1_control_WS_Tair_SRAD_RH_PRE",
                "M2_control_WS_Tair_SRAD_RH_no_PRE",
            ]))
        ].copy()
        if not tmp.empty:
            tmp = tmp.sort_values(["Year", "CCD_Group", "subset", "scenario", "rank"])
            top = tmp.groupby(["Year", "CCD_Group", "subset", "scenario"], dropna=False).head(1)
            top = top.rename(columns={
                "scenario": "control_scenario",
                "feature": "top_climate_feature",
                "SHAP_share_percent": "top_climate_SHAP_share_percent",
            })
            top = top[
                ["Year", "CCD_Group", "subset", "control_scenario",
                 "top_climate_feature", "top_climate_SHAP_share_percent"]
            ]
            result = result.merge(
                top,
                on=["Year", "CCD_Group", "subset", "control_scenario"],
                how="left",
            )

    if not quality_df.empty:
        pre_q = quality_df[
            (quality_df["Scope"] == "Year_CCD") & (quality_df["Variable"] == "PRE")
        ].copy()
        if not pre_q.empty:
            pre_q = pre_q[["Year", "CCD_Group", "zero_rate", "n_unique", "std"]].drop_duplicates()
            pre_q = pre_q.rename(columns={
                "zero_rate": "PRE_zero_rate",
                "n_unique": "PRE_n_unique",
                "std": "PRE_std",
            })
            result["Year"] = result["Year"].astype(str)
            pre_q["Year"] = pre_q["Year"].astype(str)
            result = result.merge(pre_q, on=["Year", "CCD_Group"], how="left")

    return result


def _stable_classification(row: pd.Series) -> str:
    """Classify base-feature ranking stability after adding meteorological controls."""
    overlap = row.get("top5_overlap_count", np.nan)
    rho = row.get("spearman_rank_corr", np.nan)
    tau = row.get("kendall_rank_corr", np.nan)
    jaccard = row.get("top5_jaccard", np.nan)

    if pd.notna(overlap) and pd.notna(rho) and pd.notna(tau) and overlap >= 4 and rho >= 0.70 and tau >= 0.50:
        return "Highly stable"
    if pd.notna(overlap) and pd.notna(rho) and overlap >= 3 and rho >= 0.60:
        return "Stable"
    if (pd.notna(overlap) and overlap >= 3) or (pd.notna(rho) and rho >= 0.40) or (pd.notna(jaccard) and jaccard >= 0.40):
        return "Partly stable"
    return "Unstable"


def build_result_diagnostics(
    perf_summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    climate_effect_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    shap_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create reviewer-facing diagnostics for the meteorological sensitivity results.

    This table is intentionally interpretive. It does not replace the numerical
    result tables; instead, it flags issues that a reviewer may notice, such as:
    limited year coverage in a public sample dataset, M1/M2 equivalence caused by
    zero-variance precipitation, and unstable base-feature rankings.
    """
    rows: List[dict] = []

    # Coverage diagnostics.
    if not perf_summary_df.empty:
        years = sorted(perf_summary_df["Year"].dropna().astype(str).unique().tolist())
        groups = sorted(perf_summary_df["CCD_Group"].dropna().astype(str).unique().tolist())
        scenarios = sorted(perf_summary_df["scenario"].dropna().astype(str).unique().tolist())

        rows.append({
            "Diagnostic_Type": "Coverage",
            "Year": "ALL",
            "CCD_Group": "ALL",
            "Control_Scenario": "ALL",
            "Status": "OK" if len(years) > 1 else "Check",
            "Message": (
                f"Years detected in outputs: {', '.join(years)}. "
                "If this is a public sample file, one-year coverage is acceptable for code testing; "
                "full manuscript reproduction should use all processed study years."
            ),
        })
        rows.append({
            "Diagnostic_Type": "Coverage",
            "Year": "ALL",
            "CCD_Group": "ALL",
            "Control_Scenario": "ALL",
            "Status": "OK" if {"M0_original_landscape_urban", "M1_control_WS_Tair_SRAD_RH_PRE", "M2_control_WS_Tair_SRAD_RH_no_PRE"}.issubset(set(scenarios)) else "Check",
            "Message": f"Scenarios detected: {', '.join(scenarios)}. CCD groups detected: {', '.join(groups)}.",
        })

        expected = {
            "M0_original_landscape_urban",
            "M1_control_WS_Tair_SRAD_RH_PRE",
            "M2_control_WS_Tair_SRAD_RH_no_PRE",
        }
        for key, sub in perf_summary_df.groupby(["Year", "CCD_Group", "subset"], dropna=False):
            y, g, subset = key
            found = set(sub["scenario"].astype(str))
            missing = sorted(expected - found)
            rows.append({
                "Diagnostic_Type": "Scenario_coverage",
                "Year": y,
                "CCD_Group": g,
                "Control_Scenario": "ALL",
                "Status": "OK" if not missing else "Missing",
                "Message": f"{subset}: missing scenarios = {', '.join(missing) if missing else 'None'}.",
            })

    # Performance and M1/M2 equivalence diagnostics.
    if not delta_df.empty:
        for key, sub in delta_df.groupby(["Year", "CCD_Group", "subset"], dropna=False):
            year, ccd, subset = key
            for _, row in sub.iterrows():
                d_r2 = row.get("delta_R2_control_minus_M0", np.nan)
                d_rmse = row.get("delta_RMSE_control_minus_M0", np.nan)
                scenario = row.get("control_scenario", "")
                status = "Improved" if (pd.notna(d_r2) and d_r2 > 0 and pd.notna(d_rmse) and d_rmse < 0) else "Mixed_or_no_improvement"
                rows.append({
                    "Diagnostic_Type": "Performance_change",
                    "Year": year,
                    "CCD_Group": ccd,
                    "Control_Scenario": scenario,
                    "Status": status,
                    "Message": (
                        f"{scenario}: ΔR2={d_r2:.4f} and ΔRMSE={d_rmse:.4f}. "
                        "Positive ΔR2 and negative ΔRMSE indicate better predictive performance after adding controls."
                    ) if pd.notna(d_r2) and pd.notna(d_rmse) else f"{scenario}: insufficient metrics."
                })

            m1 = sub[sub["control_scenario"] == "M1_control_WS_Tair_SRAD_RH_PRE"]
            m2 = sub[sub["control_scenario"] == "M2_control_WS_Tair_SRAD_RH_no_PRE"]
            if not m1.empty and not m2.empty:
                cols = [
                    "control_R2_mean", "delta_R2_control_minus_M0",
                    "control_RMSE_mean", "delta_RMSE_control_minus_M0",
                    "control_MAE_mean", "delta_MAE_control_minus_M0",
                ]
                diffs = []
                for c in cols:
                    if c in m1.columns and c in m2.columns:
                        diffs.append(abs(float(m1.iloc[0][c]) - float(m2.iloc[0][c])))
                identical = bool(diffs) and max(diffs) < 1e-10
                rows.append({
                    "Diagnostic_Type": "M1_M2_comparison",
                    "Year": year,
                    "CCD_Group": ccd,
                    "Control_Scenario": "M1_vs_M2",
                    "Status": "Identical" if identical else "Different",
                    "Message": (
                        "M1 and M2 performance metrics are identical. This usually means PRE was removed because it had no effective within-subset variation, or it contributed no additional information."
                        if identical else
                        "M1 and M2 differ, suggesting that PRE contributes information beyond WS, Tair, SRAD, and RH."
                    ),
                })

    # PRE variation diagnostics.
    if not climate_effect_df.empty:
        for _, row in climate_effect_df.iterrows():
            pre_zero = row.get("PRE_zero_rate", np.nan)
            pre_unique = row.get("PRE_n_unique", np.nan)
            pre_std = row.get("PRE_std", np.nan)
            scenario = row.get("control_scenario", "")
            if scenario != "M1_control_WS_Tair_SRAD_RH_PRE":
                continue

            if pd.notna(pre_unique) and pre_unique <= 1:
                status = "No_effective_PRE_variation"
                msg = (
                    f"PRE has n_unique={pre_unique}, std={pre_std}, zero_rate={pre_zero}. "
                    "M2 excluding PRE is therefore an important sensitivity check."
                )
            elif pd.notna(pre_zero) and pre_zero >= 0.90:
                status = "PRE_mostly_zero"
                msg = (
                    f"PRE zero_rate={pre_zero:.3f}; precipitation is nearly absent on this acquisition date."
                )
            else:
                status = "PRE_variable"
                msg = (
                    f"PRE shows variation: n_unique={pre_unique}, std={pre_std}, zero_rate={pre_zero}."
                )

            rows.append({
                "Diagnostic_Type": "PRE_variation",
                "Year": row.get("Year", np.nan),
                "CCD_Group": row.get("CCD_Group", np.nan),
                "Control_Scenario": scenario,
                "Status": status,
                "Message": msg,
            })

    # Rank-stability diagnostics.
    if not rank_df.empty:
        for _, row in rank_df.iterrows():
            cls = _stable_classification(row)
            rows.append({
                "Diagnostic_Type": "Attribution_stability",
                "Year": row.get("Year", np.nan),
                "CCD_Group": row.get("CCD_Group", np.nan),
                "Control_Scenario": row.get("control_scenario", ""),
                "Status": cls,
                "Message": (
                    f"Top-5 overlap={row.get('top5_overlap_count', np.nan)}, "
                    f"Jaccard={row.get('top5_jaccard', np.nan):.3f}, "
                    f"Spearman={row.get('spearman_rank_corr', np.nan):.3f}, "
                    f"Kendall={row.get('kendall_rank_corr', np.nan):.3f}. "
                    "These metrics describe whether the original landscape/urbanization ranking structure remains stable after adding meteorological controls."
                ),
            })

    # Removed feature diagnostics.
    if not removed_df.empty:
        for _, row in removed_df.iterrows():
            feature = row.get("feature", "")
            reason = row.get("reason", "")
            if feature == "PRE":
                status = "PRE_removed"
            else:
                status = "Feature_removed"
            rows.append({
                "Diagnostic_Type": "Removed_feature",
                "Year": row.get("Year", np.nan),
                "CCD_Group": row.get("CCD_Group", np.nan),
                "Control_Scenario": row.get("scenario", ""),
                "Status": status,
                "Message": f"Feature {feature} was removed from subset {row.get('subset', '')} because: {reason}.",
            })

    return pd.DataFrame(rows)


def make_response_summary(delta_df: pd.DataFrame, rank_df: pd.DataFrame, climate_effect_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact table that can be copied into a response letter."""
    rows = []
    if delta_df.empty:
        return pd.DataFrame()

    for scenario, sub in delta_df.groupby("control_scenario", dropna=False):
        out = {
            "control_scenario": scenario,
            "n_subsets": int(len(sub)),
            "mean_delta_R2": float(sub["delta_R2_control_minus_M0"].mean()),
            "mean_delta_RMSE": float(sub["delta_RMSE_control_minus_M0"].mean()),
            "mean_delta_MAE": float(sub["delta_MAE_control_minus_M0"].mean()),
            "subsets_with_R2_improvement": int((sub["delta_R2_control_minus_M0"] > 0).sum()),
            "subsets_with_RMSE_reduction": int((sub["delta_RMSE_control_minus_M0"] < 0).sum()),
        }
        if not rank_df.empty:
            r = rank_df[rank_df["control_scenario"] == scenario]
            if not r.empty:
                out.update({
                    "mean_top5_overlap": float(r["top5_overlap_count"].mean()),
                    "mean_top5_jaccard": float(r["top5_jaccard"].mean()),
                    "mean_spearman_rank_corr": float(r["spearman_rank_corr"].mean()),
                    "mean_kendall_rank_corr": float(r["kendall_rank_corr"].mean()),
                })
        if not climate_effect_df.empty:
            c = climate_effect_df[climate_effect_df["control_scenario"] == scenario]
            if not c.empty and "climate_SHAP_share_percent" in c.columns:
                out["mean_climate_SHAP_share_percent"] = float(c["climate_SHAP_share_percent"].mean())
                top_features = c["top_climate_feature"].dropna().astype(str).tolist()
                out["top_climate_features_by_subset"] = ", ".join(top_features)
        rows.append(out)

    return pd.DataFrame(rows)


def run_summary(output_dir: Path) -> None:
    print("Generating summary tables for M0/M1/M2 results...")

    quality_df = read_existing_csv(output_dir / "01_data_quality.csv")

    perf_list, shap_list, group_list, removed_list, params_list = [], [], [], [], []

    for key in ["M0", "M1", "M2"]:
        perf_list.append(read_existing_csv(output_dir / f"04_performance_all_runs_{key}.csv"))
        shap_list.append(read_existing_csv(output_dir / f"07_shap_importance_{key}.csv"))
        group_list.append(read_existing_csv(output_dir / f"08_group_shap_share_{key}.csv"))
        removed_list.append(read_existing_csv(output_dir / f"02_removed_features_{key}.csv"))
        params_list.append(read_existing_csv(output_dir / f"03_best_xgb_params_{key}.csv"))

    perf_df = pd.concat([d for d in perf_list if not d.empty], ignore_index=True) if any(not d.empty for d in perf_list) else pd.DataFrame()
    shap_df = pd.concat([d for d in shap_list if not d.empty], ignore_index=True) if any(not d.empty for d in shap_list) else pd.DataFrame()
    group_df = pd.concat([d for d in group_list if not d.empty], ignore_index=True) if any(not d.empty for d in group_list) else pd.DataFrame()
    removed_df = pd.concat([d for d in removed_list if not d.empty], ignore_index=True) if any(not d.empty for d in removed_list) else pd.DataFrame()
    params_df = pd.concat([d for d in params_list if not d.empty], ignore_index=True) if any(not d.empty for d in params_list) else pd.DataFrame()

    perf_summary_df, delta_df = summarize_performance(perf_df)
    rank_df = calc_rank_stability(shap_df)
    climate_effect_df = make_climate_effect_summary(delta_df, group_df, shap_df, quality_df)
    diagnostics_df = build_result_diagnostics(
        perf_summary_df=perf_summary_df,
        delta_df=delta_df,
        rank_df=rank_df,
        climate_effect_df=climate_effect_df,
        removed_df=removed_df,
        shap_df=shap_df,
    )
    response_summary_df = make_response_summary(delta_df, rank_df, climate_effect_df)

    write_csv(output_dir / "02_removed_features_ALL.csv", removed_df)
    write_csv(output_dir / "03_best_xgb_params_ALL.csv", params_df)
    write_csv(output_dir / "04_performance_all_runs_ALL.csv", perf_df)
    write_csv(output_dir / "05_performance_summary.csv", perf_summary_df)
    write_csv(output_dir / "06_model_delta.csv", delta_df)
    write_csv(output_dir / "07_shap_importance_ALL.csv", shap_df)
    write_csv(output_dir / "08_group_shap_share_ALL.csv", group_df)
    write_csv(output_dir / "09_rank_stability.csv", rank_df)
    write_csv(output_dir / "10_climate_effect_summary.csv", climate_effect_df)
    write_csv(output_dir / "12_result_diagnostics.csv", diagnostics_df)
    write_csv(output_dir / "13_response_summary.csv", response_summary_df)

    excel_path = output_dir / "Meteorological_Control_Sensitivity_Results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        quality_df.to_excel(writer, sheet_name="01_data_quality", index=False)
        removed_df.to_excel(writer, sheet_name="02_removed_features", index=False)
        params_df.to_excel(writer, sheet_name="03_best_xgb_params", index=False)
        perf_df.to_excel(writer, sheet_name="04_performance_all_runs", index=False)
        perf_summary_df.to_excel(writer, sheet_name="05_performance_summary", index=False)
        delta_df.to_excel(writer, sheet_name="06_model_delta", index=False)
        shap_df.to_excel(writer, sheet_name="07_shap_importance", index=False)
        group_df.to_excel(writer, sheet_name="08_group_shap_share", index=False)
        rank_df.to_excel(writer, sheet_name="09_rank_stability", index=False)
        climate_effect_df.to_excel(writer, sheet_name="10_climate_effect", index=False)
        diagnostics_df.to_excel(writer, sheet_name="11_result_diagnostics", index=False)
        response_summary_df.to_excel(writer, sheet_name="12_response_summary", index=False)

    template_path = output_dir / "11_interpretation_template.txt"
    with open(template_path, "w", encoding="utf-8") as f:
        f.write("Interpretation template for reviewer response\n")
        f.write("=" * 52 + "\n\n")
        f.write("1. Report M0, M1, and M2 performance using 05_performance_summary.csv.\n")
        f.write("2. Use 06_model_delta.csv to compare M1-M0 and M2-M0.\n")
        f.write("   Positive delta_R2 indicates higher explanatory power after adding controls;\n")
        f.write("   negative delta_RMSE or delta_MAE indicates lower prediction error.\n")
        f.write("3. Use 08_group_shap_share_ALL.csv to report the overall SHAP share of climate variables.\n")
        f.write("4. Use 09_rank_stability.csv to report top-five overlap, Jaccard similarity,\n")
        f.write("   Spearman correlation, and Kendall correlation for the original landscape/urbanization features.\n")
        f.write("5. Use 10_climate_effect_summary.csv as the integrated table for the response letter\n")
        f.write("   and supplementary material.\n")
        f.write("6. Use 12_result_diagnostics.csv to check whether the public sample contains only\n")
        f.write("   one year, whether M1 and M2 are identical because PRE was removed, and whether\n")
        f.write("   the base-feature ranking remains stable.\n")
        f.write("7. Use 13_response_summary.csv for a compact response-letter summary.\n")
        f.write("8. If PRE_zero_rate is close to 1, precipitation has little spatial variation on\n")
        f.write("   the acquisition date. In that case, M2 excluding PRE provides an additional\n")
        f.write("   sensitivity check.\n\n")
        f.write("Suggested wording:\n")
        f.write(
            "To address the reviewer’s concern, five meteorological variables were "
            "added as control variables, including wind speed, near-surface air "
            "temperature, downward shortwave radiation, relative humidity, and "
            "precipitation. The original model was compared with two "
            "meteorological-control models using RMSE, MAE, and R². SHAP-based "
            "ranking stability was further evaluated using top-five feature overlap, "
            "Jaccard similarity, Spearman’s rank correlation, and Kendall’s rank "
            "correlation. Because precipitation was zero or nearly invariant on "
            "some image-acquisition dates, an additional sensitivity model excluding "
            "precipitation was conducted."
        )

    print("\nSummary completed.")
    print(f"Excel workbook: {excel_path}")
    print(f"Interpretation template: {template_path}")


def write_metadata(args: argparse.Namespace, df: pd.DataFrame, output_dir: Path, elapsed: Optional[float] = None) -> None:
    metadata = {
        "script": Path(__file__).name,
        "input_file": str(Path(args.input).expanduser().resolve()),
        "output_dir": str(output_dir),
        "repository_root": str(REPO_ROOT),
        "rows_after_cleaning": int(len(df)),
        "years": sorted(df[YEAR_COL].dropna().unique().astype(int).tolist()),
        "ccd_groups": sorted(df["CCD_Group"].dropna().unique().tolist()),
        "base_features": BASE_FEATURES,
        "climate_with_pre": CLIMATE_WITH_PRE,
        "climate_no_pre": CLIMATE_NO_PRE,
        "scenarios": {k: v["full_name"] for k, v in SCENARIOS.items()},
        "stage": args.stage,
        "n_repeats": int(args.n_repeats),
        "min_samples": int(args.min_samples),
        "test_size": float(args.test_size),
        "seed_base": int(args.seed_base),
        "tuning": args.tuning,
        "skip_shap": bool(args.skip_shap),
        "shap_sample_max": int(args.shap_sample_max),
        "include_all_samples": bool(args.include_all_samples),
        "run_year_ccd": not bool(args.no_year_ccd),
        "n_jobs": int(args.n_jobs),
        "elapsed_seconds": None if elapsed is None else round(float(elapsed), 2),
        "python_version": sys.version,
        "generated_summary_outputs": [
            "05_performance_summary.csv",
            "06_model_delta.csv",
            "09_rank_stability.csv",
            "10_climate_effect_summary.csv",
            "12_result_diagnostics.csv",
            "13_response_summary.csv",
        ],
        "note": (
            "This meteorological-control analysis is a sensitivity test. "
            "SHAP values describe model attribution structure and should not be "
            "interpreted as causal effects. If the public sample file contains only "
            "one study year, it is intended for code testing rather than exact "
            "full-manuscript reproduction."
        ),
    }
    with open(output_dir / "Run_Metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    start = time.time()
    args = parse_args()

    if args.quick:
        args.n_repeats = 3
        args.tuning = "fixed"
        args.shap_sample_max = 500
        print("Quick mode enabled: n_repeats=3, tuning='fixed', shap_sample_max=500")

    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1.")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Meteorological-control sensitivity analysis for XGBoost-SHAP")
    print("=" * 80)
    print(f"Repository root : {REPO_ROOT}")
    print(f"Input file      : {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Stage           : {args.stage}")
    print(f"Repeats         : {args.n_repeats}")
    print(f"Tuning strategy : {args.tuning}")
    print(f"n_jobs          : {args.n_jobs}")
    print("=" * 80)

    raw_df = read_csv_smart(input_path)
    df = prepare_data(raw_df)

    quality_df = data_quality_report(df, BASE_FEATURES + CLIMATE_WITH_PRE + [TARGET_COL])
    write_csv(output_dir / "01_data_quality.csv", quality_df)

    sample_df = sample_size_table(df)
    write_csv(output_dir / "00_sample_size.csv", sample_df)

    print("\nSample distribution:")
    print(sample_df.to_string(index=False))

    if args.stage in ["M0", "M1", "M2"]:
        run_stage_model(df, args.stage, args, output_dir)
    elif args.stage == "SUMMARY":
        run_summary(output_dir)
    elif args.stage == "ALL":
        for stage in ["M0", "M1", "M2"]:
            print("\n" + "#" * 80)
            print(f"Running stage: {stage}")
            print("#" * 80)
            run_stage_model(df, stage, args, output_dir)
        run_summary(output_dir)
    else:
        raise ValueError("Unknown stage.")

    elapsed = time.time() - start
    write_metadata(args, df, output_dir, elapsed=elapsed)

    print("\nAnalysis finished successfully.")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
