# -*- coding: utf-8 -*-
"""XGBoost-SHAP analysis for Zhengzhou land surface temperature research.

This script supports the reproducible workflow used in the manuscript revision:

1. Train and compare Linear Regression, Random Forest, and XGBoost models.
2. Generate SHAP summary, feature-importance, dependence, interaction, and fitting plots.
3. Validate candidate SHAP response breakpoints using segmented regression and
   bootstrap confidence intervals.

Review-ready adjustments in this version:
- repository-relative input/output paths;
- memory-safer XGBoost execution (n_jobs=1);
- conservative breakpoint-status labelling requiring p-value, ΔR², and bootstrap CI;
- flexible parsing of detailed or reconstructed CCD-level labels.

Recommended repository layout::

    Zhengzhou-LST-CCD-XGBoost-SHAP/
    ├── scripts/
    │   └── 01_main_xgboost_shap_analysis.py
    ├── sample_data/
    │   └── 01_main_xgboost_shap_analysis.csv
    └── example_outputs/

Default execution::

    python scripts/01_main_xgboost_shap_analysis.py

The default input path is resolved relative to the repository root, so the script
can be run directly after placing the CSV in ``sample_data/``.
"""

import argparse
import os
import re
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split


# ============================================================
# 0. Repository paths and command-line options
# ============================================================

def get_repository_root() -> Path:
    """Return the project root when this file is stored in ``scripts/``."""
    script_path = Path(__file__).resolve()

    if script_path.parent.name.lower() == "scripts":
        return script_path.parent.parent

    return script_path.parent


REPO_ROOT = get_repository_root()
DEFAULT_INPUT_PATH = REPO_ROOT / "sample_data" / "01_main_xgboost_shap_analysis.csv"
DEFAULT_INPUT_DIR = REPO_ROOT / "sample_data"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "example_outputs" / "01_main_xgboost_shap_analysis_results"


def parse_args():
    """Parse optional command-line arguments while keeping direct-run defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Run XGBoost-SHAP model comparison and SHAP breakpoint "
            "validation for Zhengzhou LST analysis."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Use 'single' for one CSV or 'multi' for multiple yearly CSV files.",
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input CSV path for single-file mode.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing yearly CSV files for multi-file mode.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_BASE),
        help="Directory used to save all output tables and figures.",
    )
    parser.add_argument(
        "--no-process-by-year",
        action="store_true",
        help="Analyze all records together instead of splitting by Year.",
    )

    return parser.parse_args()


# ============================================================
# 1. 安全导入与标志位初始化
# ============================================================

HAS_SM = False
HAS_PYSAL = False
HAS_SCIPY = False

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_SM = True
except ImportError:
    pass

try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    HAS_PYSAL = True
except ImportError:
    pass

try:
    from scipy.stats import f as f_dist
    HAS_SCIPY = True
except ImportError:
    pass


# ============================================================
# 2. 字体设置
# ============================================================

def set_style_font():
    plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['font.weight'] = 'bold'

    base_size = 26
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.labelsize'] = base_size + 4
    plt.rcParams['axes.titlesize'] = base_size + 4
    plt.rcParams['xtick.labelsize'] = base_size
    plt.rcParams['ytick.labelsize'] = base_size
    plt.rcParams['legend.fontsize'] = base_size

    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.unicode_minus'] = False


set_style_font()
warnings.filterwarnings("ignore")

sns.set_theme(style="white", rc={
    "axes.edgecolor": "black",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "SimHei"],
    "font.weight": "bold",
    "axes.labelsize": 30,
    "axes.titlesize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26
})

set_style_font()


# ============================================================
# 3. 全局统一 Colormap
#    只替换 SHAP 图颜色，不改图件排版
# ============================================================

def get_paper_style_cmap():
    """
    替换原来的 YlGnBu。
    Low    = 橘色
    Middle = 稍浅黄色
    High   = 蓝色
    """
    colors = [
        (0.00, "#F46D43"),   # Low: 橘红
        (0.18, "#FDAE61"),   # 橘色
        (0.42, "#FFF2A6"),   # 稍浅黄色
        (0.58, "#E8F6C7"),   # 浅黄绿过渡
        (0.75, "#A6CEE3"),   # 浅蓝
        (1.00, "#2C7FB8")    # High: 蓝色
    ]

    return LinearSegmentedColormap.from_list(
        "Orange_LightYellow_Blue",
        colors,
        N=256
    )


GLOBAL_CMAP = get_paper_style_cmap()


# ============================================================
# 4. 基础工具函数
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_csv_smart(path):
    """Read tabular input robustly.

    Recommended GitHub input format:
        CSV UTF-8 (Comma delimited) (*.csv)

    The function also detects Excel workbooks accidentally renamed as .csv,
    which is a common cause of UnicodeDecodeError on Windows systems.
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
                "Note: For repository reproducibility, please save the input as "
                "CSV UTF-8 (Comma delimited) (*.csv)."
            )
            return df
        except Exception as exc:
            raise ValueError(
                "The input appears to be an Excel workbook rather than a plain-text CSV. "
                "Please save it as CSV UTF-8 (Comma delimited) (*.csv), or keep the "
                "correct Excel suffix and install the required Excel reader. "
                f"Original error: {exc}"
            ) from exc

    encodings = [
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
        "cp936",
        "utf-16",
        "utf-16le",
        "utf-16be",
        "latin1",
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
        "Unable to decode the input file as CSV. Please save it as CSV UTF-8 "
        "(Comma delimited) (*.csv). Tried encodings: " + "; ".join(errors)
    )


def clean_column_name(col):
    col = str(col).replace("\ufeff", "").strip()
    col = col.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    col = re.sub(r"\s+", " ", col)
    return col


def standardize_columns(df):
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    rename_map = {
        "Temperature": "temperature",
        "Temp": "temperature",
        "LST": "temperature",
        "lst": "temperature",

        "Coupling Coordination Level": "ccd_level",
        "Coupling_Coordination_Level": "ccd_level",
        "CouplingCoordinationLevel": "ccd_level",
        "CCD": "ccd_level",
        "ccd": "ccd_level",
        "Level": "ccd_level",

        "Longitude": "lon",
        "Latitude": "lat",
        "Lon": "lon",
        "Lat": "lat",
        "X": "lon",
        "Y": "lat"
    }

    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    compact_map = {}
    for c in df.columns:
        compact = re.sub(r"[\s_]+", "", str(c)).lower()

        if compact == "couplingcoordinationlevel":
            compact_map[c] = "ccd_level"

        if compact == "temperature":
            compact_map[c] = "temperature"

    if compact_map:
        df.rename(columns=compact_map, inplace=True)

    return df


# ============================================================
# 5. 辅助函数：蜂巢图抖动
# ============================================================

def simple_beeswarm(x_values, nbins=40, width=0.4):
    """
    计算蜂巢图的 Y 轴抖动值。
    保持原代码逻辑。
    """
    x_values = np.asarray(x_values)

    if len(x_values) == 0:
        return np.array([])

    finite_mask = np.isfinite(x_values)

    if finite_mask.sum() == 0:
        return np.random.uniform(-0.1, 0.1, len(x_values))

    x_finite = x_values[finite_mask]

    hist_range = (np.min(x_finite), np.max(x_finite))

    if hist_range[0] == hist_range[1]:
        hist_range = (hist_range[0] - 0.1, hist_range[1] + 0.1)

    counts, edges = np.histogram(x_finite, bins=nbins, range=hist_range)

    bin_indices = np.digitize(x_values, edges) - 1
    bin_indices = np.clip(bin_indices, 0, nbins - 1)

    y_values = np.zeros_like(x_values, dtype=float)

    max_count = counts.max()

    if max_count == 0:
        return np.random.uniform(-0.1, 0.1, len(x_values))

    for i in range(len(counts)):
        idxs = np.where(bin_indices == i)[0]

        if len(idxs) == 0:
            continue

        current_width = (counts[i] / max_count) * width
        ys = np.linspace(-current_width, current_width, len(idxs))
        np.random.shuffle(ys)
        y_values[idxs] = ys

    return y_values


# ============================================================
# 6. SHAP 阈值分段回归断点检验
#    注意：不改原图，只单独输出表和诊断图
# ============================================================

TARGET_THRESHOLD_FEATURES = ["BP", "FVC", "WP", "FP", "GDP", "POP"]
N_BOOT = 200
MAX_THRESHOLD_SAMPLE = 5000
RANDOM_STATE = 123


def _linear_design(x):
    x = np.asarray(x, dtype=float)
    return np.column_stack([
        np.ones_like(x),
        x
    ])


def _piecewise_design(x, breakpoint):
    """
    连续一断点分段线性模型：
    y = b0 + b1*x + b2*max(0, x-breakpoint)
    """
    x = np.asarray(x, dtype=float)
    return np.column_stack([
        np.ones_like(x),
        x,
        np.maximum(0.0, x - breakpoint)
    ])


def _sse(y, y_hat):
    residual = np.asarray(y) - np.asarray(y_hat)
    return float(np.sum(residual ** 2))


def estimate_piecewise_breakpoint(x, y, min_segment_ratio=0.10, min_segment_n=30):
    """
    自动估计断点。
    x = 原始变量值，例如 BP、FVC、WP
    y = 对应变量的 SHAP value
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    n = len(x)

    if n < max(60, 2 * min_segment_n + 10):
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    min_seg = max(min_segment_n, int(np.floor(n * min_segment_ratio)))

    if n <= 2 * min_seg + 5:
        return None

    candidate_x = np.unique(x[min_seg:n - min_seg])

    if len(candidate_x) > 300:
        candidate_x = np.unique(
            np.quantile(candidate_x, np.linspace(0.05, 0.95, 300))
        )

    X_lin = _linear_design(x)
    beta_lin, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
    y_lin = X_lin @ beta_lin
    sse_lin = _sse(y, y_lin)

    best = None

    for bp in candidate_x:
        X_pw = _piecewise_design(x, bp)

        try:
            beta_pw, *_ = np.linalg.lstsq(X_pw, y, rcond=None)
            y_pw = X_pw @ beta_pw
            sse_pw = _sse(y, y_pw)
        except Exception:
            continue

        if best is None or sse_pw < best["sse_piecewise"]:
            best = {
                "breakpoint": float(bp),
                "beta_piecewise": beta_pw,
                "sse_piecewise": float(sse_pw)
            }

    if best is None:
        return None

    df1 = 1
    df2 = n - 3
    sse_pw = best["sse_piecewise"]

    if df2 <= 0 or sse_pw <= 0:
        f_value = np.nan
        p_value = np.nan
    else:
        f_value = ((sse_lin - sse_pw) / df1) / (sse_pw / df2)
        f_value = max(float(f_value), 0.0)

        if HAS_SCIPY:
            p_value = float(f_dist.sf(f_value, df1, df2))
        else:
            p_value = np.nan

    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot > 0:
        r2_linear = 1.0 - sse_lin / ss_tot
        r2_piecewise = 1.0 - sse_pw / ss_tot
    else:
        r2_linear = np.nan
        r2_piecewise = np.nan

    beta = best["beta_piecewise"]

    slope_left = float(beta[1])
    slope_right = float(beta[1] + beta[2])

    return {
        "n": int(n),
        "breakpoint": float(best["breakpoint"]),
        "intercept": float(beta[0]),
        "slope_left": slope_left,
        "slope_right": slope_right,
        "slope_change": float(beta[2]),
        "sse_linear": float(sse_lin),
        "sse_piecewise": float(sse_pw),
        "r2_linear": float(r2_linear),
        "r2_piecewise": float(r2_piecewise),
        "delta_r2": float(r2_piecewise - r2_linear),
        "f_value": float(f_value) if np.isfinite(f_value) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan
    }


def bootstrap_breakpoint_ci(x, y, n_boot=N_BOOT, random_state=RANDOM_STATE):
    """
    Bootstrap 估计断点 95% CI。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    n = len(x)

    if n < 80:
        return np.nan, np.nan, 0

    rng = np.random.default_rng(random_state)
    bps = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)

        est = estimate_piecewise_breakpoint(x[idx], y[idx])

        if est is not None and np.isfinite(est["breakpoint"]):
            bps.append(est["breakpoint"])

    if len(bps) < max(20, int(n_boot * 0.20)):
        return np.nan, np.nan, len(bps)

    ci_low, ci_high = np.percentile(bps, [2.5, 97.5])

    return float(ci_low), float(ci_high), len(bps)


def validate_shap_thresholds(
    shap_values_matrix,
    X_df,
    feature_names,
    output_dir,
    analysis_name="",
    target_features=None,
    n_boot=N_BOOT,
    max_samples=MAX_THRESHOLD_SAMPLE
):
    """
    输出 SHAP 阈值分段回归断点检验表。
    不修改原图。
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(shap_values_matrix, list):
        shap_values_matrix = shap_values_matrix[0]

    shap_values_matrix = np.asarray(shap_values_matrix)

    if target_features is None:
        target_features = TARGET_THRESHOLD_FEATURES

    rows = []

    for feat in target_features:
        if feat not in feature_names:
            continue

        feat_idx = feature_names.index(feat)

        x_all = X_df[feat].values.astype(float)
        y_all = shap_values_matrix[:, feat_idx].astype(float)

        valid = np.isfinite(x_all) & np.isfinite(y_all)

        x = x_all[valid]
        y = y_all[valid]

        if len(x) > max_samples:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(x), size=max_samples, replace=False)
            x = x[idx]
            y = y[idx]

        est = estimate_piecewise_breakpoint(x, y)

        if est is None:
            rows.append({
                "Analysis": analysis_name,
                "Feature": feat,
                "N_used": int(len(x)),
                "Estimated_breakpoint": np.nan,
                "Bootstrap_CI_2.5%": np.nan,
                "Bootstrap_CI_97.5%": np.nan,
                "Bootstrap_success_n": 0,
                "F_value": np.nan,
                "P_value": np.nan,
                "R2_linear": np.nan,
                "R2_piecewise": np.nan,
                "Delta_R2": np.nan,
                "Slope_left": np.nan,
                "Slope_right": np.nan,
                "Slope_change": np.nan,
                "Status": "Not estimable"
            })
            continue

        ci_low, ci_high, boot_n = bootstrap_breakpoint_ci(
            x,
            y,
            n_boot=n_boot,
            random_state=RANDOM_STATE
        )

        p_value = est["p_value"]

        # A breakpoint is treated as statistically supported only when the
        # segmented model is significant, the bootstrap confidence interval is
        # estimable, and the piecewise model improves the linear model.
        # This avoids overclaiming a breakpoint based on the p-value alone.
        ci_valid = (
            np.isfinite(ci_low)
            and np.isfinite(ci_high)
            and ci_high > ci_low
        )
        p_valid = np.isfinite(p_value) and p_value < 0.05
        delta_r2_valid = np.isfinite(est["delta_r2"]) and est["delta_r2"] > 0

        if p_valid and ci_valid and delta_r2_valid:
            status = "Statistically supported"
        elif p_valid and not ci_valid:
            status = "Significant but bootstrap-unstable"
        else:
            status = "Candidate turning point only"

        rows.append({
            "Analysis": analysis_name,
            "Feature": feat,
            "N_used": est["n"],
            "Estimated_breakpoint": est["breakpoint"],
            "Bootstrap_CI_2.5%": ci_low,
            "Bootstrap_CI_97.5%": ci_high,
            "Bootstrap_success_n": boot_n,
            "F_value": est["f_value"],
            "P_value": p_value,
            "R2_linear": est["r2_linear"],
            "R2_piecewise": est["r2_piecewise"],
            "Delta_R2": est["delta_r2"],
            "Slope_left": est["slope_left"],
            "Slope_right": est["slope_right"],
            "Slope_change": est["slope_change"],
            "Status": status
        })

    threshold_df = pd.DataFrame(rows)

    out_csv = os.path.join(output_dir, "SHAP_Threshold_Breakpoint_Test.csv")
    threshold_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f" ✓ SHAP breakpoint test table saved: {out_csv}")

    return threshold_df


def plot_threshold_breakpoint_diagnostics(
    shap_values_matrix,
    X_df,
    feature_names,
    threshold_df,
    output_dir,
    analysis_name="",
    target_features=None
):
    """
    单独生成阈值验证图和支持表。
    不影响原来的 SHAP_Dependence_Grid_Fig7。
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(shap_values_matrix, list):
        shap_values_matrix = shap_values_matrix[0]

    shap_values_matrix = np.asarray(shap_values_matrix)

    if target_features is None:
        target_features = TARGET_THRESHOLD_FEATURES

    if threshold_df is None or threshold_df.empty:
        print(" [Info] Empty breakpoint-test table; diagnostic figure skipped.")
        return None

    support_df = threshold_df.copy()

    for col in ["Estimated_breakpoint", "Bootstrap_CI_2.5%", "Bootstrap_CI_97.5%", "P_value"]:
        if col in support_df.columns:
            support_df[col] = pd.to_numeric(support_df[col], errors="coerce")

    supported_mask = (
        support_df["Estimated_breakpoint"].notna()
        & support_df["Bootstrap_CI_2.5%"].notna()
        & support_df["Bootstrap_CI_97.5%"].notna()
        & support_df["P_value"].notna()
        & (support_df["P_value"] < 0.05)
    )

    supported_only = support_df.loc[supported_mask].copy()

    if len(supported_only) > 0:
        supported_only["Threshold_support"] = "Supported by segmented regression and bootstrap CI"

    supported_csv = os.path.join(
        output_dir,
        "SHAP_Threshold_Breakpoint_Supported_Only.csv"
    )

    supported_only.to_csv(
        supported_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f" ✓ Statistically supported breakpoint table saved: {supported_csv}")

    valid_features = [
        f for f in target_features
        if f in feature_names and f in threshold_df["Feature"].values
    ]

    if len(valid_features) == 0:
        print(" [Info] No valid breakpoint features; diagnostic figure skipped.")
        return supported_only

    n = len(valid_features)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(6.6 * n, 5.6),
        dpi=300,
        facecolor="white"
    )

    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, valid_features):
        feat_idx = feature_names.index(feat)

        x_all = X_df[feat].values.astype(float)
        y_all = shap_values_matrix[:, feat_idx].astype(float)

        valid = np.isfinite(x_all) & np.isfinite(y_all)

        x = x_all[valid]
        y = y_all[valid]

        if len(x) == 0:
            continue

        if len(x) > 6000:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(x), size=6000, replace=False)
            x_plot = x[idx]
            y_plot = y[idx]
        else:
            x_plot = x
            y_plot = y

        ax.grid(
            True,
            linestyle="--",
            alpha=0.45,
            color="gray",
            linewidth=0.5,
            zorder=0
        )

        ax.scatter(
            x_plot,
            y_plot,
            s=18,
            color="#BDBDBD",
            alpha=0.55,
            edgecolors="none",
            zorder=1
        )

        if HAS_SM and len(x) > 30:
            try:
                sort_ids = np.argsort(x)
                z = lowess(
                    y[sort_ids],
                    x[sort_ids],
                    frac=0.20,
                    return_sorted=True
                )

                ax.plot(
                    z[:, 0],
                    z[:, 1],
                    color="black",
                    linewidth=2.2,
                    alpha=0.85,
                    zorder=3,
                    label="LOWESS"
                )
            except Exception:
                pass

        row = threshold_df[threshold_df["Feature"] == feat].iloc[0]

        bp = pd.to_numeric(row.get("Estimated_breakpoint", np.nan), errors="coerce")
        ci_low = pd.to_numeric(row.get("Bootstrap_CI_2.5%", np.nan), errors="coerce")
        ci_high = pd.to_numeric(row.get("Bootstrap_CI_97.5%", np.nan), errors="coerce")
        p_value = pd.to_numeric(row.get("P_value", np.nan), errors="coerce")
        status = str(row.get("Status", ""))

        if np.isfinite(bp):
            est = estimate_piecewise_breakpoint(x, y)

            if est is not None:
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 300)

                beta = np.array([
                    est["intercept"],
                    est["slope_left"],
                    est["slope_right"] - est["slope_left"]
                ])

                y_line = _piecewise_design(x_line, bp) @ beta

                ax.plot(
                    x_line,
                    y_line,
                    color="#D73027",
                    linewidth=2.4,
                    zorder=4,
                    label="Segmented regression"
                )

            if np.isfinite(ci_low) and np.isfinite(ci_high) and ci_high > ci_low:
                ax.axvspan(
                    ci_low,
                    ci_high,
                    color="#FDAE61",
                    alpha=0.22,
                    zorder=2,
                    label="95% bootstrap CI"
                )

            ax.axvline(
                bp,
                color="#D73027",
                linestyle="--",
                linewidth=2.0,
                zorder=5
            )

            p_text = f"p = {p_value:.3g}" if np.isfinite(p_value) else "p = NA"

            if np.isfinite(ci_low) and np.isfinite(ci_high):
                ci_text = f"95% CI: {ci_low:.3f}–{ci_high:.3f}"
            else:
                ci_text = "95% CI: NA"

            ax.text(
                0.04,
                0.96,
                f"Breakpoint = {bp:.3f}\n{ci_text}\n{p_text}\n{status}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
                fontname="Times New Roman",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="#D73027",
                    alpha=0.90
                )
            )

        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.80,
            zorder=1
        )

        ax.set_xlabel(
            feat,
            fontsize=20,
            fontweight="bold",
            fontname="Times New Roman"
        )

        ax.set_ylabel(
            "SHAP Value",
            fontsize=20,
            fontweight="bold",
            fontname="Times New Roman"
        )

        ax.set_title(
            f"{feat} breakpoint test",
            fontsize=22,
            fontweight="bold",
            fontname="Times New Roman"
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=16,
            direction="in",
            width=1.5,
            length=6
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=14
        )
        plt.subplots_adjust(bottom=0.20)
    else:
        plt.tight_layout()

    figure_path = os.path.join(
        output_dir,
        "SHAP_Threshold_Breakpoint_Diagnostic_Figure.jpg"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f" ✓ Breakpoint diagnostic figure saved: {figure_path}")

    return supported_only


# ============================================================
# 7. SHAP 依赖图网格
#    保持原图排版，不标阈值，只换配色
# ============================================================

def plot_shap_dependence_grid(shap_values_matrix, X_df, feature_names, output_dir, group_label=""):
    os.makedirs(output_dir, exist_ok=True)

    num_vars = len(feature_names)
    num_plots = min(6, num_vars)

    print(f"Generating SHAP dependence grid ({num_plots} panels)...")

    if isinstance(shap_values_matrix, list):
        shap_values_matrix = shap_values_matrix[0]

    mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    top_indices = sorted_indices[:num_plots]

    rows = 2 if num_plots > 4 else 1

    fig_width = 24
    fig_height = 9 if rows == 2 else 5

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300, facecolor='white')

    chart_left_start = 0.08
    right_margin = 0.98
    top_margin = 0.92
    bottom_margin = 0.12

    ax_label = fig.add_axes([0.01, bottom_margin, 0.04, top_margin - bottom_margin])
    ax_label.set_facecolor('#E0E0E0')
    ax_label.set_xticks([])
    ax_label.set_yticks([])

    for spine in ax_label.spines.values():
        spine.set_visible(False)

    ax_label.text(
        0.5,
        0.5,
        group_label,
        ha='center',
        va='center',
        rotation=90,
        fontsize=26,
        fontweight='bold',
        fontname='Times New Roman',
        transform=ax_label.transAxes
    )

    plot_w_gap = 0.03
    plot_h_gap = 0.15
    total_plot_width = right_margin - chart_left_start
    single_plot_width = (total_plot_width - 3 * plot_w_gap) / 4

    if rows == 1:
        single_plot_height = top_margin - bottom_margin
    else:
        single_plot_height = (top_margin - bottom_margin - plot_h_gap) / 2

    positions = []

    row1_y = bottom_margin + single_plot_height + plot_h_gap if rows == 2 else bottom_margin

    for i in range(min(4, num_plots)):
        x = chart_left_start + i * (single_plot_width + plot_w_gap)
        positions.append([x, row1_y, single_plot_width, single_plot_height])

    if num_plots > 4:
        row2_y = bottom_margin
        x1 = chart_left_start + 1 * (single_plot_width + plot_w_gap)
        positions.append([x1, row2_y, single_plot_width, single_plot_height])

        if num_plots > 5:
            x2 = chart_left_start + 2 * (single_plot_width + plot_w_gap)
            positions.append([x2, row2_y, single_plot_width, single_plot_height])

    for i, (feat_idx, pos) in enumerate(zip(top_indices, positions)):
        ax = fig.add_axes(pos)

        feat_name = feature_names[feat_idx]
        x_data = X_df.iloc[:, feat_idx].values
        y_data = shap_values_matrix[:, feat_idx]

        best_inter_feat = None
        best_corr = -1
        inter_data = x_data

        for other_idx, other_name in enumerate(feature_names):
            if other_idx == feat_idx:
                continue

            try:
                candidate_data = X_df.iloc[:, other_idx].values.astype(float)
                corr = np.abs(np.corrcoef(candidate_data, y_data)[0, 1])

                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_inter_feat = other_name
                    inter_data = candidate_data
            except Exception:
                continue

        cbar_label = best_inter_feat if best_inter_feat else feat_name

        vmin = np.nanmin(inter_data)
        vmax = np.nanmax(inter_data)

        if vmin >= 0:
            vmin = 0

        norm = Normalize(vmin=vmin, vmax=vmax)

        ax.grid(
            True,
            linestyle='--',
            alpha=0.5,
            color='gray',
            linewidth=0.5,
            zorder=0
        )

        scatter = ax.scatter(
            x_data,
            y_data,
            c=inter_data,
            cmap=GLOBAL_CMAP,
            s=30,
            alpha=0.9,
            edgecolors='none',
            norm=norm,
            zorder=2
        )

        if HAS_SM and len(x_data) > 10:
            try:
                sort_ids = np.argsort(x_data)
                x_sorted = x_data[sort_ids]
                y_sorted = y_data[sort_ids]
                z = lowess(y_sorted, x_sorted, frac=0.2)

                ax.plot(
                    z[:, 0],
                    z[:, 1],
                    color='black',
                    linewidth=2.5,
                    alpha=0.8,
                    zorder=3
                )
            except Exception:
                pass

        ax.tick_params(
            axis='x',
            direction='in',
            length=6,
            width=1.5,
            top=False,
            labelsize=18
        )
        ax.tick_params(
            axis='y',
            direction='out',
            labelsize=18
        )

        ax.set_title("")

        ax.set_xlabel(
            feat_name,
            fontsize=20,
            fontweight='bold',
            fontname='Times New Roman'
        )

        if i == 0 or (rows == 2 and i == 4):
            ax.set_ylabel(
                "SHAP Value",
                fontsize=20,
                fontweight='bold',
                fontname='Times New Roman'
            )
        else:
            ax.set_ylabel("")

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
        cbar.ax.set_title(
            cbar_label,
            fontsize=16,
            fontweight='bold',
            fontname='Times New Roman',
            pad=10
        )
        cbar.set_label("")
        cbar.ax.tick_params(labelsize=14)
        cbar.outline.set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=18)

    output_path = os.path.join(output_dir, "SHAP_Dependence_Grid_Fig7.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f" ✓ SHAP dependence grid saved: {output_path}")


# ============================================================
# 8. SHAP 交互矩阵
#    保持原字体大小、色柱、布局，只换配色
# ============================================================

def plot_new_interaction_matrix(shap_interaction_values, X_test, feature_names, output_dir, file_tag=""):
    print("Generating SHAP interaction matrix...")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[0]

    total_interaction = np.abs(shap_interaction_values).sum(axis=(0, 2))

    top_n = 7

    if len(feature_names) < top_n:
        top_n = len(feature_names)

    top_indices = np.argsort(total_interaction)[::-1][:top_n]

    subset_shap = shap_interaction_values[:, top_indices, :][:, :, top_indices]
    subset_names = [feature_names[i] for i in top_indices]
    subset_X = X_test.iloc[:, top_indices]

    n = len(subset_names)

    mean_signed_interaction = subset_shap.mean(axis=0)
    mean_abs_interaction = np.abs(mean_signed_interaction)

    max_off_diag = np.max(mean_abs_interaction[~np.eye(n, dtype=bool)])

    if max_off_diag == 0:
        max_off_diag = 1e-9

    norm_matrix = Normalize(vmin=0, vmax=max_off_diag)

    fig = plt.figure(figsize=(n * 2.2, n * 2.2), dpi=300, facecolor='white')
    gs = gridspec.GridSpec(n, n, figure=fig, wspace=0.08, hspace=0.08)

    col_limits = []

    for j in range(n):
        all_vals = []

        for i in range(j, n):
            all_vals.extend(subset_shap[:, i, j])

        if len(all_vals) > 0:
            max_abs = max(abs(np.min(all_vals)), abs(np.max(all_vals)))

            if max_abs == 0:
                max_abs = 1.0

            limit = max_abs * 1.1
            col_limits.append((-limit, limit))
        else:
            col_limits.append((-1, 1))

    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])

            # 上三角：Heatmap
            if i < j:
                abs_val = mean_abs_interaction[i, j]
                signed_val = mean_signed_interaction[i, j]

                cell_color = GLOBAL_CMAP(norm_matrix(abs_val))
                ax.set_facecolor(cell_color)

                text_color = 'white' if norm_matrix(abs_val) > 0.6 else 'black'

                ax.text(
                    0.5,
                    0.5,
                    f"{signed_val:.3f}",
                    ha='center',
                    va='center',
                    fontsize=32,
                    fontweight='bold',
                    color=text_color,
                    fontname='Times New Roman'
                )

                ax.set_xticks([])
                ax.set_yticks([])

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

            # 对角线 + 下三角：Beeswarm
            else:
                ax.set_facecolor('#F2F2F2')
                ax.set_axisbelow(True)

                ax.grid(
                    axis='x',
                    linestyle='-',
                    linewidth=1.0,
                    alpha=1.0,
                    color='white',
                    zorder=0
                )

                ax.axvline(
                    x=0,
                    color='#666666',
                    linestyle='-',
                    linewidth=2.5,
                    alpha=1.0,
                    zorder=1
                )

                x_vals = subset_shap[:, i, j]
                feature_values = subset_X.iloc[:, j].values

                c_norm = Normalize(
                    vmin=np.min(feature_values),
                    vmax=np.max(feature_values)
                )

                y_vals = simple_beeswarm(x_vals, nbins=30, width=0.4)

                ax.scatter(
                    x_vals,
                    y_vals,
                    c=feature_values,
                    cmap=GLOBAL_CMAP,
                    norm=c_norm,
                    s=12,
                    alpha=0.9,
                    edgecolors='none',
                    zorder=2
                )

                ax.set_yticks([])
                ax.set_ylim(-0.6, 0.6)

                # 保持原代码的刻度限制 [-2, 2] 逻辑
                limit_raw = col_limits[j][1] / 1.1

                if limit_raw > 2.0:
                    limit_raw = 2.0

                if limit_raw < 1.0:
                    vmin, vmax = -1.2, 1.2
                    custom_ticks = np.array([-1.0, 0.0, 1.0])
                else:
                    vmin, vmax = -limit_raw * 1.1, limit_raw * 1.1
                    start_tick = np.ceil(-limit_raw)
                    end_tick = np.floor(limit_raw)
                    custom_ticks = np.arange(start_tick, end_tick + 0.1, 1)

                ax.set_xlim(vmin, vmax)
                ax.set_xticks(custom_ticks)

                for spine in ax.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1)

                ax.tick_params(
                    axis='x',
                    direction='out',
                    length=3,
                    width=1,
                    labelsize=22
                )

                if i == n - 1:
                    pass
                else:
                    ax.tick_params(labelbottom=False)

            if j == 0:
                ax.set_ylabel(
                    subset_names[i],
                    fontsize=32,
                    fontweight='bold',
                    fontname='Times New Roman',
                    labelpad=5,
                    rotation=90
                )

            if i == 0:
                ax.set_title(
                    subset_names[j],
                    fontsize=32,
                    fontweight='bold',
                    fontname='Times New Roman',
                    pad=5
                )

    # 保留原色柱
    cax = fig.add_axes([0.91, 0.25, 0.025, 0.5])

    sm = ScalarMappable(
        cmap=GLOBAL_CMAP,
        norm=Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_ticks([])
    cbar.set_label(
        'Raw feature value / |Interaction|',
        fontsize=18,
        fontweight='bold',
        fontname='Times New Roman',
        labelpad=15
    )

    cax.text(
        1.5,
        1.02,
        'High',
        transform=cax.transAxes,
        ha='center',
        va='bottom',
        fontsize=14,
        fontweight='bold',
        fontname='Times New Roman'
    )

    cax.text(
        1.5,
        -0.02,
        'Low',
        transform=cax.transAxes,
        ha='center',
        va='top',
        fontsize=14,
        fontweight='bold',
        fontname='Times New Roman'
    )

    output_path = os.path.join(output_dir, "SHAP_Interaction_Matrix.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f" ✓ SHAP interaction matrix saved: {output_path}")


# ============================================================
# 9. Bar + Rose + SHAP Summary
#    保持原排版、玫瑰图、色柱、SHAP Value 边界值，只换配色
# ============================================================

def plot_split_shap_visualizations(
    shap_values_matrix,
    X_test_df,
    feature_names,
    output_dir,
    is_urban_scale=False
):
    print("Generating SHAP bar and rose plots...")
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(shap_values_matrix, list):
        shap_values_matrix = shap_values_matrix[0]

    mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0)

    total_shap_sum = np.sum(mean_abs_shap)

    if total_shap_sum == 0:
        total_shap_sum = 1e-9

    percentages = (mean_abs_shap / total_shap_sum) * 100

    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_features = np.array(feature_names)[sorted_indices]
    sorted_shap_values = mean_abs_shap[sorted_indices]
    sorted_percentages = percentages[sorted_indices]

    num_vars = len(sorted_features)

    vmin = min(sorted_shap_values)
    vmax = max(sorted_shap_values)

    bar_norm = Normalize(vmin=vmin, vmax=vmax)
    bar_colors = GLOBAL_CMAP(bar_norm(sorted_shap_values))

    # 保存 SHAP 重要性表
    importance_df = pd.DataFrame({
        "Feature": sorted_features,
        "Mean_abs_SHAP": sorted_shap_values,
        "Percentage": sorted_percentages
    })

    importance_df.to_csv(
        os.path.join(output_dir, "SHAP_Feature_Importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # A. SHAP Summary Plot
    # --------------------------------------------------------
    fig_shap = plt.figure(figsize=(14, 12), dpi=300, facecolor='white')

    shap.summary_plot(
        shap_values_matrix,
        X_test_df,
        feature_names=feature_names,
        plot_type="dot",
        max_display=num_vars,
        cmap=GLOBAL_CMAP,
        show=False
    )

    ax_shap = plt.gca()

    ax_shap.grid(
        True,
        axis='x',
        linestyle='--',
        alpha=0.5,
        color='gray',
        zorder=0
    )

    # Keep a fixed x-axis range for visual comparability with manuscript figures.
    # If the user's data have a wider SHAP range, this line can be adjusted.
    ax_shap.set_xlim(-5, 5)

    ax_shap.set_xlabel(
        "SHAP Value",
        fontsize=28,
        fontweight='bold',
        fontname='Times New Roman'
    )

    plt.yticks(
        fontsize=24,
        fontweight='bold',
        fontname='Times New Roman'
    )

    shap_output_path = os.path.join(output_dir, "SHAP_Summary_Plot.jpg")
    plt.savefig(shap_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_shap)

    # --------------------------------------------------------
    # B. Bar + Rose
    # --------------------------------------------------------
    fig_bar = plt.figure(figsize=(12, 10), dpi=300, facecolor='white')

    base_plot_bottom = 0.1
    base_plot_height = 0.85
    base_cbar_width = 0.04
    gap = 0.02
    base_bar_width = 0.75
    base_start_left = 0.1

    if is_urban_scale:
        rose_scale = 0.85
        bar_scale = 0.90
        cbar_scale = 0.90
    else:
        rose_scale = 1.0
        bar_scale = 1.0
        cbar_scale = 1.0

    current_plot_height = base_plot_height
    current_plot_bottom = base_plot_bottom
    current_bar_width = base_bar_width

    cbar_left = base_start_left
    bar_left = cbar_left + base_cbar_width + gap

    # 色柱
    effective_cbar_width = base_cbar_width * cbar_scale

    ax_cbar = fig_bar.add_axes([
        cbar_left,
        current_plot_bottom,
        effective_cbar_width,
        current_plot_height
    ])

    sm = ScalarMappable(cmap=GLOBAL_CMAP, norm=bar_norm)

    cbar = fig_bar.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.set_ticks([])
    cbar.ax.yaxis.set_ticks_position('left')

    ax_cbar.text(
        -1.5,
        1.0,
        'High',
        transform=ax_cbar.transAxes,
        ha='center',
        va='bottom',
        fontsize=20,
        fontweight='bold',
        fontname='Times New Roman'
    )

    ax_cbar.text(
        -1.5,
        0.0,
        'Low',
        transform=ax_cbar.transAxes,
        ha='center',
        va='top',
        fontsize=20,
        fontweight='bold',
        fontname='Times New Roman'
    )

    cbar.outline.set_visible(False)

    ax_cbar.text(
        -1.5,
        0.5,
        'Mean|SHAP Value|',
        transform=ax_cbar.transAxes,
        fontsize=20,
        rotation=90,
        va='center',
        fontweight='bold',
        ha='center',
        fontname='Times New Roman'
    )

    # Bar
    ax_bar = fig_bar.add_axes([
        bar_left,
        current_plot_bottom,
        current_bar_width,
        current_plot_height
    ])

    ax_bar.grid(False)
    ax_bar.xaxis.tick_bottom()
    ax_bar.xaxis.set_label_position("bottom")
    ax_bar.invert_xaxis()

    bar_thickness = 0.65 * bar_scale

    ax_bar.barh(
        y=range(num_vars),
        width=sorted_shap_values,
        color=bar_colors,
        height=bar_thickness,
        edgecolor='white',
        linewidth=1.2
    )

    ax_bar.invert_yaxis()
    ax_bar.yaxis.tick_right()
    ax_bar.yaxis.set_label_position("right")
    ax_bar.set_yticks(range(num_vars))

    ax_bar.set_yticklabels(
        sorted_features,
        fontsize=20,
        fontweight='bold',
        fontname='Times New Roman'
    )

    ax_bar.set_xlabel(
        'Mean|SHAP Value|',
        size=24,
        fontweight='bold',
        labelpad=10,
        fontname='Times New Roman'
    )

    ax_bar.spines['left'].set_visible(False)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_position(('data', 0))
    ax_bar.spines['right'].set_visible(True)
    ax_bar.spines['bottom'].set_visible(True)

    ax_bar.tick_params(
        axis='x',
        which='major',
        direction='in',
        labelsize=18,
        length=6,
        width=1.5
    )

    for spine in ax_bar.spines.values():
        spine.set_linewidth(2)
        spine.set_color('#333333')

    for i, v in enumerate(sorted_shap_values):
        ax_bar.text(
            v,
            i,
            f'{v:.4f} ',
            ha='right',
            va='center',
            fontsize=16,
            fontweight='bold',
            color='black',
            fontname='Times New Roman'
        )

    # Rose
    if num_vars > 0:
        base_length = 4.0
        fixed_increment = 0.5
        colored_ring_width = 2.0
        one_oclock_offset = np.pi / 21

        widths = (sorted_shap_values / total_shap_sum) * 2 * np.pi
        thetas = np.cumsum([0] + widths[:-1].tolist()) - one_oclock_offset

        total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
        inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
        inner_colors = ['#F5F5F5', '#FFFFFF'] * (num_vars // 2 + 1)

        base_rose_size = 0.45
        rose_size = base_rose_size * rose_scale
        offset = -0.03

        rose_rect = [
            bar_left + offset,
            current_plot_bottom + offset,
            rose_size,
            rose_size
        ]

        ax_rose = fig_bar.add_axes(rose_rect, projection='polar')
        ax_rose.patch.set_alpha(0)

        ax_rose.bar(
            x=thetas,
            height=inner_heights,
            width=widths,
            color=inner_colors[:num_vars],
            align='edge',
            edgecolor='white',
            linewidth=1.0
        )

        ax_rose.bar(
            x=thetas,
            height=[colored_ring_width] * num_vars,
            width=widths,
            bottom=inner_heights,
            color=bar_colors,
            align='edge',
            edgecolor='white',
            linewidth=1.0
        )

        for i in range(num_vars):
            label_angle = thetas[i] + widths[i] / 2
            label_radius = total_lengths[i] + 0.5

            ax_rose.text(
                label_angle,
                label_radius,
                f'{sorted_percentages[i]:.1f}%',
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                fontname='Times New Roman',
                bbox=dict(
                    boxstyle='round,pad=0.1',
                    facecolor='white',
                    alpha=0.7,
                    edgecolor='none'
                )
            )

        ax_rose.set_axis_off()
        ax_rose.set_theta_zero_location('N')
        ax_rose.set_theta_direction(-1)

    bar_output_path = os.path.join(output_dir, "SHAP_Feature_Importance_Bar.jpg")
    plt.savefig(bar_output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_bar)

    print(f" ✓ SHAP summary plot saved: {shap_output_path}")
    print(f" ✓ SHAP bar and rose plot saved: {bar_output_path}")


# ============================================================
# 10. 拟合图
#     Fitting_Plot_FigS5 点颜色保持原来 YlGnBu
# ============================================================

def plot_fitting_curve(y_true, y_pred, output_dir, analysis_name):
    print("Generating observed-versus-predicted fitting plot...")
    os.makedirs(output_dir, exist_ok=True)

    # 这里固定为原来的 YlGnBu，不跟随新的 GLOBAL_CMAP
    point_color = plt.get_cmap('YlGnBu')(0.6)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300, facecolor='white')

    ax.grid(
        True,
        linestyle='--',
        alpha=0.5,
        color='gray',
        linewidth=0.5,
        zorder=0
    )

    ax.scatter(
        y_true,
        y_pred,
        c=[point_color],
        s=60,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,
        zorder=2
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    span = max_val - min_val

    plot_min = min_val - span * 0.05
    plot_max = max_val + span * 0.05

    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        color='black',
        linestyle='--',
        linewidth=2.0,
        label='1:1 Line',
        zorder=1
    )

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    text_str = f'$R^2 = {r2:.3f}$\n$RMSE = {rmse:.3f}$\n$MAE = {mae:.3f}$'

    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=22,
        fontweight='bold',
        verticalalignment='top',
        fontname='Times New Roman',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='white',
            alpha=0.9,
            edgecolor='#cccccc'
        )
    )

    ax.set_xlabel(
        "Observed Temperature (°C)",
        fontsize=26,
        fontweight='bold',
        fontname='Times New Roman'
    )

    ax.set_ylabel(
        "Predicted Temperature (°C)",
        fontsize=26,
        fontweight='bold',
        fontname='Times New Roman'
    )

    ax.set_title(
        f"Observed vs Predicted ({analysis_name})",
        fontsize=28,
        fontweight='bold',
        fontname='Times New Roman',
        pad=15
    )

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=22,
        direction='in',
        width=2.0,
        length=8
    )

    plt.tight_layout()

    output_path = os.path.join(output_dir, "Fitting_Plot_FigS5.jpg")

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close(fig)

    print(f" ✓ Fitting plot saved: {output_path}")


# ============================================================
# 11. 核心分析流程
# ============================================================

def run_advanced_analysis(
    data,
    predictors,
    target_col,
    coord_cols,
    output_dir,
    analysis_name,
    is_urban=False,
    group_label=""
):
    print(f"\n>>> Starting analysis: {analysis_name} (Urban={is_urban}) <<<")

    os.makedirs(output_dir, exist_ok=True)

    predictors = [p for p in predictors if p in data.columns]
    coord_cols = [c for c in coord_cols if c in data.columns]

    valid_cols = predictors + [target_col] + coord_cols
    model_data = data[valid_cols].copy()

    for col in valid_cols:
        model_data[col] = pd.to_numeric(model_data[col], errors='coerce')

    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=[target_col] + predictors)

    if len(model_data) < 20:
        print("Insufficient sample size; skipped.")
        return None

    # --------------------------------------------------------
    # A. Correlation
    # --------------------------------------------------------
    try:
        cor_data = model_data.fillna(model_data.mean(numeric_only=True))
        corr_matrix = cor_data[predictors + [target_col]].corr()

        plt.figure(figsize=(12, 10))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=GLOBAL_CMAP,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            annot_kws={"size": 12, "weight": "bold"}
        )

        plt.title(
            f"{analysis_name} Correlation",
            fontname='Times New Roman',
            fontsize=20,
            fontweight='bold'
        )

        plt.savefig(os.path.join(output_dir, "Correlation_Plot.pdf"))
        plt.close()
    except Exception as e:
        print(f" [Warning] Correlation plot generation failed: {e}")

    # --------------------------------------------------------
    # B. Model
    # --------------------------------------------------------
    X_raw = model_data[predictors]
    y_raw = model_data[target_col]

    if len(coord_cols) >= 2:
        Coords = model_data[coord_cols]
        X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
            X_raw,
            y_raw,
            Coords,
            test_size=0.25,
            random_state=123
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=0.25,
            random_state=123
        )
        coords_test = None

    kf = KFold(n_splits=5, shuffle=True, random_state=123)

    # 1. Linear Regression
    lm_model = LinearRegression()
    lm_model.fit(X_train, y_train)

    lm_cv = cross_val_score(lm_model, X_train, y_train, cv=kf, scoring='r2')
    lm_pred = lm_model.predict(X_test)

    lm_r2 = r2_score(y_test, lm_pred)
    lm_rmse = np.sqrt(mean_squared_error(y_test, lm_pred))
    lm_mae = mean_absolute_error(y_test, lm_pred)

    # 2. Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        random_state=123
    )
    rf_model.fit(X_train, y_train)

    rf_cv = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
    rf_pred = rf_model.predict(X_test)

    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mae = mean_absolute_error(y_test, rf_pred)

    # 3. XGBoost
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=123,
        n_jobs=1  # safer for reviewer-side reproduction on ordinary computers
    )

    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [300, 500],
        'subsample': [0.7],
        'colsample_bytree': [0.7],
        'reg_alpha': [0.1],
        'reg_lambda': [1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_xgb_model = grid_search.best_estimator_

    xgb_cv = cross_val_score(
        best_xgb_model,
        X_train,
        y_train,
        cv=kf,
        scoring='r2'
    )

    xgb_pred = best_xgb_model.predict(X_test)

    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_mae = mean_absolute_error(y_test, xgb_pred)

    # --------------------------------------------------------
    # 保存模型表格
    # --------------------------------------------------------
    print("Saving model comparison table...")

    compare_df = pd.DataFrame({
        'Analysis': [analysis_name] * 3,
        'Model': ["Linear Regression", "Random Forest", "XGBoost"],
        'Train_R2': [
            r2_score(y_train, lm_model.predict(X_train)),
            r2_score(y_train, rf_model.predict(X_train)),
            r2_score(y_train, best_xgb_model.predict(X_train))
        ],
        'CV_Mean_R2': [
            lm_cv.mean(),
            rf_cv.mean(),
            xgb_cv.mean()
        ],
        'CV_Std_R2': [
            lm_cv.std(),
            rf_cv.std(),
            xgb_cv.std()
        ],
        'Test_R2': [
            lm_r2,
            rf_r2,
            xgb_r2
        ],
        'RMSE': [
            lm_rmse,
            rf_rmse,
            xgb_rmse
        ],
        'MAE': [
            lm_mae,
            rf_mae,
            xgb_mae
        ]
    })

    compare_df.to_csv(
        os.path.join(output_dir, "Model_Comparison_Table_S2.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("Saving XGBoost optimal-parameter table...")

    pd.DataFrame([grid_search.best_params_]).to_csv(
        os.path.join(output_dir, "XGBoost_Optimal_Params_Table_S1.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plot_fitting_curve(y_test, xgb_pred, output_dir, analysis_name)

    # --------------------------------------------------------
    # C. Spatial Residuals
    # --------------------------------------------------------
    if HAS_PYSAL and coords_test is not None:
        print("Checking spatial autocorrelation of residuals...")

        test_residuals = y_test - xgb_pred

        try:
            k = min(5, len(coords_test) - 1)

            if k > 0:
                t_coord = coords_test.values

                knn = KNN(t_coord, k=k)
                knn.transform = 'r'

                moran = Moran(test_residuals.values, knn)
                moran_res_str = f"I={moran.I:.3f}, p={moran.p_sim:.3f}"

                res_df = coords_test.copy()
                res_df['Residuals'] = test_residuals

                plt.figure(figsize=(8, 6))

                max_res = max(
                    abs(res_df['Residuals'].min()),
                    abs(res_df['Residuals'].max())
                )

                res_norm = Normalize(vmin=-max_res, vmax=max_res)

                sc = plt.scatter(
                    res_df[coord_cols[0]],
                    res_df[coord_cols[1]],
                    c=res_df['Residuals'],
                    cmap=GLOBAL_CMAP,
                    alpha=0.8,
                    s=30,
                    edgecolor='k',
                    norm=res_norm
                )

                plt.colorbar(sc, label='Residuals')

                plt.title(
                    f"XGBoost Residuals\n{moran_res_str}",
                    fontname='Times New Roman',
                    fontweight='bold',
                    fontsize=16
                )

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "Spatial_Residuals.pdf"))
                plt.close()

                print("Spatial_Residuals.pdf saved.")

        except Exception as e:
            print(f" [Error] Moran\'s I calculation failed: {e}")
    else:
        print(" [Info] libpysal/esda or coordinate columns are unavailable; Spatial_Residuals skipped.")

    # --------------------------------------------------------
    # D. SHAP
    # --------------------------------------------------------
    explainer = shap.TreeExplainer(best_xgb_model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values_matrix = shap_values[0]
    else:
        shap_values_matrix = shap_values

    # 保存 SHAP 值
    pd.DataFrame(
        shap_values_matrix,
        columns=predictors
    ).to_csv(
        os.path.join(output_dir, "SHAP_Values_TestSet.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 阈值分段回归断点检验：输出表
    threshold_df = validate_shap_thresholds(
        shap_values_matrix,
        X_test,
        predictors,
        output_dir,
        analysis_name=analysis_name,
        target_features=TARGET_THRESHOLD_FEATURES,
        n_boot=N_BOOT
    )

    # 阈值诊断图：单独输出，不改原图
    plot_threshold_breakpoint_diagnostics(
        shap_values_matrix=shap_values_matrix,
        X_df=X_test,
        feature_names=predictors,
        threshold_df=threshold_df,
        output_dir=output_dir,
        analysis_name=analysis_name,
        target_features=TARGET_THRESHOLD_FEATURES
    )

    # 原来的 SHAP 图：只换配色，不改排版
    plot_split_shap_visualizations(
        shap_values_matrix,
        X_test,
        predictors,
        output_dir,
        is_urban_scale=is_urban
    )

    plot_shap_dependence_grid(
        shap_values_matrix,
        X_test,
        predictors,
        output_dir,
        group_label=group_label
    )

    try:
        shap_interaction = explainer.shap_interaction_values(X_test)

        plot_new_interaction_matrix(
            shap_interaction,
            X_test,
            predictors,
            output_dir,
            file_tag=analysis_name
        )

    except Exception as e:
        print(f" [Error] SHAP interaction matrix generation failed: {e}")

    result = {
        "analysis_name": analysis_name,
        "model_compare": compare_df,
        "threshold": threshold_df,
        "best_params": grid_search.best_params_
    }

    print(f"<<< Finished analysis: {analysis_name} >>>")

    return result


# ============================================================
# 12. 数据准备
# ============================================================

def parse_ccd_level_value(value):
    """Parse CCD level values from numeric or text labels.

    Supported examples:
    - 1, 2, 3, 4, 5, 6
    - "Level 1", "Level 2"
    - "Level 1-2", "Levels 3-4", "L5_6"
    - "low", "medium", "high"

    For pre-merged labels such as "Level 1-2", the function maps the label to
    the first level of the corresponding group (1, 3, or 5). This allows the
    main grouping rule [1, 2], [3, 4], [5, 6] to work with either detailed CCD
    levels or already reconstructed CCD groups.
    """
    if pd.isna(value):
        return np.nan

    # Numeric values are used directly.
    try:
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            return int(round(numeric_value))
    except Exception:
        pass

    text_value = str(value).strip().lower()
    text_value = (
        text_value
        .replace("–", "-")
        .replace("—", "-")
        .replace("_", "-")
        .replace(" ", "")
    )

    if any(token in text_value for token in ["1-2", "level1-2", "levels1-2", "l1-2", "l1"]):
        return 1
    if any(token in text_value for token in ["3-4", "level3-4", "levels3-4", "l3-4", "l3"]):
        return 3
    if any(token in text_value for token in ["5-6", "level5-6", "levels5-6", "l5-6", "l5"]):
        return 5

    if any(token in text_value for token in ["low", "lower", "低"]):
        return 1
    if any(token in text_value for token in ["medium", "middle", "moderate", "中"]):
        return 3
    if any(token in text_value for token in ["high", "higher", "高"]):
        return 5

    numbers = re.findall(r"\d+", text_value)
    if numbers:
        return int(numbers[0])

    return np.nan


def prepare_data(data):
    data = standardize_columns(data)

    if "year" in data.columns and "Year" not in data.columns:
        data.rename(columns={"year": "Year"}, inplace=True)

    numeric_cols = [
        "Year",
        "temperature",
        "lon",
        "lat",
        "PD",
        "ED",
        "LSI",
        "AWMSI",
        "AI",
        "CONTAG",
        "SHDI",
        "FP",
        "GP",
        "WP",
        "FVC",
        "ECV",
        "BP",
        "POP",
        "GDP",
        "UEI"
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "ccd_level" not in data.columns:
        raise ValueError(
            "数据中缺少 CCD 等级列。请检查字段名是否为 "
            "'Coupling Coordination Level'、'Coupling_Coordination_Level' 或 'ccd_level'。"
        )

    if "temperature" not in data.columns:
        raise ValueError(
            "数据中缺少温度列。请检查字段名是否为 "
            "'Temperature'、'LST' 或 'temperature'。"
        )

    data["ccd_level"] = data["ccd_level"].apply(parse_ccd_level_value)
    data = data.dropna(subset=["ccd_level", "temperature"]).copy()
    data["ccd_level"] = data["ccd_level"].round().astype(int)

    return data


def load_input_data(input_mode, input_path=None, input_dir=None, year_files=None):
    if input_mode == "single":
        if input_path is None or not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到输入文件：{input_path}")

        df = read_csv_smart(input_path)
        df = standardize_columns(df)

        if "Year" not in df.columns:
            m = re.search(
                r"(2003|2008|2013|2018|2019|2023)",
                os.path.basename(input_path)
            )

            if m:
                df["Year"] = int(m.group(1))

        return df

    if input_mode == "multi":
        if input_dir is None:
            raise ValueError("input_mode='multi' 时必须设置 input_dir")

        if year_files is None:
            year_files = {
                2003: "2003.csv",
                2008: "2008.csv",
                2013: "2013.csv",
                2018: "2018.csv",
                2023: "2023.csv"
            }

        frames = []

        for year, filename in year_files.items():
            path = os.path.join(input_dir, filename)

            if not os.path.exists(path):
                print(f" [Warning] Year file not found and skipped: {year}, {path}")
                continue

            df = read_csv_smart(path)
            df = standardize_columns(df)

            if "Year" not in df.columns:
                df["Year"] = int(year)

            frames.append(df)

        if not frames:
            raise FileNotFoundError("没有读取到任何年份 CSV，请检查 input_dir 和 year_files。")

        return pd.concat(frames, ignore_index=True)

    raise ValueError("input_mode 只能是 'single' 或 'multi'")


# ============================================================
# 13. 主程序入口
# ============================================================

def main():
    """Run the complete XGBoost-SHAP workflow with repository-relative paths."""
    args = parse_args()

    # ------------------------------------------------------------------
    # Input/output configuration
    # ------------------------------------------------------------------
    input_mode = args.mode
    input_path = Path(args.input).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_base = Path(args.output).expanduser().resolve()
    process_by_year = not args.no_process_by_year

    year_files = {
        2003: "2003.csv",
        2008: "2008.csv",
        2013: "2013.csv",
        2018: "2018.csv",
        2023: "2023.csv",
    }

    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Zhengzhou LST XGBoost-SHAP analysis")
    print("=" * 72)
    print(f"Repository root : {REPO_ROOT}")
    print(f"Input mode      : {input_mode}")
    print(f"Input CSV       : {input_path}")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_base}")
    print("=" * 72)

    data = load_input_data(
        input_mode=input_mode,
        input_path=str(input_path),
        input_dir=str(input_dir),
        year_files=year_files,
    )
    data = prepare_data(data)

    # Blue-green space predictors
    bg_vars = [
        "PD",
        "ED",
        "LSI",
        "AWMSI",
        "AI",
        "CONTAG",
        "SHDI",
        "FP",
        "GP",
        "WP",
        "FVC",
        "ECV",
    ]

    # Urbanization predictors
    urban_vars = ["BP", "POP", "GDP", "UEI"]

    predictors = bg_vars + urban_vars
    target_col = "temperature"
    coord_cols = ["lon", "lat"]

    # Three reconstructed CCD gradients used in the manuscript/appendix.
    level_groups = {
        "Level_1_2": {"levels": [1, 2], "label": "Levels 1-2"},
        "Level_3_4": {"levels": [3, 4], "label": "Levels 3-4"},
        "Level_5_6": {"levels": [5, 6], "label": "Levels 5-6"},
    }

    # To run the original six CCD levels separately, replace level_groups with:
    # level_groups = {
    #     "Level_1": {"levels": [1], "label": "Level 1"},
    #     "Level_2": {"levels": [2], "label": "Level 2"},
    #     "Level_3": {"levels": [3], "label": "Level 3"},
    #     "Level_4": {"levels": [4], "label": "Level 4"},
    #     "Level_5": {"levels": [5], "label": "Level 5"},
    #     "Level_6": {"levels": [6], "label": "Level 6"},
    # }

    all_model_tables = []
    all_threshold_tables = []

    if process_by_year and "Year" in data.columns:
        years = sorted(int(y) for y in data["Year"].dropna().unique())
    else:
        years = ["All"]

    for year in years:
        if year == "All":
            year_data = data.copy()
            year_dir = output_base
            year_label = "AllYears"
        else:
            year_data = data[data["Year"] == year].copy()
            year_dir = output_base / str(year)
            year_label = str(year)

        year_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"Year: {year_label}, n = {len(year_data)}")
        print("=" * 72)

        for group_name, group_info in level_groups.items():
            current_levels = group_info["levels"]
            group_label = group_info["label"]

            group_data = year_data[
                year_data["ccd_level"].isin(current_levels)
            ].copy()

            print(
                f"\n>>> Processing {year_label} - {group_label}: "
                f"n = {len(group_data)} <<<"
            )

            if len(group_data) < 20:
                print(f" [Skip] {year_label} - {group_label}: insufficient samples")
                continue

            group_dir = year_dir / group_name / "Combined"
            group_dir.mkdir(parents=True, exist_ok=True)

            analysis_name = f"{year_label}_{group_name}_Combined"

            result = run_advanced_analysis(
                data=group_data,
                predictors=predictors,
                target_col=target_col,
                coord_cols=coord_cols,
                output_dir=str(group_dir),
                analysis_name=analysis_name,
                is_urban=False,
                group_label=group_label,
            )

            if result is None:
                continue

            model_table = result["model_compare"].copy()
            model_table.insert(0, "Year", year_label)
            model_table.insert(1, "CCD_Group", group_label)
            all_model_tables.append(model_table)

            threshold_table = result["threshold"].copy()
            threshold_table.insert(0, "Year", year_label)
            threshold_table.insert(1, "CCD_Group", group_label)
            all_threshold_tables.append(threshold_table)

    if all_model_tables:
        all_model_df = pd.concat(all_model_tables, ignore_index=True)
        out_model = output_base / "ALL_Model_Comparison_Summary.csv"
        all_model_df.to_csv(out_model, index=False, encoding="utf-8-sig")
        print(f"\n✓ Model-performance summary saved: {out_model}")

    if all_threshold_tables:
        all_threshold_df = pd.concat(all_threshold_tables, ignore_index=True)
        out_threshold = output_base / "ALL_SHAP_Threshold_Breakpoint_Test_Summary.csv"
        all_threshold_df.to_csv(out_threshold, index=False, encoding="utf-8-sig")
        print(f"✓ Breakpoint-validation summary saved: {out_threshold}")

    print("\nAll analysis tasks finished.")

if __name__ == "__main__":
    main()

