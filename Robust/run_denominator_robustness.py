                      
"""Denominator sensitivity robustness check (land-normalized vs floor-normalized EUI).

This script builds two matched datasets on the same grid sample (FA > 0):
1) Land-normalized targets (existing definition): E_k / A_grid
2) Floor-normalized targets (control): E_k / FA_grid

Then it trains/evaluates:
- XGBoost (same settings as Mapping/20x_3y/xgboost/train_xgboost.py)
- OLS baseline (same CV split for comparability)

Finally it exports:
- Performance comparison tables
- SHAP group-share summaries (Form/Built/Transport)
- VCI SHAP dependence plots and threshold estimates
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import xgboost as xgb

os.environ.setdefault("OMP_NUM_THREADS", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBUST_DIR = PROJECT_ROOT / "Robust"
MAPPING_DIR = PROJECT_ROOT / "Mapping"

SOURCE_DATASET = MAPPING_DIR / "20x_3y" / "xgboost" / "dataset.csv"
BUILT_ENV_CSV = MAPPING_DIR / "built_env_250m.csv"
ENERGY_CSV = MAPPING_DIR / "energy_250m.csv"
ORIGINAL_XGB_METRICS = MAPPING_DIR / "20x_3y" / "xgboost" / "xgb_metrics.json"
ORIGINAL_SHAP_GROUP = MAPPING_DIR / "xai" / "Allmodel" / "shap_group_importance.csv"

DATA_DIR = ROBUST_DIR / "data"
MODELS_DIR = ROBUST_DIR / "models"
RESULTS_DIR = ROBUST_DIR / "results"
FIGURES_DIR = ROBUST_DIR / "figures"

FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "TimesNewRoman.ttf"
if FONT_PATH.exists():
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(FONT_PATH))

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TimesNewRoman"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    }
)

FEATURE_COLUMNS = [
    "ci_norm",
    "vci_norm",
    "lum_norm",
    "lum_adjacency_norm",
    "lum_intensity_norm",
    "lum_proximity_norm",
    "gi_norm",
    "li_norm",
    "single_family_ha",
    "multi_family_ha",
    "facility_neighborhood_ha",
    "facility_sales_ha",
    "facility_office_ha",
    "facility_education_ha",
    "facility_industrial_ha",
    "parks_green_ha",
    "water_area_ha",
    "road_area_ha",
    "subway_influence_ha",
    "bus_routes_cnt",
]

FORM_FEATURES = [
    "ci_norm",
    "vci_norm",
    "lum_norm",
    "lum_adjacency_norm",
    "lum_intensity_norm",
    "lum_proximity_norm",
    "gi_norm",
    "li_norm",
]

BUILT_FEATURES = [
    "single_family_ha",
    "multi_family_ha",
    "facility_neighborhood_ha",
    "facility_sales_ha",
    "facility_office_ha",
    "facility_education_ha",
    "facility_industrial_ha",
    "parks_green_ha",
    "water_area_ha",
    "road_area_ha",
]

TRANSPORT_FEATURES = [
    "subway_influence_ha",
    "bus_routes_cnt",
]

TARGET_SPECS: Dict[str, Dict[str, str]] = {
    "cooling": {
        "label": "Cooling load",
        "land_col": "cooling_kwh_per_m2",
        "energy_col": "cooling_kwh",
        "floor_col": "cooling_kwh_per_floor_m2",
    },
    "heating": {
        "label": "Heating load",
        "land_col": "heating_kwh_per_m2",
        "energy_col": "heating_kwh",
        "floor_col": "heating_kwh_per_floor_m2",
    },
    "other_electricity": {
        "label": "Other electricity",
        "land_col": "other_electricity_kwh_per_m2",
        "energy_col": "other_electricity_kwh",
        "floor_col": "other_electricity_kwh_per_floor_m2",
    },
}

N_SPLITS = 5
SEED = 42
VCI_BINS = 20


def ensure_dirs() -> None:
    for p in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    for norm in ["land", "floor"]:
        (MODELS_DIR / norm).mkdir(parents=True, exist_ok=True)
        (FIGURES_DIR / norm).mkdir(parents=True, exist_ok=True)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_folds(n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    dummy = np.zeros((n_samples, 1))
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(dummy)]


def build_matched_datasets() -> pd.DataFrame:
    base = pd.read_csv(SOURCE_DATASET)
    land_cols = [spec["land_col"] for spec in TARGET_SPECS.values()]
    needed = ["grid_id_main", *FEATURE_COLUMNS, *land_cols]
    missing = [c for c in needed if c not in base.columns]
    if missing:
        raise ValueError(f"Source dataset missing columns: {missing}")
    base = base[needed].copy()

    built = pd.read_csv(BUILT_ENV_CSV, usecols=["grid_id_main", "floor_area_total_m2"])
    energy_cols = ["grid_id_main", *[spec["energy_col"] for spec in TARGET_SPECS.values()]]
    energy = pd.read_csv(ENERGY_CSV, usecols=energy_cols)

    df = (
        base.merge(built, on="grid_id_main", how="left")
        .merge(energy, on="grid_id_main", how="left")
        .sort_values("grid_id_main")
        .reset_index(drop=True)
    )

    if df["floor_area_total_m2"].isna().any():
        raise ValueError("Found NaN floor area values after merge.")

                                                                    
    df = df[df["floor_area_total_m2"] > 0].copy()

    for _, spec in TARGET_SPECS.items():
        df[spec["floor_col"]] = df[spec["energy_col"]] / df["floor_area_total_m2"]

    check_cols = (
        FEATURE_COLUMNS
        + [spec["land_col"] for spec in TARGET_SPECS.values()]
        + [spec["floor_col"] for spec in TARGET_SPECS.values()]
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=check_cols).reset_index(drop=True)

    land_out = df[["grid_id_main", *FEATURE_COLUMNS, *[spec["land_col"] for spec in TARGET_SPECS.values()]]]
    floor_out = df[["grid_id_main", *FEATURE_COLUMNS, *[spec["floor_col"] for spec in TARGET_SPECS.values()]]]
    land_out.to_csv(DATA_DIR / "dataset_land_filtered.csv", index=False)
    floor_out.to_csv(DATA_DIR / "dataset_floor_filtered.csv", index=False)

    summary = {
        "rows_after_filter": int(len(df)),
        "rows_with_fa_gt_0": int(len(df)),
        "feature_count": len(FEATURE_COLUMNS),
        "fa_min": float(df["floor_area_total_m2"].min()),
        "fa_max": float(df["floor_area_total_m2"].max()),
        "fa_median": float(df["floor_area_total_m2"].median()),
    }
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return df


def train_xgb_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> tuple[Dict[str, float], pd.DataFrame, xgb.XGBRegressor]:
    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            random_state=SEED,
            tree_method="hist",
            n_jobs=1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        m = metric_dict(y[val_idx], preds)
        fold_rows.append({"fold": fold_idx, **m})

    fold_df = pd.DataFrame(fold_rows)
    avg = {
        "rmse": float(fold_df["rmse"].mean()),
        "mae": float(fold_df["mae"].mean()),
        "r2": float(fold_df["r2"].mean()),
    }

    final_model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
    )
    final_model.fit(X, y)
    return avg, fold_df, final_model


def train_ols_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> tuple[Dict[str, float], pd.DataFrame]:
    rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        m = metric_dict(y[val_idx], preds)
        rows.append({"fold": fold_idx, **m})
    fold_df = pd.DataFrame(rows)
    avg = {
        "rmse": float(fold_df["rmse"].mean()),
        "mae": float(fold_df["mae"].mean()),
        "r2": float(fold_df["r2"].mean()),
    }
    return avg, fold_df


def compute_shap(booster: xgb.Booster, X_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dmatrix = xgb.DMatrix(X_df, feature_names=list(X_df.columns))
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    shap_values = shap_matrix[:, :-1]
    base_values = shap_matrix[:, -1]
    preds = booster.predict(dmatrix)
    return shap_values, base_values, preds


def estimate_threshold_from_bins(x: np.ndarray, y: np.ndarray, bins: int = VCI_BINS) -> tuple[float, pd.DataFrame]:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if math.isclose(x_min, x_max):
        return float("nan"), pd.DataFrame(columns=["center", "mean_shap", "count"])
    edges = np.linspace(x_min, x_max, bins + 1)
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    rows = []
    for i in range(bins):
        mask = idx == i
        if not np.any(mask):
            continue
        rows.append(
            {
                "center": float((edges[i] + edges[i + 1]) / 2.0),
                "mean_shap": float(np.mean(y[mask])),
                "count": int(np.sum(mask)),
            }
        )
    bin_df = pd.DataFrame(rows).sort_values("center").reset_index(drop=True)
    if len(bin_df) < 9:
        return float("nan"), bin_df

                                                                         
    centers = bin_df["center"].to_numpy(dtype=np.float64)
    means = bin_df["mean_shap"].to_numpy(dtype=np.float64)
    min_seg = 4
    best_sse = None
    best_center = float("nan")
    for i in range(min_seg, len(bin_df) - min_seg):
        x1, y1 = centers[: i + 1], means[: i + 1]
        x2, y2 = centers[i:], means[i:]
        p1 = np.polyfit(x1, y1, 1)
        p2 = np.polyfit(x2, y2, 1)
        sse1 = float(np.sum((y1 - (p1[0] * x1 + p1[1])) ** 2))
        sse2 = float(np.sum((y2 - (p2[0] * x2 + p2[1])) ** 2))
        total_sse = sse1 + sse2
        if best_sse is None or total_sse < best_sse:
            best_sse = total_sse
            best_center = float(centers[i])
    threshold = best_center
    return threshold, bin_df


def plot_vci_dependence(
    out_path: Path,
    target_label: str,
    norm_label: str,
    vci_values: np.ndarray,
    shap_values: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    scatter = ax.scatter(
        vci_values,
        shap_values,
        c=vci_values,
        cmap="viridis",
        s=18,
        alpha=0.5,
        linewidths=0,
    )

    _, bin_df = estimate_threshold_from_bins(vci_values, shap_values, bins=VCI_BINS)
    if not bin_df.empty:
        ax.plot(
            bin_df["center"].to_numpy(),
            bin_df["mean_shap"].to_numpy(),
            color="#C00000",
            linewidth=1.5,
            marker="o",
            markersize=3,
            label="Binned mean",
        )
        ax.legend()

    ax.set_xlabel("vci")
    ax.set_ylabel("SHAP value")
    ax.set_title(f"{target_label} vs vci ({norm_label})", fontweight="bold", pad=6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.ax.set_ylabel("vci")
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def normalization_target_vector(df: pd.DataFrame, normalization: str, target_key: str) -> np.ndarray:
    spec = TARGET_SPECS[target_key]
    col = spec["land_col"] if normalization == "land" else spec["floor_col"]
    return df[col].to_numpy(dtype=np.float64)


def run_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_df = df[FEATURE_COLUMNS].copy()
    X = X_df.to_numpy(dtype=np.float64)
    folds = make_folds(len(df))
    vci_idx = FEATURE_COLUMNS.index("vci_norm")

    performance_rows = []
    shap_group_rows = []
    threshold_rows = []

    for normalization in ["land", "floor"]:
        norm_models_dir = MODELS_DIR / normalization
        norm_fig_dir = FIGURES_DIR / normalization
        global_rows = []

        for target_key, spec in TARGET_SPECS.items():
            y = normalization_target_vector(df, normalization, target_key)

            xgb_avg, xgb_fold_df, xgb_model = train_xgb_with_cv(X, y, folds)
            xgb_fold_df.to_csv(norm_models_dir / f"xgb_cv_folds_{target_key}.csv", index=False)
            xgb_model.save_model(str(norm_models_dir / f"xgb_{target_key}.json"))

            ols_avg, ols_fold_df = train_ols_with_cv(X, y, folds)
            ols_fold_df.to_csv(norm_models_dir / f"ols_cv_folds_{target_key}.csv", index=False)

            performance_rows.append(
                {
                    "normalization": normalization,
                    "model": "xgboost",
                    "target_key": target_key,
                    "target": spec["label"],
                    **xgb_avg,
                }
            )
            performance_rows.append(
                {
                    "normalization": normalization,
                    "model": "ols",
                    "target_key": target_key,
                    "target": spec["label"],
                    **ols_avg,
                }
            )

            booster = xgb_model.get_booster()
            shap_values, base_values, preds = compute_shap(booster, X_df)
            residual = float(np.max(np.abs(shap_values.sum(axis=1) + base_values - preds)))
            if residual > 5e-4:
                raise RuntimeError(
                    f"SHAP decomposition mismatch for {normalization}/{target_key}: {residual:.3e}"
                )

            np.save(norm_models_dir / f"shap_values_{target_key}.npy", shap_values)

            mean_abs = np.abs(shap_values).mean(axis=0)
            total = float(mean_abs.sum())
            mean_abs_series = pd.Series(mean_abs, index=FEATURE_COLUMNS)

            form_sum = float(mean_abs_series[FORM_FEATURES].sum())
            built_sum = float(mean_abs_series[BUILT_FEATURES].sum())
            transport_sum = float(mean_abs_series[TRANSPORT_FEATURES].sum())

            shap_group_rows.append(
                {
                    "normalization": normalization,
                    "target_key": target_key,
                    "target": spec["label"],
                    "Form": form_sum / total if total else 0.0,
                    "Built": built_sum / total if total else 0.0,
                    "Transport": transport_sum / total if total else 0.0,
                }
            )

            for feat, val in mean_abs_series.items():
                global_rows.append(
                    {
                        "normalization": normalization,
                        "target_key": target_key,
                        "target": spec["label"],
                        "feature": feat,
                        "mean_abs_shap": float(val),
                        "share": float(val / total) if total else 0.0,
                        "group": (
                            "Form"
                            if feat in FORM_FEATURES
                            else "Built"
                            if feat in BUILT_FEATURES
                            else "Transport"
                        ),
                    }
                )

            vci_values = X_df["vci_norm"].to_numpy(dtype=np.float64)
            vci_shap = shap_values[:, vci_idx]
            threshold, bin_df = estimate_threshold_from_bins(vci_values, vci_shap, bins=VCI_BINS)
            threshold_rows.append(
                {
                    "normalization": normalization,
                    "target_key": target_key,
                    "target": spec["label"],
                    "vci_threshold_estimate": threshold,
                }
            )
            bin_df.to_csv(
                RESULTS_DIR / f"vci_bins_{normalization}_{target_key}.csv",
                index=False,
            )
            plot_vci_dependence(
                norm_fig_dir / f"vci_dependence_{target_key}.png",
                spec["label"],
                normalization,
                vci_values,
                vci_shap,
            )

        pd.DataFrame(global_rows).to_csv(
            RESULTS_DIR / f"shap_global_importance_{normalization}.csv",
            index=False,
        )

    perf_df = pd.DataFrame(performance_rows).sort_values(
        ["model", "target_key", "normalization"]
    )
    shap_group_df = pd.DataFrame(shap_group_rows).sort_values(
        ["target_key", "normalization"]
    )
    thresholds_df = pd.DataFrame(threshold_rows).sort_values(
        ["target_key", "normalization"]
    )
    return perf_df, shap_group_df, thresholds_df


def make_summary_tables(perf_df: pd.DataFrame, shap_group_df: pd.DataFrame) -> None:
    perf_df.to_csv(RESULTS_DIR / "performance_all_models.csv", index=False)
    shap_group_df.to_csv(RESULTS_DIR / "shap_group_share_land_vs_floor.csv", index=False)

    xgb_df = perf_df[perf_df["model"] == "xgboost"].copy()
    pivot = (
        xgb_df.pivot_table(
            index=["target_key", "target"],
            columns="normalization",
            values=["r2", "rmse", "mae"],
        )
        .sort_index(axis=1)
        .reset_index()
    )
    pivot.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in pivot.columns.to_flat_index()
    ]

    with ORIGINAL_XGB_METRICS.open("r", encoding="utf-8") as f:
        original = json.load(f)

    key_to_land_col = {k: spec["land_col"] for k, spec in TARGET_SPECS.items()}
    original_rows = []
    for target_key, land_col in key_to_land_col.items():
        m = original[land_col]
        original_rows.append(
            {
                "target_key": target_key,
                "r2_original_land_full": float(m["r2"]),
                "rmse_original_land_full": float(m["rmse"]),
                "mae_original_land_full": float(m["mae"]),
            }
        )
    original_df = pd.DataFrame(original_rows)

    summary = pivot.merge(original_df, on="target_key", how="left")
    summary["delta_r2_floor_vs_land_filtered"] = summary["r2_floor"] - summary["r2_land"]
    summary["delta_r2_floor_vs_original_land_full"] = (
        summary["r2_floor"] - summary["r2_original_land_full"]
    )
    summary["delta_rmse_floor_vs_land_filtered"] = summary["rmse_floor"] - summary["rmse_land"]
    summary["delta_rmse_floor_vs_original_land_full"] = (
        summary["rmse_floor"] - summary["rmse_original_land_full"]
    )
    summary.to_csv(RESULTS_DIR / "performance_summary_xgboost.csv", index=False)

                                                                         
    original_shap = pd.read_csv(ORIGINAL_SHAP_GROUP)
    target_map = {
        "Cooling load": "cooling",
        "Heating load": "heating",
        "Other electricity": "other_electricity",
    }
    original_form = (
        original_shap[original_shap["group"] == "Form"][["target", "share"]]
        .assign(target_key=lambda d: d["target"].map(target_map))
        .dropna(subset=["target_key"])
        .rename(columns={"share": "Form_original_land_full"})
        [["target_key", "Form_original_land_full"]]
    )

    group_cmp = shap_group_df.pivot_table(
        index=["target_key", "target"],
        columns="normalization",
        values=["Form", "Built", "Transport"],
    ).reset_index()
    group_cmp.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in group_cmp.columns.to_flat_index()
    ]
    group_cmp = group_cmp.merge(original_form, on="target_key", how="left")
    group_cmp["delta_Form_floor_vs_land_filtered"] = (
        group_cmp["Form_floor"] - group_cmp["Form_land"]
    )
    group_cmp["delta_Form_floor_vs_original_land_full"] = (
        group_cmp["Form_floor"] - group_cmp["Form_original_land_full"]
    )
    group_cmp.to_csv(RESULTS_DIR / "morphology_share_comparison.csv", index=False)


def main() -> None:
    ensure_dirs()
    df = build_matched_datasets()
    perf_df, shap_group_df, thresholds_df = run_models(df)
    thresholds_df.to_csv(RESULTS_DIR / "vci_threshold_estimates.csv", index=False)
    make_summary_tables(perf_df, shap_group_df)
    print(f"Robustness outputs saved under: {ROBUST_DIR}")


if __name__ == "__main__":
    main()
