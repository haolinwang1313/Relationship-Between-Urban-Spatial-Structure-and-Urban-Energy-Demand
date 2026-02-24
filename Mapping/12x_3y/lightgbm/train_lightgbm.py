#!/usr/bin/env python3
"""Train LightGBM surrogate models with 8 urban-form features."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "lightgbm is not installed; run 'pip install lightgbm' in .venv_geo."  # noqa: E501
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "Mapping" / "12x_3y" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Mapping" / "12x_3y" / "lightgbm"

FEATURE_COLUMNS = [
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

TARGET_COLUMNS = [
    "cooling_kwh_per_m2",
    "heating_kwh_per_m2",
    "other_electricity_kwh_per_m2",
]

N_SPLITS = 5
SEED = 42


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    missing = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[dict, lgb.Booster]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics = []
    best_iterations = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"  Fold {fold_idx}/{N_SPLITS}...")
        train_set = lgb.Dataset(X[train_idx], label=y[train_idx])
        val_set = lgb.Dataset(X[val_idx], label=y[val_idx], reference=train_set)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 30,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": SEED,
            "num_threads": 1,
        }
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=600,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        best_iterations.append(booster.best_iteration or 600)
        preds = booster.predict(X[val_idx])
        mse = mean_squared_error(y[val_idx], preds)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y[val_idx], preds)
        r2 = r2_score(y[val_idx], preds)
        fold_metrics.append({"rmse": rmse, "mae": mae, "r2": r2})
    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "r2": float(np.mean([m["r2"] for m in fold_metrics])),
    }
    full_set = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 30,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": 1,
    }
    booster = lgb.train(params, full_set, num_boost_round=int(np.mean(best_iterations)))
    return avg_metrics, booster


def save_importance(model: lgb.Booster, target: str) -> None:
    imp = model.feature_importance(importance_type="gain")
    df_imp = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_gain": imp})
    df_imp = df_imp.sort_values("importance_gain", ascending=False)
    df_imp.to_csv(OUTPUT_DIR / f"lgbm_feature_importance_{target}.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    X = data[FEATURE_COLUMNS].values
    results = {}
    for target in TARGET_COLUMNS:
        print(f"\n=== Training target: {target} ===")
        y = data[target].values
        metrics, booster = train_model(X, y)
        results[target] = metrics
        booster.save_model(str(OUTPUT_DIR / f"lgbm_{target}.txt"))
        save_importance(booster, target)
    (OUTPUT_DIR / "lgbm_metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved metrics to", OUTPUT_DIR / "lgbm_metrics.json")


if __name__ == "__main__":
    main()
