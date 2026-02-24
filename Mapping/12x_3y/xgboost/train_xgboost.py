#!/usr/bin/env python3
"""Train XGBoost surrogate models for cooling/heating/other electricity demand."""

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
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "xgboost is not installed in the current environment. Install it via 'pip install xgboost' "
        "inside .venv_geo before running this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "Mapping" / "12x_3y" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Mapping" / "12x_3y" / "xgboost"

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
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}. Run the preparation step first.")
    df = pd.read_csv(DATASET_PATH)
    missing_features = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing expected columns: {missing_features}")
    return df


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[dict, xgb.XGBRegressor]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics: list[dict[str, float]] = []
    models: list[xgb.XGBRegressor] = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"  Training fold {fold_idx}/{N_SPLITS} for current target...")
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
        mse = mean_squared_error(y[val_idx], preds)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y[val_idx], preds)
        r2 = r2_score(y[val_idx], preds)
        fold_metrics.append({"rmse": rmse, "mae": mae, "r2": r2})
        models.append(model)
    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "r2": float(np.mean([m["r2"] for m in fold_metrics])),
    }
    # retrain final model on all data with slightly more estimators for stability
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
    return avg_metrics, final_model


def save_feature_importance(model: xgb.XGBRegressor, target_name: str) -> None:
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")
    records = [
        {"feature": feat, "importance_gain": float(scores.get(f"f{idx}", 0.0))}
        for idx, feat in enumerate(FEATURE_COLUMNS)
    ]
    df_imp = pd.DataFrame(records).sort_values("importance_gain", ascending=False)
    out_path = OUTPUT_DIR / f"xgb_feature_importance_{target_name}.csv"
    df_imp.to_csv(out_path, index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    X = data[FEATURE_COLUMNS].values
    results = {}
    for target in TARGET_COLUMNS:
        print(f"\n=== Training target: {target} ===")
        y = data[target].values
        metrics, final_model = train_model(X, y)
        results[target] = metrics
        model_path = OUTPUT_DIR / f"xgb_{target}.json"
        final_model.save_model(model_path)
        save_feature_importance(final_model, target)
    metrics_path = OUTPUT_DIR / "xgb_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
