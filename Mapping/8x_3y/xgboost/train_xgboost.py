                      
"""Train XGBoost models using 8 urban-form features for 3 energy targets."""

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
except ImportError as exc:                    
    raise SystemExit(
        "xgboost is not installed. Run 'pip install xgboost' inside .venv_geo first."              
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "Mapping" / "8x_3y" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Mapping" / "8x_3y" / "xgboost"

FEATURE_COLUMNS = [
    "ci_norm",
    "vci_norm",
    "lum_norm",
    "lum_adjacency_norm",
    "lum_intensity_norm",
    "lum_proximity_norm",
    "gi_norm",
    "li_norm",
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
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def train_model(X: np.ndarray, y: np.ndarray) -> tuple[dict, xgb.XGBRegressor]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"  Fold {fold_idx}/{N_SPLITS}...")
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
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
        metrics.append({"rmse": rmse, "mae": mae, "r2": r2})
    avg_metrics = {
        "rmse": float(np.mean([m["rmse"] for m in metrics])),
        "mae": float(np.mean([m["mae"] for m in metrics])),
        "r2": float(np.mean([m["r2"] for m in metrics])),
    }
    final_model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
    )
    final_model.fit(X, y)
    return avg_metrics, final_model


def save_importance(model: xgb.XGBRegressor, target: str) -> None:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    rows = [
        {"feature": feat, "importance_gain": float(gain.get(f"f{i}", 0.0))}
        for i, feat in enumerate(FEATURE_COLUMNS)
    ]
    df_imp = pd.DataFrame(rows).sort_values("importance_gain", ascending=False)
    df_imp.to_csv(OUTPUT_DIR / f"xgb_feature_importance_{target}.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    X = data[FEATURE_COLUMNS].values
    results = {}
    for target in TARGET_COLUMNS:
        print(f"\n=== Training target: {target} ===")
        y = data[target].values
        metrics, model = train_model(X, y)
        results[target] = metrics
        model.save_model(str(OUTPUT_DIR / f"xgb_{target}.json"))
        save_importance(model, target)
    (OUTPUT_DIR / "xgb_metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved metrics to", OUTPUT_DIR / "xgb_metrics.json")


if __name__ == "__main__":
    main()

