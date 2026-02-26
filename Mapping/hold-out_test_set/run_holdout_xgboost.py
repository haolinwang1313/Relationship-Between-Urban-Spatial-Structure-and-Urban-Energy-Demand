                      
"""Run hold-out experiments for three feature groups and three targets."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import xgboost as xgb
except ImportError as exc:                                               
    raise SystemExit(
        "xgboost is not installed in the current environment. "
        "Install it via 'pip install xgboost' inside .venv_geo before running this script."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "Mapping" / "hold-out_test_set"

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

BUILT_TRANSPORT_FEATURES = [
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

TARGET_COLUMNS = {
    "cooling": "cooling_kwh_per_m2",
    "heating": "heating_kwh_per_m2",
    "other": "other_electricity_kwh_per_m2",
}

FEATURE_GROUPS = {
    "Form-only": {
        "dataset_path": PROJECT_ROOT / "Mapping" / "8x_3y" / "dataset.csv",
        "feature_columns": FORM_FEATURES,
        "colsample_bytree": 0.9,
    },
    "Built+transport": {
        "dataset_path": PROJECT_ROOT / "Mapping" / "12x_3y" / "dataset.csv",
        "feature_columns": BUILT_TRANSPORT_FEATURES,
        "colsample_bytree": 0.8,
    },
    "All": {
        "dataset_path": PROJECT_ROOT / "Mapping" / "20x_3y" / "xgboost" / "dataset.csv",
        "feature_columns": FORM_FEATURES + BUILT_TRANSPORT_FEATURES,
        "colsample_bytree": 0.8,
    },
}

SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5

CV_MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "min_child_weight": 1.0,
    "random_state": SEED,
    "tree_method": "hist",
    "n_jobs": 1,
}

FINAL_MODEL_PARAMS = {
    **CV_MODEL_PARAMS,
    "n_estimators": 600,
}


def load_dataset(path: Path, features: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = features + list(TARGET_COLUMNS.values())
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")
    return df


def run_one_target(
    df: pd.DataFrame, features: list[str], target_col: str, colsample_bytree: float
) -> dict[str, float]:
    X = df[features].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_r2_scores: list[float] = []

    for train_idx, val_idx in kf.split(X_train):
        model = xgb.XGBRegressor(**CV_MODEL_PARAMS, colsample_bytree=colsample_bytree)
        model.fit(X_train[train_idx], y_train[train_idx])
        y_val_pred = model.predict(X_train[val_idx])
        cv_r2_scores.append(float(r2_score(y_train[val_idx], y_val_pred)))

    final_model = xgb.XGBRegressor(**FINAL_MODEL_PARAMS, colsample_bytree=colsample_bytree)
    final_model.fit(X_train, y_train)

    y_test_pred = final_model.predict(X_test)
    y_train_pred = final_model.predict(X_train)

    return {
        "cv_r2_mean": float(np.mean(cv_r2_scores)),
        "cv_r2_std": float(np.std(cv_r2_scores, ddof=0)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    for model_name, group_cfg in FEATURE_GROUPS.items():
        dataset_path = Path(group_cfg["dataset_path"])
        feature_columns = list(group_cfg["feature_columns"])
        colsample_bytree = float(group_cfg["colsample_bytree"])
        df = load_dataset(dataset_path, feature_columns)

        print(f"\n=== Feature group: {model_name} ===")
        for target_name, target_col in TARGET_COLUMNS.items():
            print(f"  Running target: {target_name}")
            metrics = run_one_target(df, feature_columns, target_col, colsample_bytree)
            results.append(
                {
                    "target": target_name,
                    "target_column": target_col,
                    "model": model_name,
                    "cv_r2": metrics["cv_r2_mean"],
                    "cv_r2_std": metrics["cv_r2_std"],
                    "test_r2": metrics["test_r2"],
                    "train_r2": metrics["train_r2"],
                    "n_train": metrics["n_train"],
                    "n_test": metrics["n_test"],
                    "test_size": TEST_SIZE,
                    "random_seed": SEED,
                    "cv_folds": N_SPLITS,
                    "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
                }
            )

    df_long = pd.DataFrame(results).sort_values(["target", "model"]).reset_index(drop=True)
    long_path = OUTPUT_DIR / "holdout_xgboost_r2_long.csv"
    df_long.to_csv(long_path, index=False)

    df_target_table = (
        df_long.loc[:, ["target", "model", "cv_r2", "test_r2"]]
        .sort_values(["target", "model"])
        .reset_index(drop=True)
    )
    target_table_path = OUTPUT_DIR / "holdout_xgboost_r2_by_target.csv"
    df_target_table.to_csv(target_table_path, index=False)

    metadata = {
        "split": {
            "method": "train_test_split",
            "test_size": TEST_SIZE,
            "random_state": SEED,
        },
        "cv": {
            "method": "KFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
            "applied_on": "X_train only",
        },
        "xgboost_cv_params": CV_MODEL_PARAMS,
        "xgboost_final_params": FINAL_MODEL_PARAMS,
        "feature_groups": {
            name: {
                "dataset_path": str(Path(cfg["dataset_path"]).relative_to(PROJECT_ROOT)),
                "n_features": len(cfg["feature_columns"]),
                "colsample_bytree": cfg["colsample_bytree"],
            }
            for name, cfg in FEATURE_GROUPS.items()
        },
    }
    meta_path = OUTPUT_DIR / "holdout_xgboost_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved long results to: {long_path}")
    print(f"Saved target table to: {target_table_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
