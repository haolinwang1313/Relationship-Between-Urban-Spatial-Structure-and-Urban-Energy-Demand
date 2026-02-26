                      
"""OLS baseline on morphology features with fixed hold-out split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "Mapping" / "8x_3y" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "Mapping" / "OLS"
OUTPUT_CSV = OUTPUT_DIR / "ols_morphology_test_r2.csv"

MORPHOLOGY_FEATURES = [
    "ci_norm",
    "vci_norm",
    "lum_norm",
    "lum_adjacency_norm",
    "lum_intensity_norm",
    "lum_proximity_norm",
    "gi_norm",
    "li_norm",
]

TARGET_COLUMNS = {
    "cooling": "cooling_kwh_per_m2",
    "heating": "heating_kwh_per_m2",
    "other": "other_electricity_kwh_per_m2",
}

TEST_SIZE = 0.2
SEED = 42


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    needed = MORPHOLOGY_FEATURES + list(TARGET_COLUMNS.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    X = df[MORPHOLOGY_FEATURES].to_numpy(dtype=float)
    row_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(row_idx, test_size=TEST_SIZE, random_state=SEED)

    rows: list[dict[str, float | int | str]] = []

    for target_key, target_col in TARGET_COLUMNS.items():
        y = df[target_col].to_numpy(dtype=float)
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        test_r2 = float(r2_score(y[test_idx], y_pred))
        rows.append(
            {
                "target": target_key,
                "target_column": target_col,
                "model": "OLS (Morphology-only)",
                "test_r2": test_r2,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_size": TEST_SIZE,
                "random_state": SEED,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved OLS hold-out results to: {OUTPUT_CSV}")
    print(out_df[["target", "test_r2"]].to_string(index=False))


if __name__ == "__main__":
    main()
