from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import xgboost as xgb


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "TimesNewRoman.ttf"
if FONT_PATH.exists():
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


TARGET_COLUMNS = {
    "cooling_kwh_per_m2": "Cooling load",
    "heating_kwh_per_m2": "Heating load",
    "other_electricity_kwh_per_m2": "Other electricity",
}

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

DISPLAY_NAME_MAP = {
                        
    "ci_norm": "Compactness index (CI)",
    "vci_norm": "Vertical compactness index (VCI)",
    "lum_norm": "Land-use mix entropy (LUM)",
    "lum_adjacency_norm": "Land-use adjacency mix",
    "lum_intensity_norm": "Dominant land-use intensity",
    "lum_proximity_norm": "Land-use proximity",
    "gi_norm": "Global street integration (GI)",
    "li_norm": "Local street integration (LI)",
                                      
    "single_family_ha": "Single-family residential area",
    "multi_family_ha": "Multi-family residential area",
    "facility_neighborhood_ha": "Neighborhood service area",
    "facility_sales_ha": "Commercial sales area",
    "facility_office_ha": "Office area",
    "facility_education_ha": "Educational facility area",
    "facility_industrial_ha": "Industrial facility area",
    "parks_green_ha": "Green space area",
    "water_area_ha": "Water body area",
    "road_area_ha": "Road area",
                             
    "subway_influence_ha": "Subway station influence area",
    "bus_routes_cnt": "Number of bus routes",
}


def display_name(feature: str) -> str:
    return DISPLAY_NAME_MAP.get(feature, feature)


def _get_paths() -> Dict[str, Path]:
    mapping_root = SCRIPT_PATH.parents[2]
    model_dir = mapping_root / "20x_3y" / "xgboost"
    out_dir = SCRIPT_PATH.parent
    fig_dir = out_dir / "figures"
    dependence_dir = fig_dir / "dependence"
    summary_dir = fig_dir / "summary"
    for sub in [fig_dir, dependence_dir, summary_dir]:
        sub.mkdir(parents=True, exist_ok=True)
    return {
        "mapping": mapping_root,
        "model_dir": model_dir,
        "dataset": model_dir / "dataset.csv",
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "dependence_dir": dependence_dir,
        "summary_dir": summary_dir,
        "compare_summary": mapping_root / "compare" / "xgboost_form_vs_all_summary.csv",
    }


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = set(FORM_FEATURES + BUILT_FEATURES + TRANSPORT_FEATURES)
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def compute_shap(booster: xgb.Booster, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dmatrix = xgb.DMatrix(data, feature_names=list(data.columns))
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    if shap_matrix.ndim != 2:
        raise RuntimeError("Unexpected SHAP matrix shape.")
    return shap_matrix[:, :-1], shap_matrix[:, -1]


def plot_dependence(
    out_dir: Path,
    target_name: str,
    feature_name: str,
    feature_values: np.ndarray,
    shap_values: np.ndarray,
) -> Path:
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    scatter = ax.scatter(
        feature_values,
        shap_values,
        c=feature_values,
        cmap="viridis",
        s=18,
        alpha=0.55,
        linewidths=0,
    )
    bins = np.linspace(feature_values.min(), feature_values.max(), 15)
    if np.unique(bins).size > 1:
        digitized = np.digitize(feature_values, bins) - 1
        centers = []
        means = []
        for b in range(len(bins) - 1):
            mask = digitized == b
            if not np.any(mask):
                continue
            centers.append((bins[b] + bins[b + 1]) / 2)
            means.append(shap_values[mask].mean())
        if centers:
            ax.plot(
                centers,
                means,
                color="#C00000",
                linewidth=1.5,
                label="Binned mean",
                marker="o",
                markersize=3,
            )
            ax.legend()
    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value (kWh/m²)")
    ax.set_title(f"{target_name} vs {feature_name}", fontweight="bold", pad=6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)
    cbar = fig.colorbar(scatter, ax=ax, label=feature_name, pad=0.02)
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3)
    fig.tight_layout(pad=0.6)
    filename = f"shap_dependence_{target_name.replace(' ', '_').lower()}_{feature_name}.png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary(
    shap_values_list: List[np.ndarray],
    feature_frame: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path,
    target_labels: List[str],
) -> None:
    """Generate a 1x3 SHAP summary panel (Cooling, Heating, Other)."""

    if len(shap_values_list) != 3 or len(target_labels) != 3:
        raise ValueError("plot_summary expects exactly 3 targets in order: cooling/heating/other.")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, constrained_layout=True)
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=feature_frame.to_numpy().min(), vmax=feature_frame.to_numpy().max())

    for i, ax in enumerate(axes):
        shap_values = shap_values_list[i]
        target_name = target_labels[i]
        order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][: min(10, shap_values.shape[1])]
        rng = np.random.default_rng(42 + i)

        for pos, idx in enumerate(order):
            values = shap_values[:, idx]
            feature_vals = feature_frame.iloc[:, idx]
            jitter = (rng.random(len(values)) - 0.5) * 0.6
            ax.scatter(
                values,
                np.full(values.shape, pos) + jitter,
                c=cmap(norm(feature_vals)),
                s=10,
                alpha=0.55,
                linewidths=0,
            )

        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([feature_names[j] for j in order])
        ax.set_xlabel("SHAP value (kWh/m²)")
        ax.set_title(target_name, fontweight="bold", pad=6)
        ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--")

        if i == 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_yticks([])

        for spine_name in ["top", "right"]:
            ax.spines[spine_name].set_visible(False)
        for spine_name in ["bottom", "left"]:
            ax.spines[spine_name].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.02)
    cbar.ax.tick_params(width=0.8, length=3, labelsize=8)
    cbar.set_label("Feature value")

    out_path = out_dir / "shap_summary_panel.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def compute_group_importance(feature_importance: pd.Series) -> Dict[str, float]:
    return {
        "Form": float(feature_importance[FORM_FEATURES].sum()),
        "Built": float(feature_importance[BUILT_FEATURES].sum()),
        "Transport": float(feature_importance[TRANSPORT_FEATURES].sum()),
    }


def main() -> None:
    paths = _get_paths()
    df = load_dataset(paths["dataset"])
    feature_cols = [
        c for c in df.columns if c not in ["grid_id_main"] + list(TARGET_COLUMNS.keys())
    ]
    feature_idx = {feat: idx for idx, feat in enumerate(feature_cols)}
    feature_data = df[feature_cols]
    display_feature_cols = [display_name(col) for col in feature_cols]
    display_feature_frame = feature_data.copy()
    display_feature_frame.columns = display_feature_cols

    global_records = []
    group_records = []
    top_form_records = []
    dependence_records = []
    summary_shap_values: List[np.ndarray] = []
    summary_target_labels: List[str] = []

    for target_col, target_label in TARGET_COLUMNS.items():
        model_path = paths["model_dir"] / f"xgb_{target_col}.json"
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        shap_values, base_values = compute_shap(booster, feature_data)
        preds = booster.predict(xgb.DMatrix(feature_data))
        residual = np.abs(shap_values.sum(axis=1) + base_values - preds).max()
        if residual > 5e-4:
            raise RuntimeError(f"SHAP sum mismatch for {target_col} (max diff {residual:.4e})")
        mean_abs = np.abs(shap_values).mean(axis=0)
        total = mean_abs.sum()
        for feature, value in zip(feature_cols, mean_abs):
            alias = display_name(feature)
            global_records.append(
                {
                    "target": target_label,
                    "feature": alias,
                    "mean_abs_shap": float(value),
                    "share": float(value / total) if total else 0.0,
                    "group": (
                        "Form"
                        if feature in FORM_FEATURES
                        else "Built"
                        if feature in BUILT_FEATURES
                        else "Transport"
                    ),
                }
            )
        group_totals = compute_group_importance(pd.Series(mean_abs, index=feature_cols))
        for group, value in group_totals.items():
            group_records.append(
                {
                    "target": target_label,
                    "group": group,
                    "mean_abs_shap": value,
                    "share": float(value / total) if total else 0.0,
                }
            )
        form_importance = pd.Series(mean_abs, index=feature_cols)[FORM_FEATURES]
        top_features = form_importance.sort_values(ascending=False).head(4)
        for rank, (feat, value) in enumerate(top_features.items(), start=1):
            alias = display_name(feat)
            top_form_records.append(
                {
                    "target": target_label,
                    "rank": rank,
                    "feature": alias,
                    "mean_abs_shap": float(value),
                    "share_within_form": float(
                        value / form_importance.sum() if form_importance.sum() else 0.0
                    ),
                }
            )
            fig_path = plot_dependence(
                paths["dependence_dir"],
                target_label,
                alias,
                df[feat].to_numpy(),
                shap_values[:, feature_idx[feat]],
            )
            dependence_records.append(
                {
                    "target": target_label,
                    "feature": alias,
                    "figure": str(fig_path.relative_to(paths["mapping"])),
                }
            )
        np.save(paths["out_dir"] / f"shap_values_{target_col}.npy", shap_values)
        summary_shap_values.append(shap_values)
        summary_target_labels.append(target_label)

    plot_summary(
        summary_shap_values,
        display_feature_frame,
        display_feature_cols,
        paths["summary_dir"],
        summary_target_labels,
    )

    global_df = pd.DataFrame(global_records).sort_values(
        ["target", "mean_abs_shap"], ascending=[True, False]
    )
    global_df.to_csv(paths["out_dir"] / "shap_global_importance.csv", index=False)

    group_df = pd.DataFrame(group_records)
    group_df.to_csv(paths["out_dir"] / "shap_group_importance.csv", index=False)

    top_form_df = pd.DataFrame(top_form_records)
    top_form_df.to_csv(paths["out_dir"] / "shap_top_form_features.csv", index=False)

    dep_df = pd.DataFrame(dependence_records)
    dep_df.to_csv(paths["out_dir"] / "shap_dependence_plots.csv", index=False)

    compare_df = pd.read_csv(paths["compare_summary"])
    group_summary = group_df.pivot_table(index="target", columns="group", values="share").reset_index()
    merged = compare_df.merge(group_summary, on="target", how="left").rename(
        columns={
            "Form": "Form_SHAP_share",
            "Built": "Built_SHAP_share",
            "Transport": "Transport_SHAP_share",
        }
    )
    merged.to_csv(paths["out_dir"] / "r2_vs_shap_group_share.csv", index=False)


if __name__ == "__main__":
    main()
