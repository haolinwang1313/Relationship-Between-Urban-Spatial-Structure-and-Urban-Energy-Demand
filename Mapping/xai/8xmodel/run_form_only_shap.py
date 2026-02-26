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

DISPLAY_NAME_MAP = {
    "ci_norm": "Compactness index (CI)",
    "vci_norm": "Vertical compactness index (VCI)",
    "lum_norm": "Land-use mix entropy (LUM)",
    "lum_adjacency_norm": "Land-use adjacency mix",
    "lum_intensity_norm": "Dominant land-use intensity",
    "lum_proximity_norm": "Land-use proximity",
    "gi_norm": "Global street integration (GI)",
    "li_norm": "Local street integration (LI)",
}

def display_name(feature: str) -> str:
    name = DISPLAY_NAME_MAP.get(feature, feature)
    if name.endswith("_ha"):
        name = name[:-3]
    return name


def _get_paths() -> Dict[str, Path]:
    mapping_dir = SCRIPT_PATH.parents[2]
    model_dir = mapping_dir / "8x_3y" / "xgboost"
    out_dir = SCRIPT_PATH.parent
    fig_dir = out_dir / "figures"
    dependence_dir = fig_dir / "dependence"
    summary_dir = fig_dir / "summary"
    lre_dir = fig_dir / "LRE"
    for sub in [fig_dir, dependence_dir, summary_dir, lre_dir]:
        sub.mkdir(parents=True, exist_ok=True)
    return {
        "mapping": mapping_dir,
        "model_dir": model_dir,
        "dataset": mapping_dir / "8x_3y" / "dataset.csv",
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "dependence_dir": dependence_dir,
        "summary_dir": summary_dir,
        "lre_dir": lre_dir,
    }


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FORM_FEATURES + list(TARGET_COLUMNS)).difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def compute_shap(booster: xgb.Booster, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dmatrix = xgb.DMatrix(data, feature_names=list(data.columns))
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    if shap_matrix.ndim != 2:
        raise RuntimeError("Unexpected SHAP output.")
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

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), sharex=True, constrained_layout=True)

    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(
        vmin=feature_frame.to_numpy().min(),
        vmax=feature_frame.to_numpy().max(),
    )

    for i, ax in enumerate(axes):

        shap_values = shap_values_list[i]
        target_name = target_labels[i]

                                    
        order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
        max_display = min(8, shap_values.shape[1])
        order = order[:max_display]
        rng = np.random.default_rng(42 + i)

        for pos, idx in enumerate(order):
            values = shap_values[:, idx]
            feature_vals = feature_frame.iloc[:, idx]

            jitter = (rng.random(len(values)) - 0.5) * 0.6

            ax.scatter(
                values,
                np.full(values.shape, pos) + jitter,
                c=cmap(norm(feature_vals)),
                s=12,
                alpha=0.55,
                linewidths=0,
            )

        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([feature_names[j] for j in order])

        ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_title(target_name, fontweight="bold", pad=6)

        if i == 0:
            ax.set_ylabel("")
            ax.set_xlabel("SHAP value (kWh/m²)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_yticks([])
            ax.set_xlabel("SHAP value (kWh/m²)")

                 
        for spine_name in ["top", "right"]:
            ax.spines[spine_name].set_visible(False)
        for spine_name in ["bottom", "left"]:
            ax.spines[spine_name].set_linewidth(0.8)

        ax.tick_params(width=0.8, length=3)

                     
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("Feature value")
    cbar.ax.tick_params(width=0.8, length=3, labelsize=8)

    out_path = out_dir / "shap_summary_panel.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def linear_regression_effects(
    df: pd.DataFrame, features: List[str], target_col: str
) -> List[Dict[str, float]]:
    X = df[features].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)
    n = X.shape[0]
    X_design = np.column_stack([np.ones(n), X])
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X_design.T @ y
    preds = X_design @ beta
    residuals = y - preds
    dof = n - X_design.shape[1]
    if dof <= 0:
        raise ValueError("Not enough degrees of freedom for regression.")
    sigma2 = (residuals @ residuals) / dof
    cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (residuals**2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    rmse = float(np.sqrt(ss_res / len(y)))
    mae = float(np.mean(np.abs(residuals)))
    records = []
    for idx, feature in enumerate(features, start=1):
        coef = beta[idx]
        coef_se = se[idx]
        effect = 0.1 * coef
        effect_se = 0.1 * coef_se
        ci_low = effect - 1.96 * effect_se
        ci_high = effect + 1.96 * effect_se
        records.append(
            {
                "target": TARGET_COLUMNS[target_col],
                "feature": display_name(feature),
                "coef": coef,
                "coef_se": coef_se,
                "effect_per_0.1": effect,
                "effect_ci_low": ci_low,
                "effect_ci_high": ci_high,
                "r2_model": r2,
                "rmse": rmse,
                "mae": mae,
            }
        )
    return records


def main() -> None:
    paths = _get_paths()
    df = load_dataset(paths["dataset"])
    feature_data = df[FORM_FEATURES]
    feature_idx = {feat: idx for idx, feat in enumerate(FORM_FEATURES)}
    display_feature_cols = [display_name(col) for col in FORM_FEATURES]
    display_feature_frame = feature_data.copy()
    display_feature_frame.columns = display_feature_cols

    global_records = []
    top_form_records = []
    dependence_records = []
    dependence_stats = []
    regression_records = []
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
        for feature, value in zip(FORM_FEATURES, mean_abs):
            alias = display_name(feature)
            global_records.append(
                {
                    "target": target_label,
                    "feature": alias,
                    "mean_abs_shap": float(value),
                    "share": float(value / total) if total else 0.0,
                }
            )
        top_features = pd.Series(mean_abs, index=FORM_FEATURES).sort_values(ascending=False).head(4)
        for rank, (feat, value) in enumerate(top_features.items(), start=1):
            alias = display_name(feat)
            top_form_records.append(
                {
                    "target": target_label,
                    "rank": rank,
                    "feature": alias,
                    "mean_abs_shap": float(value),
                    "share_within_form": float(value / total) if total else 0.0,
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
            values = df[feat].to_numpy()
            shap_arr = shap_values[:, feature_idx[feat]]
            corr = float(np.corrcoef(values, shap_arr)[0, 1])
            q10 = float(np.quantile(values, 0.1))
            q90 = float(np.quantile(values, 0.9))
            shap_low = float(shap_arr[values <= q10].mean()) if np.any(values <= q10) else float("nan")
            shap_high = float(shap_arr[values >= q90].mean()) if np.any(values >= q90) else float("nan")
            dependence_stats.append(
                {
                    "target": target_label,
                    "feature": alias,
                    "corr": corr,
                    "value_q10": q10,
                    "value_q90": q90,
                    "shap_low": shap_low,
                    "shap_high": shap_high,
                    "shap_mean": float(shap_arr.mean()),
                }
            )
        np.save(paths["out_dir"] / f"shap_values_{target_col}.npy", shap_values)
        summary_shap_values.append(shap_values)
        summary_target_labels.append(target_label)
        regression_records.extend(linear_regression_effects(df, FORM_FEATURES, target_col))

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

    top_df = pd.DataFrame(top_form_records)
    top_df.to_csv(paths["out_dir"] / "shap_top_form_features.csv", index=False)

    dep_df = pd.DataFrame(dependence_records)
    dep_df.to_csv(paths["out_dir"] / "shap_dependence_plots.csv", index=False)

    dep_stats_df = pd.DataFrame(dependence_stats)
    dep_stats_df.to_csv(paths["out_dir"] / "shap_dependence_stats.csv", index=False)

    reg_df = pd.DataFrame(regression_records)
    reg_df.to_csv(paths["out_dir"] / "linear_regression_effects.csv", index=False)
    plot_linear_effects(reg_df, paths["lre_dir"])


def plot_linear_effects(effects_df: pd.DataFrame, lre_dir: Path) -> None:
    color_pos = "#C00000"
    color_neg = "#1f4e79"
    for target in TARGET_COLUMNS.values():
        subset = effects_df[effects_df["target"] == target].copy()
        if subset.empty:
            continue
        subset.sort_values("effect_per_0.1", inplace=True)
        y_pos = np.arange(len(subset))
        effects = subset["effect_per_0.1"].to_numpy()
        lower_err = effects - subset["effect_ci_low"].to_numpy()
        upper_err = subset["effect_ci_high"].to_numpy() - effects
        xerr = np.vstack([lower_err, upper_err])
        colors = [color_pos if val >= 0 else color_neg for val in effects]
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        bars = ax.barh(
            y_pos,
            effects,
            xerr=xerr,
            color=colors,
            alpha=0.85,
            height=0.55,
            capsize=3,
            linewidth=0,
        )
        ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subset["feature"])
        ax.set_xlabel("Effect per +0.1 change (kWh/m²)")
        ax.set_title(f"{target} elasticity (OLS)", fontweight="bold", pad=6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(0.8)
        ax.tick_params(width=0.8, length=3)
        ax.margins(y=0.05)
        fig.tight_layout(pad=0.6)
        filename = f"linear_effects_{target.replace(' ', '_').lower()}.png"
        fig.savefig(lre_dir / filename, dpi=600, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
