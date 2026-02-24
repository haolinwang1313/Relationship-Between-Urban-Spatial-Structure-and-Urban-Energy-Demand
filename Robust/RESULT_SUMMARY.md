# Robustness Check: Land vs Floor Denominator

## 1) Outcome Definitions

For each grid `g` and end-use `k`:

- Land-normalized (existing): `EI_land(k,g) = E(k,g) / A_grid(g)`
- Floor-normalized (control): `EI_floor(k,g) = E(k,g) / FA(g)`

`k ∈ {cooling, heating, other_electricity}`.

Filtering rule used for both datasets:
- Keep only grids with `FA(g) > 0`.
- Final matched sample size: `N = 2889`.

## 2) Modeling Protocol (kept identical)

- Features: same 20 features as `Mapping/20x_3y/xgboost/train_xgboost.py`.
- Split: same `KFold(n_splits=5, shuffle=True, random_state=42)`.
- XGBoost hyperparameters: exactly matched to main experiment.
- OLS baseline: same folds for comparison.
- Only change: target denominator (land vs floor).

## 3) Minimum Deliverables

### A. Performance Comparison (XGBoost)

| Target | R2 (original land, full sample) | R2 (land, FA>0) | R2 (floor, FA>0) | Delta R2 floor - original land |
|---|---:|---:|---:|---:|
| Cooling load | 0.9381 | 0.9175 | 0.4536 | -0.4845 |
| Heating load | 0.9682 | 0.9612 | 0.3661 | -0.6021 |
| Other electricity | 0.8373 | 0.7686 | 0.2397 | -0.5977 |

Source table:
- `Robust/results/performance_summary_xgboost.csv`

OLS baseline:
- `Robust/results/performance_all_models.csv`

### B1. VCI SHAP Dependence Plots

Existing land-normalized (main study):
- `Mapping/xai/Allmodel/figures/dependence/shap_dependence_cooling_load_vci.png`
- `Mapping/xai/Allmodel/figures/dependence/shap_dependence_heating_load_vci.png`
- `Mapping/xai/Allmodel/figures/dependence/shap_dependence_other_electricity_vci.png`

New robust plots (matched FA>0 sample):
- Land:
  - `Robust/figures/land/vci_dependence_cooling.png`
  - `Robust/figures/land/vci_dependence_heating.png`
  - `Robust/figures/land/vci_dependence_other_electricity.png`
- Floor:
  - `Robust/figures/floor/vci_dependence_cooling.png`
  - `Robust/figures/floor/vci_dependence_heating.png`
  - `Robust/figures/floor/vci_dependence_other_electricity.png`

Threshold estimates (binning + piecewise breakpoint):
- `Robust/results/vci_threshold_estimates.csv`

### B2. Morphology Contribution (SHAP group share)

Form-group SHAP share comparison:

| Target | Form share (original land, full sample) | Form share (land, FA>0) | Form share (floor, FA>0) |
|---|---:|---:|---:|
| Cooling load | 0.1526 | 0.1595 | 0.4522 |
| Heating load | 0.0891 | 0.0776 | 0.4554 |
| Other electricity | 0.2319 | 0.2370 | 0.5196 |

Source tables:
- `Robust/results/shap_group_share_land_vs_floor.csv`
- `Robust/results/morphology_share_comparison.csv`

Interpretation for robustness purpose:
- Under floor normalization, morphology contribution does not collapse to zero.
- In this matched-sample setting, morphology share increases notably.

