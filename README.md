# Relationship Between Urban Spatial Structure and Urban Energy Demand

This repository contains the data processing, UBEM simulation, machine learning, and explainability workflows used for city-scale analysis in Wuxi, China.

## Repository Structure

- `Data/`: raw and processed spatial/energy datasets.
- `Eplus/`: local EnergyPlus installation and weather files used by simulation scripts.
- `config/`: simulation templates and configuration files (for example `base_energy_model.idf` and prototype settings).
- `scripts/`: preprocessing and indicator-generation scripts.
- `Mapping/`: ML experiments and outputs.
- `Mapping/8x_3y/`: morphology-only models (8 features, 3 targets).
- `Mapping/12x_3y/`: built-environment + transport models (12 features, 3 targets).
- `Mapping/20x_3y/`: full models (20 features, 3 targets).
- `Mapping/hold-out_test_set/`: hold-out (train/test split + CV) evaluation workflow.
- `Mapping/OLS/`: OLS baseline (morphology-only) on fixed hold-out split.
- `Mapping/xai/`: SHAP analysis scripts and exported SHAP results.
- `Mapping/compare/`: model comparison tables and plotting script.
- `Robust/`: denominator robustness experiments (`land`-normalized vs `floor`-normalized targets).
- `UBEM8760/`: archived high-resolution per-building outputs (timestep = 4) and related logs/reference files.
- `assets/`: project assets (for example fonts).
- `figures/`: generated visualization outputs.

## Environment Setup

1. Create and activate a Python environment (Python 3.9 recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Install EnergyPlus locally (the project was run with EnergyPlus `24.1.0`).

3. If your EnergyPlus path is different from this repository default, pass it explicitly when running simulation scripts:

```bash
python scripts/simulate_building_energy.py --energyplus-root /path/to/EnergyPlus-24.1.0
```

## Quick Start (Using Existing Prepared Datasets)

Run these commands from the repository root:

```bash
python Mapping/8x_3y/xgboost/train_xgboost.py
python Mapping/12x_3y/xgboost/train_xgboost.py
python Mapping/20x_3y/xgboost/train_xgboost.py

python Mapping/8x_3y/lightgbm/train_lightgbm.py
python Mapping/12x_3y/lightgbm/train_lightgbm.py
python Mapping/20x_3y/lightgbm/train_lightgbm.py
```

Hold-out and OLS baselines:

```bash
python Mapping/hold-out_test_set/run_holdout_xgboost.py
python Mapping/OLS/run_ols_morphology_holdout.py
```

SHAP analysis:

```bash
python Mapping/xai/8xmodel/run_form_only_shap.py
python Mapping/xai/Allmodel/run_shap_analysis.py
```

Robustness analysis:

```bash
python Robust/run_denominator_robustness.py
```

## Optional: Full Pipeline From Raw Spatial Data

If you need to regenerate intermediate data:

```bash
python scripts/preprocess_xinwu.py
python scripts/calc_urban_form.py
python scripts/calc_built_environment.py
python scripts/calc_transportatio.py
python scripts/compute_integration_variants.py \
  --network Data/Processed/Road/xinwu_buffer15km_road_centerlines.gpkg \
  --tag buffer15km
python scripts/simulate_building_energy.py --overwrite
```

## Main Output Locations

- Processed spatial indicators: `Data/Processed/`
- Building simulation outputs: `Data/Processed/Energy/`
- Model artifacts and metrics: `Mapping/*/xgboost/`, `Mapping/*/lightgbm/`
- SHAP outputs: `Mapping/xai/`
- Robustness outputs: `Robust/results/`, `Robust/figures/`
