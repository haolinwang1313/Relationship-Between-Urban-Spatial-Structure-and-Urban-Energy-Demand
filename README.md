# Unlocking the potential of urban spatial structure to shape building energy demand at a large scale

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.scs.2026.107381-blue.svg)](https://doi.org/10.1016/j.scs.2026.107381)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![EnergyPlus](https://img.shields.io/badge/EnergyPlus-24.1.0-green.svg)](https://energyplus.net/)
[![Journal](https://img.shields.io/badge/Journal-Sustainable%20Cities%20and%20Society-2f855a.svg)](https://doi.org/10.1016/j.scs.2026.107381)

This repository accompanies the paper **"Unlocking the potential of urban spatial structure to shape building energy demand at a large scale"** (*Sustainable Cities and Society*, 2026). It contains the geospatial preprocessing, urban building energy modeling (UBEM), machine learning, explainability, and robustness-analysis workflows used for the city-scale case study in Xinwu District, Wuxi, China.

## Overview

Understanding how urban spatial structure shapes building energy demand at a large scale requires both physically grounded simulation and interpretable data-driven modeling. This repository links geospatial data preparation, EnergyPlus-based UBEM, and surrogate modeling to quantify how urban form, built-environment composition, and transport accessibility relate to three grid-level energy targets: **cooling load**, **heating load**, and **other electricity demand**.

The current archive includes:

- a **250 m grid** representation of Xinwu District;
- processed datasets derived from **55,284 building footprints**, **5,793 road features**, and **1,733 bus stops**;
- three feature configurations: **8 urban-form features**, **12 built-environment and transport features**, and **20 combined features**;
- reproducible workflows for **XGBoost**, **LightGBM**, **hold-out evaluation**, **OLS benchmarking**, **SHAP-based explainability**, and **denominator robustness checks**.

The archived hold-out results indicate that the 20-feature XGBoost model achieves test `R²` values of **0.937** for cooling, **0.974** for heating, and **0.849** for other electricity demand on the released split.

![Grid-level energy loads in Xinwu District](figures/energy/xinwu_energy_250m.png)

## Repository Structure

The codebase is organized as follows:

```text
├── Data/
│   ├── Processed/
│   │   ├── Indicators/         # Urban-form indicators on the 250 m grid
│   │   ├── BuiltEnvironment/   # Built-environment aggregation outputs
│   │   ├── Transportatio/      # Transport accessibility and route summaries
│   │   └── Energy/             # UBEM-derived grid-level energy demand tables
│   └── ...                     # Raw and intermediate spatial source data
├── Eplus/                      # Local EnergyPlus installation and weather files
├── config/
│   ├── base_energy_model.idf   # Base EnergyPlus model template
│   └── energy_prototypes.json  # Building prototype configuration
├── scripts/
│   ├── preprocess_xinwu.py
│   ├── calc_urban_form.py
│   ├── calc_built_environment.py
│   ├── calc_transportatio.py
│   ├── compute_integration_variants.py
│   └── simulate_building_energy.py
├── Mapping/
│   ├── 8x_3y/                  # Morphology-only datasets and ML models
│   ├── 12x_3y/                 # Built-environment + transport datasets and ML models
│   ├── 20x_3y/                 # Full-feature datasets and ML models
│   ├── hold-out_test_set/      # Train/test split evaluation outputs
│   ├── OLS/                    # Linear baseline on the fixed hold-out split
│   ├── compare/                # Cross-model comparison tables
│   └── xai/                    # SHAP scripts, arrays, figures, and summary tables
├── Robust/                     # Denominator robustness experiments and outputs
├── UBEM8760/                   # Archived high-resolution simulation outputs and logs
├── figures/                    # Study-area, built-environment, and energy maps
├── requirements.txt            # Python dependency list
└── README.md                   # Project documentation
```

## Dependencies & Installation

This project combines Python-based spatial analysis, EnergyPlus simulation, and machine-learning workflows. We recommend Python `3.9` and EnergyPlus `24.1.0`.

1. **Clone the repository**

```bash
git clone https://github.com/haolinwang1313/Relationship-Between-Urban-Spatial-Structure-and-Urban-Energy-Demand.git
cd Relationship-Between-Urban-Spatial-Structure-and-Urban-Energy-Demand
```

2. **Create a virtual environment and install Python dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Install EnergyPlus locally**

- Version used in this project: `24.1.0`
- Default local path used by the repository: `Eplus/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64`
- Default weather file: `CHN_JS_Wuxi_Proxy_583580.epw`

4. **System dependencies**

Some geospatial Python packages may require native libraries such as `GDAL`, `PROJ`, and `GEOS`, depending on your platform.

## Usage

Run all commands from the repository root. The processed datasets already included in `Data/Processed/` and `Mapping/` are sufficient for most analyses, so you do not need to regenerate the full pipeline unless you want to rebuild the study from source data.

### 1. Train surrogate models on the prepared datasets

Train XGBoost models for the three feature groups:

```bash
python Mapping/8x_3y/xgboost/train_xgboost.py
python Mapping/12x_3y/xgboost/train_xgboost.py
python Mapping/20x_3y/xgboost/train_xgboost.py
```

Train LightGBM models for the same feature groups:

```bash
python Mapping/8x_3y/lightgbm/train_lightgbm.py
python Mapping/12x_3y/lightgbm/train_lightgbm.py
python Mapping/20x_3y/lightgbm/train_lightgbm.py
```

### 2. Evaluate predictive performance and interpretability

Run the hold-out comparison and OLS baseline:

```bash
python Mapping/hold-out_test_set/run_holdout_xgboost.py
python Mapping/OLS/run_ols_morphology_holdout.py
```

Run the SHAP-based explainability workflows:

```bash
python Mapping/xai/8xmodel/run_form_only_shap.py
python Mapping/xai/Allmodel/run_shap_analysis.py
```

Run denominator robustness analysis (`land`-normalized vs `floor`-normalized targets):

```bash
python Robust/run_denominator_robustness.py
```

### 3. Rebuild the full preprocessing and simulation pipeline

If you want to regenerate intermediate data products from source layers, run:

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

If your EnergyPlus installation is located elsewhere, pass it explicitly:

```bash
python scripts/simulate_building_energy.py \
  --energyplus-root /path/to/EnergyPlus-24.1.0
```

## Citation

If this repository or workflow is useful in your research, please cite:

```bibtex
@article{wang2026unlocking,
  title={Unlocking the potential of urban spatial structure to shape building energy demand at a large scale},
  author={Wang, Haolin and Wu, Zhi and Gu, Wei and Liu, Pengxiang and Sun, Qirun and Wang, Wei},
  journal={Sustainable Cities and Society},
  year={2026},
  pages={107381},
  publisher={Elsevier},
  doi={10.1016/j.scs.2026.107381},
  url={https://doi.org/10.1016/j.scs.2026.107381}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
