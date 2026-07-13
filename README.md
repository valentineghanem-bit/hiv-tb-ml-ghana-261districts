# Spatial Distribution, Determinants, and Machine Learningâ€“Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts

[![CI](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/) [![R 4.3+](https://img.shields.io/badge/R-4.3+-blue.svg)](https://www.r-project.org/) [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--8332--0220-green.svg)](https://orcid.org/0009-0002-8332-0220)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE
**Date:** May 2026
**Status:** Manuscript in preparation

---

## 1. Abstract

A nationwide district-level analysis of HIV-TB co-infection in Ghana combining spatial statistics, geographically weighted regression, and ensemble machine learning across all 261 health districts (post-2018 Local Governance Act). Univariate and bivariate Moran's I characterise spatial autocorrelation; LISA and Getis-Ord Gi* delineate hotspots; GWR estimates spatially varying determinants; and a stacked ensemble (Random Forest + XGBoost + LightGBM) with SHAP interpretation yields district-level risk predictions. Analysis reveals moderate spatial clustering of co-infection (global Moran's I = 0.469, p < 0.001), 50 LISA High-High clusters, and a LightGBM AUC of 0.998 under standard 10-fold cross-validation -- which drops to 0.798 +/- 0.250 under leave-one-region-out spatial cross-validation (N=12 regional folds), the more honest estimate of out-of-region generalisation given strong spatial autocorrelation in the predictors.

---

## 2. Research Question & Aims

- **Primary:** Map district-level HIV-TB co-infection burden and identify spatial co-clusters across Ghana's 261 districts.
- **Secondary:** (a) Identify socioeconomic and behavioural determinants of co-infection burden using GWR; (b) build an ensemble ML risk-prediction pipeline (RF + XGB + LightGBM + Stacked) evaluated under both standard 10-fold and leave-one-region-out spatial cross-validation; (c) interpret predictions via SHAP feature importance.

---

## 3. Methods Summary

| Method | Tool | Purpose |
|--------|------|---------|
| Global Moran's I | esda / libpysal | HIV-TB spatial autocorrelation |
| Univariate & bivariate LISA | esda | Local cluster delineation |
| Getis-Ord Gi* | esda | Hotspot / coldspot detection |
| OLS + LM diagnostics | spreg | Spatial model selection |
| Spatial Error Model | spreg | Spatial dependency correction |
| Geographically Weighted Regression | mgwr | Spatially varying coefficient estimation |
| Random Forest | scikit-learn | Ensemble risk prediction |
| XGBoost | xgboost | Gradient boosted risk prediction |
| LightGBM | lightgbm | Best 10-fold CV performer (AUC 0.998); spatial CV AUC 0.798 Â± 0.250 |
| Stacked ensemble | scikit-learn | Meta-learner combining RF + XGB + LGBM |
| SMOTE | imbalanced-learn | Class imbalance correction |
| SHAP | shap | TreeExplainer interpretability |
| GWR diagnostics | GWmodel (R) | Spatial non-stationarity validation |

---

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| Ghana DHS 2003 | Regional HIV prevalence, behaviour, VCT, attitudes | 2003 | [dhsprogram.com](https://dhsprogram.com) (registration) |
| WHO Global Health Observatory | National HIV, TB, workforce, financing indicators | 2001â€“2024 | [who.int/data/gho](https://www.who.int/data/gho) (open) |
| Ghana Statistical Service 2021 Census | District socioeconomic variables | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |
| Ghana 260-feature district GeoJSON + Guan convention | District boundary polygons for 261 administrative districts; Guan retained as an administrative unit using the Krachi East shared-polygon approximation | 2021/2018 frame | Ghana Statistical Service / Local Governance (Amendment) instruments |

> DHS data accessed under signed Data Use Agreement (ICF International). No individual participant data redistributed.

---

## 5. Key Findings

| Metric | Value |
|--------|-------|
| Global Moran's I (HIV-TB co-infection) | 0.469 (p < 0.001, 999 permutations) |
| Bivariate Moran's I (HIV Ã— TB) | 0.526 (p < 0.001) |
| LISA High-High clusters | 50 districts |
| LISA Low-Low clusters | 41 districts |
| Bivariate LISA High-High clusters | 48 districts |
| Gi* hotspots | 1 district at 95%, 11 at 90% (none at 99%/99.9%) |
| RandomForest 10-fold CV AUC | 0.996 Â± 0.004 (non-spatial, IID folds) |
| RandomForest spatial (leave-one-region-out) CV AUC | 0.763 Â± 0.265 (N=12 regional folds) |
| LightGBM 10-fold CV AUC | 0.998 Â± 0.004 (non-spatial, IID folds; best 10-fold performer) |
| LightGBM spatial (leave-one-region-out) CV AUC | 0.798 Â± 0.250 (N=12 regional folds; the honest generalisation estimate) |
| GWR global R² | 0.917 |
| GWR local R² (mean) | 0.854 (range 0.509-0.985) |
| Districts analysed | 261 (Guan District, Oti Region, added 2026-05; shares its parent Krachi East Municipal polygon for spatial weighting -- no distinct legacy boundary exists) |

> **Note on the 10-fold vs. spatial-CV gap:** the ~20-point AUC drop and large SD (Â±0.25â€“0.27) between standard 10-fold CV and leave-one-region-out spatial CV reflects genuine, strong spatial autocorrelation in Ghanaian district-level health data (neighbouring districts have highly correlated HIV/TB rates), not a modelling error. The 10-fold number should not be quoted as the model's ability to generalise to an unseen region; the spatial-CV number is the more defensible generalisation estimate, and its wide SD signals real fold-to-fold instability that any downstream use of this model should account for.

---

## 6. Repository Structure

```
hiv-tb-ml-ghana-261districts/
â”œâ”€â”€ analysis/
â”‚   â”œâ”€â”€ build_master_dataset.py     # Data integration pipeline
â”‚   â”œâ”€â”€ spatial_analysis.py         # Moran's I, LISA, GWR, SEM
â”‚   â”œâ”€â”€ ml_pipeline.py              # RF, XGBoost, LightGBM, Stacked, SHAP
â”‚   â””â”€â”€ generate_figures.py         # 300 DPI publication figures
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ deduplicate_master_dataset.py  # 260â†’261 migration audit trail
â”‚   â”œâ”€â”€ spatial_utils.py            # Reusable spatial analysis utilities
â”‚   â””â”€â”€ spatial_diagnostics.R       # R: spatial autocorrelation diagnostics
â”œâ”€â”€ app.py                          # Plotly Dash interactive application
â”œâ”€â”€ analysis.R                      # R: spatial regression + GWR diagnostics
â”œâ”€â”€ dashboard/
â”‚   â””â”€â”€ HIV_TB_Ghana_Dashboard.html
â”œâ”€â”€ poster/
â”‚   â””â”€â”€ HIV_TB_Ghana_260_Districts_Poster.html
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ data/                       # Master CSV + result tables
â”‚   â””â”€â”€ figures/                    # Publication figures (300 DPI)
â”œâ”€â”€ tests/
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ Dockerfile
â””â”€â”€ CITATION.cff
```

---

## 7. Reproducibility

### 7.1 Requirements

- Python 3.12 (pinned in `requirements.txt`)
- R 4.3+ with packages: GWmodel, spdep, spatialreg, dplyr (see `analysis.R` header)
- Random seed: 42 throughout
- Estimated runtime: ~10â€“15 minutes on a standard laptop (GWR is compute-intensive)
- Tested on: Ubuntu 22.04 / macOS 14 / Windows 11 (CI: GitHub Actions)

### 7.2 Clone & install

```bash
git clone https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts.git
cd hiv-tb-ml-ghana-261districts
pip install -r requirements.txt
```

### 7.3 Run the analytical pipeline

```bash
python analysis/build_master_dataset.py
python analysis/spatial_analysis.py
python analysis/ml_pipeline.py
python analysis/generate_figures.py
```

### 7.4 Run the test suite

```bash
pytest tests/ -v
```

### 7.5 Launch the interactive Dash application

```bash
python app.py
# Visit http://127.0.0.1:8050
```

### 7.6 Open the static HTML dashboard

```bash
# macOS
open dashboard/HIV_TB_Ghana_Dashboard.html
# Windows
start dashboard/HIV_TB_Ghana_Dashboard.html
# Linux
xdg-open dashboard/HIV_TB_Ghana_Dashboard.html
```

---

## 8. Outputs

| Output | Description |
|--------|-------------|
| `outputs/data/` | Master CSV, LISA results, GWR coefficients, SHAP values |
| `outputs/figures/` | Publication-quality PNG figures (300 DPI) |
| `dashboard/` | Self-contained interactive HTML dashboard |
| `poster/` | A0 conference poster (HTML, print-ready) |

## 8a. Downloadable Artefacts (HTML)

Both the interactive dashboard and the conference poster are committed as self-contained HTML files â€” no server, no build step required.

| Artefact | View on GitHub | Live preview | Direct download (raw HTML) |
|----------|---------------|--------------|---------------------------|
| Interactive dashboard | [View](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/dashboard/HIV_TB_Ghana_Dashboard.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/dashboard/HIV_TB_Ghana_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/main/dashboard/HIV_TB_Ghana_Dashboard.html) |
| Conference poster | [View](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/poster/HIV_TB_Ghana_260_Districts_Poster.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/poster/HIV_TB_Ghana_260_Districts_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/main/poster/HIV_TB_Ghana_260_Districts_Poster.html) |

> **Tip:** The dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 Ã— 1189 mm).

---

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies.

---

## 10. Ethical Statement

This study analyses publicly released aggregate data from the Ghana Demographic and Health Survey (ICF International), the WHO Global Health Observatory, and the Ghana Statistical Service 2021 Census. No individual participant data were accessed. All inputs are de-identified district and regional summary statistics. Ethical review was not required for analysis of publicly available aggregate statistics; DHS data were accessed under the standard DHS Programme Data Use Agreement.

---

## 11. Citation

**APA:**
Ghanem, V. G. (2026). *Spatial Distribution, Determinants, and Machine Learningâ€“Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts.* GitHub. https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts

**BibTeX:**
```bibtex
@misc{ghanem2026hivtb,
  author = {Ghanem, Valentine Golden},
  title  = {Spatial Distribution, Determinants, and Machine Learning--Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts}
}
```

A machine-readable citation is provided in `CITATION.cff`.

---

## 12. License

Code is released under the **MIT License** â€” see [LICENSE](LICENSE) for details.
Outputs and figures: **CC BY 4.0**.

---

## 13. Author & Contact

**Valentine Golden Ghanem**
Ghana COCOBOD Cocoa Clinic, Accra, Ghana
Email: valentineghanem@gmail.com
ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

---

## 14. Acknowledgements

The author thanks the DHS Programme (ICF International) for the Ghana DHS data, the WHO for the Global Health Observatory indicators, and the Ghana Statistical Service for Census district files and boundary geometries. Spatial analysis relied on esda, libpysal, spreg, spdep, and the R GWmodel package. Ensemble modelling used scikit-learn, XGBoost, and LightGBM; interpretability used SHAP.

