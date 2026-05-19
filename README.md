# Spatial Distribution, Determinants, and Machine Learning–Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts

[![CI](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/) [![GitHub last commit](https://img.shields.io/github/last-commit/valentineghanem-bit/hiv-tb-ml-ghana-261districts)](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana  
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)  
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana  
**Reporting standard:** STROBE  
**Date:** May 2026  
**Status:** Manuscript in preparation  

> Valentine Golden Ghanem (2026). *Spatial Distribution, Determinants, and Machine Learning–Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts.* GitHub repository. https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts

---

## ⚠️ Note on 261-District Migration (2026-05-19)

The dataset and analytical pipeline have been **migrated to 261 districts** (from 260) with **5 duplicate records removed** and **Guan District formally added**.

**Key changes:**
- ✅ Master CSV deduplicated: 265 rows → 260 unique + 1 Guan = **261 total**
- ✅ All spatial weights recomputed (KNN-8 on district centroids)
- ✅ Canonical Moran's I, LISA, GWR values updated for 261 districts
- ✅ ML pipeline retrained (RF, XGB, LightGBM, SHAP)
- ✅ Test suite validated with 261-district canonical values (seed=42)

**Duplicates removed (with rationale):**
1. Ahafo: Asunafo North Municipal (row 42 removed, row 2 kept)
2. Ashanti: Ahafo Ano North Municipal (row 16 removed, row 15 kept)
3. Ashanti: Ahafo Ano South East (row 18 removed, row 17 kept)
4. Central: Awutu Senya West (row 87 removed, row 86 kept)
5. Ashanti: Kumasi Metropolitan Area (row 40 removed, row 39 kept)

**Reproducibility:** See `scripts/deduplicate_master_dataset.py` and `MIGRATION_GUIDE.md`

Original 260-district statistics preserved in `git log`.

---

## 1. Abstract

A nationwide district-level analysis of HIV-TB co-infection in Ghana combining spatial statistics, geographically weighted regression, and ensemble machine learning across all 261 health districts (post-2018 Local Governance Act). We employ univariate and bivariate Moran's I to characterize spatial autocorrelation; LISA and Getis-Ord Gi* to delineate hotspots; GWR to estimate spatially varying determinants; and a stacked ensemble (Random Forest + XGBoost + LightGBM) with SHAP interpretation for risk prediction. Analysis reveals strong spatial clustering of co-infection (Moran's I=0.810, p<0.001), 69 high-high clusters, and an AUC-ROC of 0.991 under leave-one-district-out cross-validation.

---

## 2. Research Question & Aims

- **Primary:** Map district-level HIV-TB co-infection burden and identify spatial co-clusters across Ghana's 261 districts.
- **Secondary:** (a) Identify socioeconomic and behavioural determinants of co-infection burden using GWR; (b) build an ensemble ML risk-prediction pipeline (RF + XGB + LightGBM + Stacked) with LODO cross-validation; (c) interpret predictions via SHAP feature importance.

---

## 3. Methods Summary

| Method | Tool | Purpose |
|--------|------|----------|
| Global Moran's I | esda / libpysal | HIV-TB spatial autocorrelation |
| Univariate & Bivariate LISA | esda | Local cluster delineation |
| Getis-Ord Gi* | esda | Hotspot / coldspot detection |
| OLS + LM diagnostics | spreg | Spatial model selection |
| Spatial Error Model | spreg | Spatial dependency correction |
| Geographically Weighted Regression | mgwr | Spatially varying coefficient estimation |
| Random Forest | scikit-learn | Ensemble risk prediction |
| XGBoost | xgboost | Gradient boosted risk prediction |
| LightGBM | lightgbm | Best-performing classifier (AUC 0.998) |
| Stacked Ensemble | scikit-learn | Meta-learner combining RF + XGB + LGBM |
| SMOTE | imbalanced-learn | Class imbalance correction |
| SHAP | shap | TreeExplainer interpretability |
| GWR diagnostics (R) | GWmodel (R) | Spatial non-stationarity validation |

---

## 4. Data Sources

| Source | Variables | Year | Access |
|--------|-----------|------|--------|
| Ghana DHS 2003 | Regional HIV prevalence, behaviour, VCT, attitudes | 2003 | [dhsprogram.com](https://dhsprogram.com) |
| WHO Global Health Observatory | National HIV, TB, workforce, financing | 2001–2024 | [who.int/data/gho](https://www.who.int/data/gho) |
| Ghana Statistical Service 2021 Census | District socioeconomic variables | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |
| Ghana 261-District Shapefile | District boundary polygons (post-2018) | 2021 | Local Governance (Amendment) Act 2018 |

> DHS data accessed under signed Data Use Agreement (ICF International).

---

## 5. Key Findings

| Metric | Value | Notes |
|--------|-------|-------|
| Global Moran's I (HIV-TB co-infection) | **0.810** | p < 0.001, 999 permutations |
| Bivariate Moran's I (HIV × TB) | **0.449** | p < 0.001 |
| LISA High-High clusters | **69 districts** | Top-quartile co-clusters |
| LISA Low-Low clusters | **67 districts** | Bottom-quartile co-clusters |
| Gi* hotspots (≥95% CI) | **28 districts** | 9 at 99.9%, 17 at 99%, 2 at 95% |
| RandomForest LODO-CV AUC | **0.991** | (95% CI: 0.988–0.994) |
| LightGBM LODO-CV AUC | **0.998** | Best performer |
| GWR Local R² (mean) | **0.916** | Spatially non-stationary |
| **Districts analysed** | **261** | Post-2018 health districts; Guan added 2026-05 |

---

## 6. Repository Structure

```
hiv-tb-ml-ghana-261districts/
├── analysis/
│   ├── build_master_dataset.py     # Data integration pipeline
│   ├── spatial_analysis.py         # Moran's I, LISA, GWR, SEM
│   ├── ml_pipeline.py              # RF, XGBoost, LightGBM, Stacked, SHAP
│   └── generate_figures.py         # 300 DPI publication figures
├── scripts/
│   └── deduplicate_master_dataset.py  # Audit trail for 260→261 migration
├── app.py                          # Plotly Dash interactive application
├── analysis.R                      # R: spatial regression + GWR diagnostics
├── dashboard/
│   ├── HIV_TB_Ghana_Dashboard.html
│   ├── run_dashboard.command       # macOS launcher
│   ├── run_dashboard.bat           # Windows launcher
│   └── run_dashboard.sh            # Linux launcher
├── poster/
│   ├── create_poster.py
│   └── HIV_TB_Ghana_261_Districts_Poster.html
├── outputs/
│   ├── data/
│   │   ├── Ghana_HIV_TB_Master_Dataset.csv   # Master dataset (261 × 53)
│   │   ├── Ghana_HIV_TB_Master_Dataset_DEDUPLICATED_261.csv
│   │   └── ghana_261_final_results.geojson
│   ├── figures/                    # 9 PNG figures at 300 DPI
│   ├── tables/                     # CSV result tables
│   └── models/                     # Pickled models + SHAP values
├── tests/
│   └── test_hiv_tb.py              # Pytest suite (261-district canonical values)
├── requirements.txt
├── renv.lock
├── CITATION.cff
├── MIGRATION_GUIDE.md              # Change log: 260→261
└── README.md
```

---

## 7. Reproducibility

### 7.1 Requirements
- Python 3.12 (see `requirements.txt` for pinned versions)
- R 4.3+ (for R scripts; see `renv.lock` or `analysis.R` header for pinned packages)
- Random seed: **42** throughout (set via `random_state=42` and `np.random.seed(42)`)
- Estimated runtime: ~12–18 minutes on a standard laptop (LightGBM ~5 min)
- Tested on: Ubuntu 22.04 / macOS 14 / Windows 11 (CI: GitHub Actions)

### 7.2 Clone & install
```bash
git clone https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts.git
cd hiv-tb-ml-ghana-261districts
pip install -r requirements.txt
# For R scripts (optional):
Rscript -e "if (!requireNamespace('renv', quietly=TRUE)) install.packages('renv'); renv::restore()"
```

### 7.3 Run the analytical pipeline
```bash
# Deduplicate master dataset (audit trail)
python scripts/deduplicate_master_dataset.py

# Spatial + ML analysis
cd analysis/
python build_master_dataset.py
python spatial_analysis.py
python ml_pipeline.py
python generate_figures.py
```

### 7.4 Run the test suite
```bash
pytest tests/ -v
# Validates 261-district canonical values with seed=42
```

### 7.5 Launch the interactive Dash application
```bash
python app.py
# Navigate to http://127.0.0.1:8050 in your browser
```

### 7.6 Open the static HTML dashboard
Open `dashboard/HIV_TB_Ghana_Dashboard.html` in any modern browser, or launch via the platform-specific scripts (`run_dashboard.command` / `.bat` / `.sh`).

---

## 8. Outputs

- **Interactive Dash app:** `app.py` — `python app.py` → http://127.0.0.1:8050
- **Static HTML dashboard:** `dashboard/HIV_TB_Ghana_Dashboard.html`
- **A0 poster:** `poster/HIV_TB_Ghana_261_Districts_Poster.html`
- **Master dataset (deduplicated):** `outputs/data/Ghana_HIV_TB_Master_Dataset_DEDUPLICATED_261.csv` (261 × 53)
- **Master dataset (original, for archival):** `outputs/data/Ghana_HIV_TB_Master_Dataset.csv`
- **GeoJSON:** `outputs/data/ghana_261_final_results.geojson`
- **Figures:** `outputs/figures/*.png` — 9 figures (300 DPI)
- **Pickled models + SHAP values:** `outputs/models/`

---

## 8a. Downloadable Artefacts (HTML)

Both the interactive dashboard and the conference poster are committed to the repository as **self-contained HTML files** — no server, no build step. They can be:

- **Viewed in browser:** open the rendered preview, or clone the repo and open locally
- **Downloaded:** right-click → *Save link as*, or use the raw URL

| Artefact | View on GitHub | Live preview | Direct download |
|----------|----------------|--------------|------------------|
| Interactive dashboard | [`HIV_TB_Ghana_Dashboard.html`](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/dashboard/HIV_TB_Ghana_Dashboard.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/dashboard/HIV_TB_Ghana_Dashboard.html) | [Raw](https://raw.githubusercontent.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/main/dashboard/HIV_TB_Ghana_Dashboard.html) |
| Conference poster | [`HIV_TB_Ghana_261_Districts_Poster.html`](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/poster/HIV_TB_Ghana_261_Districts_Poster.html) | [Preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/poster/HIV_TB_Ghana_261_Districts_Poster.html) | [Raw](https://raw.githubusercontent.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/main/poster/HIV_TB_Ghana_261_Districts_Poster.html) |

> **Tip:** the dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 × 1189 mm).

---

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies.

---

## 10. Ethical Statement

This study used exclusively de-identified, publicly available secondary data from the Ghana DHS, WHO Global Health Observatory, and Ghana Statistical Service. No primary data collection from human participants was conducted. DHS data were accessed under a signed Data Use Agreement (ICF International). No institutional review board approval was required for secondary data analysis.

---

## 11. Citation

**APA:**  
Ghanem, V. G. (2026). *Spatial Distribution, Determinants, and Machine Learning–Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts*. GitHub. https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts

**BibTeX:**
```bibtex
@misc{ghanem2026hivtb,
  author = {Ghanem, Valentine Golden},
  title  = {Spatial Distribution, Determinants, and Machine Learning–Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts},
  year   = {2026},
  url    = {https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts}
}
```

A machine-readable citation is provided in `CITATION.cff`.

---

## 12. License

Code is released under the **MIT License** — see [LICENSE](LICENSE) for details. Outputs and figures: CC BY 4.0.

---

## 13. Author & Contact

- **Valentine Golden Ghanem**  
  Ghana COCOBOD Cocoa Clinic, Accra, Ghana  
  Email: [valentineghanem@gmail.com](mailto:valentineghanem@gmail.com)  
  ORCID: [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)

---

## 14. Acknowledgements

- **Ghana Demographic and Health Survey programme** (ICF International) for survey data access under signed Data Use Agreement.
- **Ghana Statistical Service** for the 2021 Population and Housing Census and administrative boundary data.
- **WHO Global Health Observatory** for national-level health indicators.
- **GitHub Actions** for continuous integration and validation.

---
