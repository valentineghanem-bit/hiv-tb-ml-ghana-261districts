# Spatial Distribution, Determinants, and Machine Learning–Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts

[![CI](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/actions/workflows/ci.yml/badge.svg)](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

**Author:** Valentine Golden Ghanem | Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**ORCID:** [0009-0002-8332-0220](https://orcid.org/0009-0002-8332-0220)
**Affiliation:** Ghana COCOBOD Cocoa Clinic, Accra, Ghana
**Reporting standard:** STROBE
**Date:** May 2026
**Status:** Manuscript in preparation

> Valentine Golden Ghanem (2026). *Spatial Distribution, Determinants, and Machine Learning–Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts.* GitHub repository. https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts

---

## Note on 261-district framework (2026-05-18)

This repository now comprehensively covers **all 261 administrative health districts** in Ghana (post-2018 Local Governance Amendment). The Guan District (Oti Region) was added in May 2026. 

**Dataset deduplication:** The master dataset has been deduplicated to ensure each district appears exactly once with distinct computed values for all derived variables:

- **Spatial weights:** KNN-8 from district centroids (lat / lon)
- **Global / Local Moran's I:** primary outcome variable, 999 permutations
- **Bivariate LISA:** primary × secondary variable (where defined)
- **Getis-Ord Gi\*:** hotspot tiering at 95% / 99% / 99.9% CI
- **ML risk:** RandomForest classifier, 5-fold cross-validated probabilities

The original 260-district statistics are preserved in `git log` for comparison.

---

## 1. Abstract

A nationwide district-level analysis of HIV-TB co-infection in Ghana combining spatial statistics, geographically weighted regression, and ensemble machine learning across all 261 health districts. Data from Ghana DHS 2003 (regional HIV/behaviour), WHO GHO (national TB/workforce), and Ghana Statistical Service 2021 Census (socioeconomic). Global Moran's I = 0.810 (HIV-TB co-infection, p < 0.001); 69 districts in high-high LISA clusters; LightGBM LODO-CV AUC = 0.998. GWR explains 91.6% of co-infection variance. **Conclusions:** Marked spatial clustering; poverty, VCT coverage, and treatment success are key modifiable determinants. Machine learning can stratify districts for targeted public health investment.

---

## 2. Research Question & Aims

- **Primary:** Map district-level HIV-TB co-infection burden and identify spatial co-clusters across Ghana's 261 districts.
- **Secondary:** (a) Identify socioeconomic and behavioural determinants of co-infection burden using GWR; (b) build an ensemble ML risk-prediction pipeline (RF + XGB + LightGBM + Stacked) with LODO-CV validation; (c) provide SHAP interpretability for policy translation.

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
| WHO Global Health Observatory | National HIV, TB, workforce, financing | 2001–2022 | [who.int/data/gho](https://www.who.int/data/gho) |
| Ghana Statistical Service 2021 Census | District socioeconomic variables | 2021 | [statsghana.gov.gh](https://statsghana.gov.gh) |
| Ghana 261-District Shapefile | District boundary polygons | 2021 | Local Governance (Amendment) Act 2018 |

> DHS data accessed under signed Data Use Agreement (ICF International).

---

## 5. Key Findings

| Metric | Value |
|--------|-------|
| Global Moran's I (HIV-TB co-infection) | 0.810 (p < 0.001) |
| Bivariate Moran's I (HIV × TB) | 0.449 (p < 0.001) |
| LISA High-High clusters | 69 districts |
| LISA Low-Low clusters | 67 districts |
| Gi* hotspots (≥95% CI) | 28 districts (9 at 99.9%, 17 at 99%, 2 at 95%) |
| RandomForest 5-fold CV AUC | 0.991 |
| Districts analysed | **261** (all post-2018 health districts) |
| Data quality | 261 unique districts, fully deduplicated (2026-05-18) |

---

## 6. Repository Structure

```
hiv-tb-ml-ghana-261districts/
├── analysis/
│   ├── build_master_dataset.py     # Data integration pipeline
│   ├── spatial_analysis.py         # Moran's I, LISA, GWR, SEM
│   ├── ml_pipeline.py              # RF, XGBoost, LightGBM, Stacked, SHAP
│   └── generate_figures.py         # 300 DPI publication figures
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
│   │   ├── Ghana_HIV_TB_Master_Dataset.csv   # Master dataset (261 × 52, deduplicated)
│   │   └── ghana_261_final_results.geojson
│   ├── figures/                    # 9 PNG figures at 300 DPI
│   ├── tables/                     # CSV result tables
│   └── models/                     # Pickled models + SHAP values
├── requirements.txt
├── renv.lock
├── CITATION.cff
└── README.md
```

---

## 7. Reproducibility

### 7.1 Requirements
- Python 3.12 (see `requirements.txt` for pinned versions)
- R 4.3+ (for R scripts; see `renv.lock` or `analysis.R` header for pinned packages)
- Random seed: 42 throughout (set via `random_state=42` and `np.random.seed(42)`)
- Estimated runtime: ~10–15 minutes on a standard laptop (longer with LightGBM)
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
cd analysis/
python build_master_dataset.py
python spatial_analysis.py
python ml_pipeline.py
python generate_figures.py
```

### 7.4 Run the test suite
```bash
pytest tests/ -v
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
- **Master dataset:** `outputs/data/Ghana_HIV_TB_Master_Dataset.csv` (261 × 52, deduplicated)
- **GeoJSON:** `outputs/data/ghana_261_final_results.geojson`
- **Figures:** `outputs/figures/*.png` — 9 figures (300 DPI)
- **Pickled models + SHAP values:** `outputs/models/`

---

## 8a. Downloadable artefacts (HTML)

Both the interactive dashboard and the conference poster are committed to the repository as **self-contained HTML files** — no server, no build step. They can be:

- **Viewed in browser:** open the rendered preview, or clone the repo and open locally
- **Downloaded:** right-click → *Save link as*, or use the raw URL

| Artefact | View on GitHub | Live preview | Direct download (raw HTML) |
|----------|----------------|--------------|------------------------------|
| Interactive dashboard | [`HIV_TB_Ghana_Dashboard.html`](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/dashboard/HIV_TB_Ghana_Dashboard.html) | [Open preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/dashboard/HIV_TB_Ghana_Dashboard.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/main/dashboard/HIV_TB_Ghana_Dashboard.html) |
| Conference poster | [`HIV_TB_Ghana_261_Districts_Poster.html`](https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/poster/HIV_TB_Ghana_261_Districts_Poster.html) | [Open preview](https://htmlpreview.github.io/?https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/blob/main/poster/HIV_TB_Ghana_261_Districts_Poster.html) | [Download](https://raw.githubusercontent.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts/main/poster/HIV_TB_Ghana_261_Districts_Poster.html) |

> **Tip:** the dashboard works fully offline once downloaded. The poster is print-ready at A0 (841 × 1189 mm).

---

## 9. Reporting Standard

This study follows the **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology) reporting guideline for observational ecological studies.

---

## 10. Ethical Statement

This study used exclusively de-identified, publicly available secondary data from the Ghana DHS, WHO, and Ghana Statistical Service. No primary data collection from human participants was conducted. DHS data were accessed under a signed Data Use Agreement with ICF International. The study is exempt from ethical review as secondary data analysis with no human participant involvement.

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
- **WHO Global Health Observatory** for national-level indicators.

---
