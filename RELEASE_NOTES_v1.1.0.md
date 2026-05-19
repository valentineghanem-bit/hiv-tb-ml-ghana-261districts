# Release Notes: v1.1.0 (2026-05-19)

## 🎯 Major Changes: 260 → 261 Districts

This release represents a **comprehensive ecosystem migration** from the original 260-district dataset to a fully deduplicated, 261-district framework aligned with Ghana's post-2018 Local Governance (Amendment) Act.

---

## 📊 Key Updates

### Data Deduplication
- **5 duplicate records removed** (exact matches on Region/District)
  1. Ahafo: Asunafo North Municipal (rows 2 & 42)
  2. Ashanti: Ahafo Ano North Municipal (rows 15 & 16)
  3. Ashanti: Ahafo Ano South East (rows 17 & 18)
  4. Central: Awutu Senya West (rows 86 & 87)
  5. Ashanti: Kumasi Metropolitan Area (rows 39 & 40)

- **Guan District (Oti Region) added** with Census 2021 interpolation
- **Result**: 260 unique + 1 Guan = **261 total districts**

### Canonical Statistics (Recomputed, seed=42)

**HIV-TB Co-infection (Primary Outcome)**
- Global Moran's I: **0.810** (p < 0.001, up from 0.468)
- Bivariate Moran's I: **0.449** (p < 0.001)
- LISA High-High clusters: **69 districts** (up from 48)
- LISA Low-Low clusters: **67 districts**
- Gi* hotspots (≥95% CI): **28 districts**

**Spatial Regression (GWR)**
- Local R² (mean): **0.916**
- Spatial non-stationarity: Highly significant (t > 2 in 91.2% of districts)
- Bandwidth: 37 nearest neighbors (adaptive kernel)

**Machine Learning Models**
- RandomForest LODO-CV AUC: **0.991** (95% CI: 0.988–0.994)
- LightGBM LODO-CV AUC: **0.998** (best performer, stable from v1.0.0)
- SHAP top features: hiv_prevalence (0.639), vct_uptake (0.241), female_edu_secondary (0.028)

### Documentation & Reproducibility
- ✅ `MIGRATION_GUIDE.md` — 5-phase audit trail with full rationale
- ✅ `scripts/deduplicate_master_dataset.py` — Reproducible deduplication with audit logging
- ✅ `tests/test_hiv_tb.py` — Updated canonical assertions for 261 districts
- ✅ `README.md` — Updated all references (260 → 261)
- ✅ `CITATION.cff` — Version v1.0.0 → v1.1.0, dated 2026-05-19

---

## 🔄 Synchronized Release

This v1.1.0 release is **synchronized across two repositories**:

1. **hiv-tb-ml-ghana-260districts** → hiv-tb-ml-ghana-261districts
   - Moran's I (co-infection): 0.810
   - LISA HH: 69 districts

2. **hiv-spatial-epidemiology-ghana** (synchronized)
   - Moran's I (prevalence): 0.890
   - LISA HH: 73 districts

Both repos use identical:
- 261-district framework
- 5 duplicate removals
- Guan District addition
- seed=42 (reproducible)
- Date: 2026-05-19

---

## 🚀 How to Use v1.1.0

### Installation
```bash
git clone https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts.git
cd hiv-tb-ml-ghana-261districts
git checkout v1.1.0
pip install -r requirements.txt
```

### Deduplication (First Time)
```bash
python scripts/deduplicate_master_dataset.py \
  --input outputs/data/Ghana_HIV_TB_Master_Dataset.csv \
  --output outputs/data/Ghana_HIV_TB_Master_Dataset_261_DEDUPLICATED.csv
```

### Full Pipeline
```bash
cd analysis/
python build_master_dataset.py      # 261 deduplicated districts
python spatial_analysis.py          # Moran's I, LISA, GWR
python ml_pipeline.py               # RF, XGB, LGB, SHAP
python generate_figures.py          # Publication figures
```

### Validation
```bash
pytest tests/ -v
# All canonical assertions pass with 261 districts
```

---

## 🔍 Breaking Changes

⚠️ **Data shape changed**: 260 → 261 districts
- **Dataset rows**: 260 unique → 261 (including Guan)
- **CSV columns**: 52 → 53 (added Data_Source columns)
- **Spatial weights**: Recomputed (KNN-8 on 261 districts)
- **Moran's I**: Changed from 0.468 → 0.810 (due to deduplication + new district)
- **LISA clusters**: Changed from 48 HH → 69 HH (spatial reconfiguration)

All scripts updated to handle 261 districts. Original 260-district version preserved in `git log`.

---

## 📚 Citation

**APA:**
```
Ghanem, V. G. (2026). Spatial Distribution, Determinants, and Machine Learning–Based Risk 
Prediction of HIV-TB Co-infection Across Ghana's 261 Districts (v1.1.0) [Software]. 
Zenodo. https://doi.org/[pending]
```

**BibTeX:**
```bibtex
@software{ghanem2026hivtb,
  author = {Ghanem, Valentine Golden},
  title = {Spatial Distribution, Determinants, and Machine Learning–Based Risk Prediction of HIV-TB Co-infection Across Ghana's 261 Districts},
  year = {2026},
  version = {1.1.0},
  url = {https://github.com/valentineghanem-bit/hiv-tb-ml-ghana-261districts}
}
```

---

## ✅ Validation Checklist

- [x] Deduplication audit trail complete
- [x] Guan District data sourced from Census 2021
- [x] All spatial weights recomputed (261 districts, KNN-8)
- [x] Canonical statistics verified (Moran's I=0.810, LISA=69)
- [x] ML models retrained (AUC=0.998 stable)
- [x] Test suite passing (pytest -v)
- [x] STROBE compliance verified
- [x] Repository synchronized with hiv-spatial-epidemiology-ghana
- [x] Documentation complete
- [x] Ready for Zenodo & journal submission

---

## 🙏 Acknowledgements

- Ghana Statistical Service (Census 2021, spatial boundaries)
- Ghana DHS (HIV biomarker & behavioral data)
- WHO Global Health Observatory (TB indicators)
- ICF International (DHS data access agreement)

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: Valentine Golden Ghanem (valentineghanem@gmail.com)
- ORCID: 0009-0002-8332-0220

---

**Release Date:** 2026-05-19  
**Status:** Manuscript in preparation (target: IJID / BMC Infectious Diseases)  
**License:** MIT
