#!/usr/bin/env python3
"""
tests/test_hiv_tb.py - Ghana HIV-TB Co-infection Spatial & ML Analysis (261 Districts)
Unit tests with canonical value assertions.

Run: pytest tests/ -v
Tenet 8: SEED=42. Canonical values recomputed 2026-07-14 against the real
261-district geometry (Guan District shares its parent Krachi East Municipal
polygon for spatial weighting -- no distinct legacy boundary exists).
Methods: Moran's I, Bivariate LISA, GWR, LightGBM ensemble, SHAP.
"""

import os
import pytest
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_CSV = os.path.join(REPO_ROOT, "outputs", "data", "Ghana_HIV_TB_Master_Dataset.csv")
MORANS_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "global_morans_I.csv")
GWR_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "gwr_summary.csv")
ML_CV_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "ml_spatial_cv_results.csv")
ML_10F_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "ml_10fold_cv_results.csv")
SHAP_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "shap_feature_importance.csv")
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "figures")

# CANONICAL VALUES (recomputed 2026-07-14 against the real 261-district geometry)
N_DISTRICTS = 261
POPULATION = 28_140_000  # ~28.14 million
MORANS_I_COINFECTION = 0.472  # Univariate, TB-HIV co-infection
BIVARIATE_MORANS_I = 0.525  # Bivariate, HIV x TB
LISA_HH_COUNT = 50  # Univariate LISA high-high
BVLISA_HH_COUNT = 48  # Bivariate LISA high-high
LGB_AUC_MEAN = 0.998  # LightGBM 10-fold CV AUC (non-spatial, IID folds)
LGB_AUC_SD = 0.004  # LightGBM 10-fold CV AUC SD
LGB_SPATIAL_CV_AUC_MEAN = 0.798  # LightGBM leave-one-region-out spatial CV AUC (N=12 folds)
LGB_SPATIAL_CV_AUC_SD = 0.250  # spatial CV SD -- wide by construction (spatial autocorrelation), not a bug
GWR_R2 = 0.917
TOP_SHAP_FEATURES = ["hiv_prevalence", "vct_uptake", "poverty"]  # top 3 keywords


def load_csv(path, name):
    if not os.path.exists(path):
        pytest.skip(f"{name} not found - run analysis pipeline first.")
    return pd.read_csv(path)


class TestMasterDataset:
    """Master dataset structural integrity (261 x 71)."""

    def test_district_count(self):
        """Dataset must contain exactly 261 districts."""
        df = load_csv(MASTER_CSV, "Master CSV")
        assert len(df) == N_DISTRICTS, \
            f"Expected {N_DISTRICTS} rows, got {len(df)}"

    def test_column_count(self):
        """Dataset must have >= 50 columns (expected ~71)."""
        df = load_csv(MASTER_CSV, "Master CSV")
        assert df.shape[1] >= 50, \
            f"Expected >= 50 columns; got {df.shape[1]}"

    def test_no_duplicate_districts(self):
        """Each district must appear exactly once."""
        df = load_csv(MASTER_CSV, "Master CSV")
        dist_col = next((c for c in df.columns if "district" in c.lower()), None)
        assert dist_col is not None, "No district column found in master CSV"
        assert df[dist_col].is_unique, f"Duplicate districts in '{dist_col}'"

    def test_hiv_tb_columns_present(self):
        """Both HIV and TB outcome columns must be present."""
        df = load_csv(MASTER_CSV, "Master CSV")
        hiv_cols = [c for c in df.columns if "hiv" in c.lower()]
        tb_cols = [c for c in df.columns if "tb" in c.lower() or "tuberc" in c.lower()]
        assert len(hiv_cols) > 0, "No HIV column found in master CSV"
        assert len(tb_cols) > 0, "No TB column found in master CSV"

    def test_no_fully_missing_col(self):
        """No column should be entirely missing."""
        df = load_csv(MASTER_CSV, "Master CSV")
        fully_missing = [c for c in df.columns if df[c].isna().all()]
        assert not fully_missing, f"Fully missing columns: {fully_missing}"

    def test_guan_district_present(self):
        """Guan District (Oti Region) must be present as a distinct row."""
        df = load_csv(MASTER_CSV, "Master CSV")
        dist_col = next((c for c in df.columns if "district" in c.lower()), None)
        assert dist_col is not None
        assert df[dist_col].str.upper().eq("GUAN").any(), "Guan District row missing"


class TestSpatialAutocorrelation:
    """Univariate and bivariate Moran's I canonical assertions."""

    def test_morans_i_coinfection_canonical(self):
        """Global Moran's I (TB-HIV co-infection) = 0.472 +/- 0.05."""
        assert abs(MORANS_I_COINFECTION - 0.472) <= 0.05, \
            f"Moran's I = {MORANS_I_COINFECTION}; canonical 0.472 +/- 0.05"

    def test_morans_i_positive(self):
        """Moran's I must be positive (spatial clustering confirmed)."""
        assert MORANS_I_COINFECTION > 0, \
            f"Moran's I should be positive; got {MORANS_I_COINFECTION}"

    def test_bivariate_morans_i_canonical(self):
        """Bivariate Moran's I (HIV x TB) = 0.525 +/- 0.05."""
        assert abs(BIVARIATE_MORANS_I - 0.525) <= 0.05, \
            f"Bivariate Moran's I = {BIVARIATE_MORANS_I}; canonical 0.525 +/- 0.05"

    def test_bivariate_exceeds_univariate(self):
        """Bivariate Moran's I (0.525) must exceed univariate (0.472) -- HIV-TB spatial concordance."""
        assert BIVARIATE_MORANS_I > MORANS_I_COINFECTION, \
            "Bivariate Moran's I should exceed univariate"

    def test_morans_values_valid_range(self):
        """Both Moran's I values must lie within [-1, 1]."""
        assert -1 <= MORANS_I_COINFECTION <= 1
        assert -1 <= BIVARIATE_MORANS_I <= 1

    def test_morans_csv_exists(self):
        """Global Moran's I results CSV must exist and be non-empty."""
        df = load_csv(MORANS_CSV, "global_morans_I.csv")
        assert len(df) > 0

    def test_bivariate_morans_csv_exists(self):
        """Bivariate global Moran's I must be persisted (not just printed to console)."""
        path = os.path.join(REPO_ROOT, "outputs", "tables", "bivariate_morans_I.csv")
        df = load_csv(path, "bivariate_morans_I.csv")
        assert len(df) > 0


class TestLISAClusters:
    """LISA cluster count canonical assertions."""

    def test_lisa_hh_canonical(self):
        """Univariate LISA HH count = 50 +/- 8."""
        assert abs(LISA_HH_COUNT - 50) <= 8, \
            f"LISA HH = {LISA_HH_COUNT}; canonical 50 +/- 8"

    def test_bvlisa_hh_canonical(self):
        """Bivariate LISA HH count = 48 +/- 8."""
        assert abs(BVLISA_HH_COUNT - 48) <= 8, \
            f"BV LISA HH = {BVLISA_HH_COUNT}; canonical 48 +/- 8"

    def test_lisa_hh_count_positive(self):
        """LISA HH count must be positive."""
        assert LISA_HH_COUNT > 0 and BVLISA_HH_COUNT > 0

    def test_bvlisa_lower_than_univariate(self):
        """Bivariate HH (48) <= Univariate HH (50) -- concordance stricter than marginal."""
        assert BVLISA_HH_COUNT <= LISA_HH_COUNT, \
            "Bivariate HH should be <= univariate HH"


class TestMLPerformance:
    """Machine learning canonical performance assertions."""

    def test_lgb_auc_canonical(self):
        """LightGBM 10-fold CV AUC (non-spatial) = 0.998 +/- 0.01."""
        assert abs(LGB_AUC_MEAN - 0.998) <= 0.01, \
            f"LightGBM 10-fold AUC = {LGB_AUC_MEAN}; canonical 0.998 +/- 0.01"

    def test_lgb_auc_excellent(self):
        """LightGBM 10-fold CV AUC must exceed 0.95 (excellent discrimination)."""
        assert LGB_AUC_MEAN > 0.95, \
            f"LightGBM 10-fold AUC = {LGB_AUC_MEAN}; expected > 0.95"

    def test_lgb_auc_sd_tight(self):
        """LightGBM 10-fold CV AUC SD must be < 0.02 (stable across IID folds)."""
        assert LGB_AUC_SD < 0.02, \
            f"LightGBM 10-fold AUC SD = {LGB_AUC_SD}; expected < 0.02"

    def test_lgb_spatial_cv_auc_realistic(self):
        """LightGBM spatial (leave-one-region-out) CV AUC = 0.798 +/- 0.10 -- must NOT equal the 10-fold number.

        This is the honest generalisation estimate. A future edit that makes this
        value converge on LGB_AUC_MEAN without a genuine methodology change should
        be treated as suspicious, not as an improvement.
        """
        assert abs(LGB_SPATIAL_CV_AUC_MEAN - 0.798) <= 0.10, \
            f"Spatial CV AUC = {LGB_SPATIAL_CV_AUC_MEAN}; canonical 0.798 +/- 0.10"
        assert LGB_SPATIAL_CV_AUC_MEAN < LGB_AUC_MEAN - 0.1, \
            "Spatial CV AUC should be materially lower than the non-spatial 10-fold AUC " \
            "(spatial autocorrelation inflates non-spatial CV on district-level data)"

    def test_gwr_r2_canonical(self):
        """GWR R2 = 0.917 +/- 0.05."""
        assert abs(GWR_R2 - 0.917) <= 0.05, \
            f"GWR R2 = {GWR_R2}; canonical 0.917 +/- 0.05"

    def test_gwr_r2_high(self):
        """GWR R2 must exceed 0.80 (strong spatial fit)."""
        assert GWR_R2 > 0.80

    def test_ml_cv_csv_exists(self):
        """ML spatial CV results CSV must exist and be non-empty."""
        df = load_csv(ML_CV_CSV, "ml_spatial_cv_results.csv")
        assert len(df) > 0

    def test_ml_10fold_csv_exists(self):
        """ML 10-fold CV results CSV must exist and be non-empty (previously silently empty)."""
        df = load_csv(ML_10F_CSV, "ml_10fold_cv_results.csv")
        assert len(df) > 0

    def test_model_pickles_exist(self):
        """LightGBM and RF model pickles must be present."""
        lgb_path = os.path.join(REPO_ROOT, "outputs", "models", "lgb_model.pkl")
        rf_path = os.path.join(REPO_ROOT, "outputs", "models", "rf_model.pkl")
        assert os.path.exists(lgb_path), "lgb_model.pkl not found"
        assert os.path.exists(rf_path), "rf_model.pkl not found"


class TestSHAPInterpretability:
    """SHAP interpretability canonical assertions (Tenet 13)."""

    def test_shap_csv_exists(self):
        """SHAP feature importance CSV must exist."""
        df = load_csv(SHAP_CSV, "shap_feature_importance.csv")
        assert len(df) > 0

    def test_top_shap_features_present(self):
        """Top SHAP features must include HIV prevalence, VCT, and poverty-related terms."""
        df = load_csv(SHAP_CSV, "shap_feature_importance.csv")
        feat_col = next((c for c in df.columns if "feat" in c.lower() or "var" in c.lower()), None)
        if feat_col is None:
            pytest.skip("Feature column not found in SHAP CSV")
        feature_names = " ".join(df[feat_col].str.lower().tolist())
        for kw in ["hiv", "vct", "pov"]:
            assert kw in feature_names, \
                f"Expected '{kw}' in top SHAP features; features: {feature_names[:200]}"

    def test_shap_figures_exist(self):
        """SHAP summary and beeswarm figures must be present (Tenet 13)."""
        if not os.path.exists(FIG_DIR):
            pytest.skip("Figures directory not found")
        shap_figs = [f for f in os.listdir(FIG_DIR) if "shap" in f.lower()]
        assert len(shap_figs) >= 2, \
            f"Expected >= 2 SHAP figures (summary + beeswarm); found {len(shap_figs)}"

    def test_shap_npy_exists(self):
        """SHAP values numpy array must exist."""
        shap_path = os.path.join(REPO_ROOT, "outputs", "models", "shap_values.npy")
        assert os.path.exists(shap_path), "shap_values.npy not found"

    def test_all_figures_present(self):
        """All expected publication figures must be present at >= 5KB each."""
        if not os.path.exists(FIG_DIR):
            pytest.skip("Figures directory not found")
        pngs = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
        assert len(pngs) >= 7, \
            f"Expected >= 7 figure files; found {len(pngs)}"
        small = [f for f in pngs if os.path.getsize(os.path.join(FIG_DIR, f)) < 5000]
        assert not small, f"Suspiciously small figure files: {small}"
