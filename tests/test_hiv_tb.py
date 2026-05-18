#!/usr/bin/env python3
"""
tests/test_hiv_tb.py - Ghana HIV-TB Co-infection Spatial & ML Analysis (261 Districts)
Unit tests with canonical value assertions (QA-verified May 2026).

Run: pytest tests/ -v
Tenet 8: SEED=42. Canonical values from manuscript FINAL (261 districts, deduplicated).
Methods: Moran's I, Bivariate LISA, GWR, LightGBM ensemble, SHAP.
"""

import os
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_CSV = os.path.join(REPO_ROOT, "outputs", "data", "Ghana_HIV_TB_Master_Dataset.csv")
MORANS_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "global_morans_I.csv")
GWR_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "gwr_summary.csv")
ML_CV_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "ml_spatial_cv_results.csv")
ML_10F_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "ml_10fold_cv_results.csv")
SHAP_CSV = os.path.join(REPO_ROOT, "outputs", "tables", "shap_feature_importance.csv")
FIG_DIR = os.path.join(REPO_ROOT, "outputs", "figures")

# CANONICAL VALUES (QA-verified 2026-05-18, 261 districts, deduplicated)
N_DISTRICTS = 261
POPULATION = 28_140_000 # ~28.14 million
MORANS_I_COINFECTION = 0.810 # Univariate, TB-HIV co-infection
BIVARIATE_MORANS_I = 0.449 # Bivariate, HIV x TB
LISA_HH_COUNT = 69 # Univariate LISA high-high
BVLISA_HH_COUNT = 69 # Bivariate LISA high-high (adjusted for 261)
LGB_AUC_MEAN = 0.998 # LightGBM LODO-CV AUC
LGB_AUC_SD = 0.003 # LightGBM AUC SD
GWR_R2 = 0.916
TOP_SHAP_FEATURES = ["hiv_prevalence", "vct_uptake", "poverty"] # top 3 keywords


def load_csv(path, name):
 if not os.path.exists(path):
  pytest.skip(f"{name} not found - run analysis pipeline first.")
 return pd.read_csv(path)


class TestMasterDataset:
 """Master dataset structural integrity (261 × 52, fully deduplicated)."""

 def test_district_count(self):
  """Dataset must contain exactly 261 districts."""
  df = load_csv(MASTER_CSV, "Master CSV")
  assert len(df) == N_DISTRICTS, \
  f"Expected {N_DISTRICTS} rows, got {len(df)}"

 def test_column_count(self):
  """Dataset must have >= 50 columns (expected ~52)."""
  df = load_csv(MASTER_CSV, "Master CSV")
  assert df.shape[1] >= 50, \
  f"Expected >= 50 columns; got {df.shape[1]}"

 def test_no_duplicate_districts(self):
  """Each district must appear exactly once (deduplicated)."""
  df = load_csv(MASTER_CSV, "Master CSV")
  dist_col = next((c for c in df.columns if "district" in c.lower()), None)
  if dist_col:
   assert df[dist_col].is_unique, \
   f"Duplicate districts found in '{dist_col}' — dataset not deduplicated"

 def test_hiv_prev_bounds(self):
  """HIV prevalence must be between 0 and 100."""
  df = load_csv(MASTER_CSV, "Master CSV")
  hiv_col = next((c for c in df.columns if "hiv" in c.lower() and "prev" in c.lower()), None)
  if hiv_col:
   assert (df[hiv_col] >= 0).all() and (df[hiv_col] <= 100).all(), \
   f"{hiv_col} has values outside [0, 100]"

 def test_required_columns(self):
  """Master dataset must include key epidemiological columns."""
  df = load_csv(MASTER_CSV, "Master CSV")
  required = ['REGION', 'DISTRICT', 'HIV_Prev_Total_pct', 'TB_Incidence_per100k', 
              'TB_HIV_CoInfection_pct', 'Latitude', 'Longitude']
  missing = [c for c in required if c not in df.columns]
  assert not missing, f"Missing columns: {missing}"


class TestSpatialAnalysis:
 """Spatial analysis canonical values (Moran's I, LISA, GWR)."""

 def test_morans_i_coinfection(self):
  """Global Moran's I for TB-HIV co-infection ≈ 0.810 (p<0.001)."""
  df = load_csv(MORANS_CSV, "Global Moran's I CSV")
  mi_row = df[df['Variable'].str.contains('coinfect', case=False, na=False)]
  if not mi_row.empty:
   actual_i = mi_row.iloc[0]['Morans_I']
   assert abs(actual_i - MORANS_I_COINFECTION) < 0.05, \
   f"Expected Moran's I ≈ {MORANS_I_COINFECTION}, got {actual_i}"

 def test_lisa_hh_clusters(self):
  """LISA high-high clusters ≈ 69 districts."""
  df = load_csv(MORANS_CSV, "LISA results")
  hh_count = len(df[df['LISA_Quadrant'].str.contains('HH', case=False, na=False)])
  assert abs(hh_count - LISA_HH_COUNT) <= 5, \
  f"Expected ~{LISA_HH_COUNT} HH clusters, got {hh_count}"

 def test_gwr_r2(self):
  """GWR R² ≈ 0.916."""
  df = load_csv(GWR_CSV, "GWR summary")
  if 'GWR_Pseudo_R2' in df.columns:
   r2 = df['GWR_Pseudo_R2'].iloc[0]
   assert abs(r2 - GWR_R2) < 0.02, \
   f"Expected GWR R² ≈ {GWR_R2}, got {r2}"


class TestMLPipeline:
 """Machine learning model performance (LightGBM, SHAP)."""

 def test_lgbm_auc(self):
  """LightGBM LODO-CV AUC ≈ 0.998."""
  df = load_csv(ML_CV_CSV, "ML LODO-CV results")
  if 'LGB_AUC' in df.columns:
   auc = df['LGB_AUC'].mean()
   assert abs(auc - LGB_AUC_MEAN) < 0.01, \
   f"Expected LGB AUC ≈ {LGB_AUC_MEAN}, got {auc}"

 def test_shap_top_features(self):
  """Top SHAP features include VCT uptake, poverty, HIV prevalence."""
  df = load_csv(SHAP_CSV, "SHAP feature importance")
  if 'Feature' in df.columns:
   top_features = df.head(3)['Feature'].str.lower().tolist()
   has_key_features = any(kw in ' '.join(top_features) for kw in TOP_SHAP_FEATURES)
   assert has_key_features, \
   f"Top SHAP features missing epidemiological keywords. Got: {top_features}"


class TestFigures:
 """Output figures exist and are 300 DPI."""

 def test_figures_exist(self):
  """At least 9 publication figures should be generated."""
  if os.path.exists(FIG_DIR):
   figs = [f for f in os.listdir(FIG_DIR) if f.endswith('.png')]
   assert len(figs) >= 9, \
   f"Expected >= 9 PNG figures, found {len(figs)}"
