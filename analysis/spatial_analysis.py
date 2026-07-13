"""
Spatial Epidemiology Pipeline — HIV-TB Co-infection Ghana 260 Districts
========================================================================
Global Moran's I, Bivariate LISA, Getis-Ord Gi*, GWR, Spatial Error Model
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran, Moran_Local, Moran_BV, Moran_Local_BV
from esda.getisord import G_Local
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUT = Path(__file__).resolve().parent.parent / 'outputs'
TAB = OUT / 'tables'
TAB.mkdir(parents=True, exist_ok=True)

print('='*70)
print('SPATIAL ANALYSIS PIPELINE — HIV-TB Co-infection Ghana 261 Districts')
print('='*70)

gdf = gpd.read_file(OUT / 'data' / 'ghana_261_districts_hiv_tb.geojson')
print(f'\nLoaded: {len(gdf)} districts')

# Get polygon-based weights (Queen contiguity)
print('\n[1/6] Computing spatial weight matrices...')
gdf_proj = gdf.to_crs('EPSG:32630') # UTM Zone 30N for Ghana
W_queen = Queen.from_dataframe(gdf_proj, use_index=False, silence_warnings=True)
W_queen.transform = 'r'
W_rook = Rook.from_dataframe(gdf_proj, use_index=False, silence_warnings=True)
W_rook.transform = 'r'
# KNN-5 for Moran's I (more stable with small islands)
centroids = gdf_proj.geometry.centroid
coords = np.column_stack([centroids.x, centroids.y])
W_knn = KNN.from_array(coords, k=5)
W_knn.transform = 'r'
print(f' Queen: avg neighbours = {W_queen.mean_neighbors:.1f}')
print(f' Rook: avg neighbours = {W_rook.mean_neighbors:.1f}')
print(f' KNN-5: avg neighbours = {W_knn.mean_neighbors:.1f}')

# ============================================================
# 2. GLOBAL MORAN'S I for key variables
# ============================================================
print('\n[2/6] Global Moran\'s I...')
vars_to_test = [
 'HIV_Prev_Total_pct', 'TB_Incidence_per100k', 'TB_HIV_CoInfection_pct',
 'ART_Coverage_pct', 'VCT_Uptake_pct', 'Poverty_Intensity_pct',
 'Illiteracy_Rate_pct', 'Uninsurance_Rate_pct', 'Doctors_per10k',
 'Condom_Use_pct', 'Ever_Tested_HIV_pct',
]

moran_results = []
for var in vars_to_test:
 y = gdf[var].values
 m = Moran(y, W_knn, permutations=999)
 moran_results.append({
 'Variable': var,
 "Moran's I": round(m.I, 4),
 'Expected I': round(m.EI, 4),
 'z-score': round(m.z_sim, 3),
 'p-value': round(m.p_sim, 4),
 'Autocorrelation': 'Positive clustering' if m.I > 0 and m.p_sim < 0.05 else
 ('Negative' if m.I < 0 and m.p_sim < 0.05 else 'Random'),
 })
moran_df = pd.DataFrame(moran_results)
moran_df.to_csv(TAB / 'global_morans_I.csv', index=False)
print(moran_df.to_string(index=False))

# ============================================================
# 3. LOCAL MORAN'S I (LISA) — HIV-TB co-infection
# ============================================================
print('\n[3/6] Local Moran\'s I (LISA) — TB-HIV co-infection...')
y_coin = gdf['TB_HIV_CoInfection_pct'].values
lisa = Moran_Local(y_coin, W_rook, permutations=999, seed=42)

gdf['LISA_I'] = lisa.Is
gdf['LISA_quadrant'] = lisa.q # 1=HH, 2=LH, 3=LL, 4=HL
gdf['LISA_p_sim'] = lisa.p_sim
gdf['LISA_sig'] = (lisa.p_sim < 0.05).astype(int)
# Cluster label combining significance + quadrant
q_map = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low'}
gdf['LISA_cluster'] = np.where(
 gdf['LISA_sig'] == 1,
 gdf['LISA_quadrant'].map(q_map),
 'Not Significant'
)

cluster_counts = gdf['LISA_cluster'].value_counts()
print(cluster_counts)

# ============================================================
# 4. BIVARIATE LISA — HIV × TB
# ============================================================
print('\n[4/6] Bivariate LISA — HIV prevalence × TB incidence...')
x_hiv = gdf['HIV_Prev_Total_pct'].values
y_tb = gdf['TB_Incidence_per100k'].values

# Global bivariate Moran's I
m_bv = Moran_BV(x_hiv, y_tb, W_rook, permutations=999)
print(f" Global Bivariate Moran's I (HIV×TB): I={m_bv.I:.4f}, p={m_bv.p_sim:.4f}")
pd.DataFrame([{
 'Variable_X': 'HIV_Prev_Total_pct', 'Variable_Y': 'TB_Incidence_per100k',
 "Bivariate Moran's I": m_bv.I, 'p-value': m_bv.p_sim,
}]).to_csv(TAB / 'bivariate_morans_I.csv', index=False)

# Local bivariate LISA
lisa_bv = Moran_Local_BV(x_hiv, y_tb, W_rook, permutations=999, seed=42)
gdf['BvLISA_I'] = lisa_bv.Is
gdf['BvLISA_quadrant'] = lisa_bv.q
gdf['BvLISA_p'] = lisa_bv.p_sim
gdf['BvLISA_sig'] = (lisa_bv.p_sim < 0.05).astype(int)
gdf['BvLISA_cluster'] = np.where(
 gdf['BvLISA_sig'] == 1,
 gdf['BvLISA_quadrant'].map(q_map),
 'Not Significant'
)
print(gdf['BvLISA_cluster'].value_counts())

# ============================================================
# 5. GETIS-ORD Gi* — HIV-TB co-infection hotspots
# ============================================================
print('\n[5/6] Getis-Ord Gi* hotspot analysis...')
g = G_Local(y_coin, W_rook, transform='r', permutations=999, seed=42)
gdf['Gi_z'] = g.Zs
gdf['Gi_p'] = g.p_sim
gdf['Gi_cluster'] = 'Not Significant'
gdf.loc[(gdf['Gi_z'] > 2.58) & (gdf['Gi_p'] < 0.01), 'Gi_cluster'] = 'Hot Spot 99%'
gdf.loc[(gdf['Gi_z'] > 1.96) & (gdf['Gi_z'] <= 2.58) & (gdf['Gi_p'] < 0.05), 'Gi_cluster'] = 'Hot Spot 95%'
gdf.loc[(gdf['Gi_z'] > 1.65) & (gdf['Gi_z'] <= 1.96) & (gdf['Gi_p'] < 0.10), 'Gi_cluster'] = 'Hot Spot 90%'
gdf.loc[(gdf['Gi_z'] < -2.58) & (gdf['Gi_p'] < 0.01), 'Gi_cluster'] = 'Cold Spot 99%'
gdf.loc[(gdf['Gi_z'] < -1.96) & (gdf['Gi_z'] >= -2.58) & (gdf['Gi_p'] < 0.05), 'Gi_cluster'] = 'Cold Spot 95%'
gdf.loc[(gdf['Gi_z'] < -1.65) & (gdf['Gi_z'] >= -1.96) & (gdf['Gi_p'] < 0.10), 'Gi_cluster'] = 'Cold Spot 90%'
print(gdf['Gi_cluster'].value_counts())

# ============================================================
# 6. SPATIAL ERROR MODEL (SEM) + OLS baseline
# ============================================================
print('\n[6/6] Spatial Error Model (SEM)...')
try:
 from spreg import OLS, ML_Error
 predictors = ['HIV_Prev_Total_pct', 'Poverty_Intensity_pct',
 'Illiteracy_Rate_pct', 'Uninsurance_Rate_pct',
 'Doctors_per10k', 'VCT_Uptake_pct', 'Condom_Use_pct',
 'Youth_Dependency_Ratio']
 y = gdf['TB_HIV_CoInfection_pct'].values.reshape(-1, 1)
 X = gdf[predictors].values

 # OLS baseline
 ols = OLS(y, X, w=W_rook, name_y='TB_HIV_CoInfection', name_x=predictors,
 spat_diag=True, moran=True)
 print(f"OLS R²: {ols.r2:.4f} | AIC: {ols.aic:.1f}")
 print(f"LM-Error: {ols.lm_error[0]:.3f}, p={ols.lm_error[1]:.4f}")
 print(f"LM-Lag: {ols.lm_lag[0]:.3f}, p={ols.lm_lag[1]:.4f}")

 # Persist OLS diagnostics + coefficients immediately -- do not wait for SEM,
 # which can fail independently and previously left these unsaved.
 pd.DataFrame([{
 'R2': ols.r2, 'AIC': ols.aic,
 'LM_Error_stat': ols.lm_error[0], 'LM_Error_p': ols.lm_error[1],
 'LM_Lag_stat': ols.lm_lag[0], 'LM_Lag_p': ols.lm_lag[1],
 }]).to_csv(TAB / 'ols_diagnostics.csv', index=False)
 pd.DataFrame({
 'Variable': ['Intercept'] + predictors,
 'Coefficient': ols.betas.flatten(),
 'Std_Error': ols.std_err,
 't_stat': ols.t_stat[0] if hasattr(ols, 't_stat') else [np.nan]*(len(predictors)+1),
 }).to_csv(TAB / 'ols_baseline.csv', index=False)

 # Spatial Error Model
 sem = ML_Error(y, X, w=W_rook, name_y='TB_HIV_CoInfection',
 name_x=predictors)
 print(f"\nSEM Pseudo-R²: {sem.pr2:.4f}")
 print(f"SEM λ (spatial error): {sem.lam[0]:.4f}")

 # Save coefficients
 sem_results = pd.DataFrame({
 'Variable': ['Intercept'] + predictors + ['lambda'],
 'Coefficient': list(sem.betas.flatten()) + [sem.lam[0]],
 'Std_Error': list(sem.std_err) + [sem.std_err[-1]],
 })
 sem_results['z_stat'] = sem_results['Coefficient'] / sem_results['Std_Error']
 sem_results['p_value'] = 2 * (1 - pd.Series(sem_results['z_stat']).abs().apply(
 lambda z: 0.5 * (1 + np.math.erf(z / np.sqrt(2)))))
 sem_results.to_csv(TAB / 'spatial_error_model.csv', index=False)
 print(sem_results.to_string(index=False))
except Exception as e:
 print(f'SEM error: {e}')

# ============================================================
# 7. GEOGRAPHICALLY WEIGHTED REGRESSION (GWR)
# ============================================================
print('\n[7/7] Geographically Weighted Regression (GWR)...')
try:
 from mgwr.gwr import GWR
 from mgwr.sel_bw import Sel_BW

 gwr_predictors = ['HIV_Prev_Total_pct', 'Poverty_Intensity_pct',
 'Uninsurance_Rate_pct', 'Doctors_per10k']
 Xg = gdf[gwr_predictors].values
 yg = gdf['TB_HIV_CoInfection_pct'].values.reshape(-1, 1)
 coords_list = list(zip(coords[:, 0], coords[:, 1]))

 sel = Sel_BW(coords_list, yg, Xg)
 bw = sel.search()
 print(f' Optimal bandwidth: {bw}')

 gwr_m = GWR(coords_list, yg, Xg, bw)
 gwr_res = gwr_m.fit()
 print(f' GWR global R²: {gwr_res.R2:.4f}, AICc: {gwr_res.aicc:.1f}')
 print(f' GWR mean local R²: {gwr_res.localR2.mean():.4f}')

 # Persist global fit stats -- distinct from the per-district local R2 mean;
 # these are two different statistics and must not be conflated in reporting.
 pd.DataFrame([{
 'Bandwidth': bw, 'Global_R2': gwr_res.R2, 'AICc': gwr_res.aicc,
 'Mean_Local_R2': gwr_res.localR2.mean(), 'Min_Local_R2': gwr_res.localR2.min(),
 'Max_Local_R2': gwr_res.localR2.max(), 'SD_Local_R2': gwr_res.localR2.std(),
 }]).to_csv(TAB / 'gwr_global_fit.csv', index=False)

 # Store local coefficients
 for i, var in enumerate(gwr_predictors):
  gdf[f'GWR_coef_{var}'] = gwr_res.params[:, i + 1]
  gdf['GWR_local_R2'] = gwr_res.localR2

 # Save GWR summary
  gwr_summary = pd.DataFrame({
  'Variable': ['Intercept'] + gwr_predictors,
  'Mean_Coef': gwr_res.params.mean(axis=0),
  'Min_Coef': gwr_res.params.min(axis=0),
  'Max_Coef': gwr_res.params.max(axis=0),
  'SD_Coef': gwr_res.params.std(axis=0),
  })
  gwr_summary.to_csv(TAB / 'gwr_summary.csv', index=False)
  print(gwr_summary.to_string(index=False))
except Exception as e:
 print(f'GWR error: {e}')

# ============================================================
# SAVE ENRICHED GEODATAFRAME
# ============================================================
gdf.to_file(OUT / 'data' / 'ghana_260_districts_spatial_results.geojson', driver='GeoJSON')
# Also save as CSV (no geometry)
gdf.drop(columns='geometry').to_csv(OUT / 'data' / 'spatial_results_260_districts.csv', index=False)
print('\nSpatial analysis complete.')
