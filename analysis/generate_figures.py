"""
Publication Figures — HIV-TB Co-infection Ghana 261 Districts
==============================================================
Generates 8 publication-ready figures at 300 DPI
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns
from pathlib import Path
import shap
import warnings
import textwrap
warnings.filterwarnings('ignore')

OUT = Path(__file__).resolve().parent.parent / 'outputs'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
 'font.family': 'DejaVu Serif',
 'font.size': 11,
 'axes.spines.top': False,
 'axes.spines.right': False,
 'figure.facecolor': 'none',
 'axes.facecolor': 'none',
 'savefig.dpi': 300,
 'savefig.bbox': 'tight',
 'savefig.transparent': True,
 'savefig.facecolor': 'none',
 'savefig.edgecolor': 'none',
})
TITLE_KW = dict(fontsize=13, fontweight='bold', color='#000', pad=12)
CAP_KW = dict(ha='center', fontsize=12, style='italic', color='#333333')

GHANA_UTM_EPSG = 32630  # UTM Zone 30N -- metres, correct for scale bars/bearings over Ghana


def add_map_essentials(ax, source_note=None):
 """Add a neatline frame, north arrow, and scale bar to a choropleth map axis.
 Assumes the plotted geometry is in a projected (metric) CRS -- see GHANA_UTM_EPSG."""
 # Neatline frame: keep a clean border, drop tick marks/labels (equivalent to axis('off')
 # for a map reader, but preserves a printable frame around the plate).
 ax.set_xticks([])
 ax.set_yticks([])
 for spine in ax.spines.values():
  spine.set_visible(True)
  spine.set_edgecolor('#4d4d4d')
  spine.set_linewidth(0.8)

 # Scale bar (metres, auto-converts to km) -- lower right, clear of the categorical
 # legends this project places at lower left.
 ax.add_artist(ScaleBar(
  1, units='m', dimension='si-length', location='lower right',
  box_alpha=0.75, color='#1a1a1a', box_color='white',
  font_properties={'size': 8}, scale_loc='top', border_pad=0.5, pad=0.4,
 ))

 # North arrow -- upper right, inside the frame.
 x, y0, y1 = 0.94, 0.80, 0.92
 ax.annotate('', xy=(x, y1), xytext=(x, y0), xycoords=ax.transAxes,
  arrowprops=dict(arrowstyle='-|>', color='#1a1a1a', linewidth=1.6,
  mutation_scale=16))
 ax.text(x, y1 + 0.015, 'N', transform=ax.transAxes, ha='center', va='bottom',
  fontsize=11, fontweight='bold', color='#1a1a1a')

 # Optional small source/CRS text box -- bottom edge, clear of the scale bar.
 if source_note:
  ax.text(0.02, 0.02, source_note, transform=ax.transAxes, ha='left', va='bottom',
  fontsize=7.5, color='#333333', style='italic',
  bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#999999',
  linewidth=0.5, alpha=0.8))


MAP_SOURCE_NOTE = 'Ghana LGA post-2018\n261 districts\nCRS: UTM Zone 30N'


def save_transparent_figure(fig, filename, **kwargs):
 """Save a publication PNG with a transparent canvas and transparent axis panels."""
 fig.patch.set_alpha(0)
 for ax in fig.axes:
  ax.patch.set_alpha(0)
 save_kwargs = {
  'dpi': 300,
  'transparent': True,
  'facecolor': 'none',
  'edgecolor': 'none',
 }
 save_kwargs.update(kwargs)
 fig.savefig(FIG / filename, **save_kwargs)


def add_caption(fig, text, y=0.012, width=105):
 """Add a wrapped caption inside the exported figure boundary."""
 fig.text(0.5, y, textwrap.fill(text, width=width), **CAP_KW)

gdf = gpd.read_file(OUT / 'data' / 'ghana_261_final_results.geojson')
gdf = gdf.to_crs(GHANA_UTM_EPSG)  # project to metres so bearings/scale bars are geographically correct
print(f'Loaded: {len(gdf)} districts')

# ============================================================
# FIGURE 1 — Study Area & Disease Burden Choropleths (2×2)
# ============================================================
print('[1/8] Fig 1: Disease burden choropleths...')
fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.8))
panels = [
 ('HIV_Prev_Total_pct', 'A. HIV Prevalence (%)', 'OrRd'),
 ('TB_Incidence_per100k', 'B. TB Incidence (per 100,000)', 'YlOrBr'),
 ('TB_HIV_CoInfection_pct', 'C. TB-HIV Co-infection (%)', 'Reds'),
 ('ART_Coverage_pct', 'D. ART Coverage (%)', 'BuGn'),
]
for i, (ax, (var, title, cmap)) in enumerate(zip(axes.flat, panels)):
 gdf.plot(column=var, cmap=cmap, legend=True, ax=ax, edgecolor='white',
 linewidth=0.3, legend_kwds={'shrink': 0.6})
 ax.set_title(title, **TITLE_KW)
 add_map_essentials(ax, source_note=MAP_SOURCE_NOTE if i == 0 else None)
plt.tight_layout(rect=[0, 0.04, 1, 1])
add_caption(fig, 'Figure 1. Spatial distribution of HIV prevalence, TB incidence, TB-HIV co-infection, and ART coverage across 261 districts in Ghana.')
save_transparent_figure(fig, 'Figure_1_disease_burden.png')
plt.close()

# ============================================================
# FIGURE 2 — LISA & Getis-Ord Hotspot Maps
# ============================================================
print('[2/8] Fig 2: LISA + Getis-Ord...')
fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.8))

# Panel A: LISA for TB-HIV co-infection
lisa_cmap = {'High-High': '#d7191c', 'Low-Low': '#2c7bb6',
 'High-Low': '#fdae61', 'Low-High': '#abd9e9',
 'Not Significant': '#f0f0f0'}
ax = axes[0, 0]
legend_patches = []
for cluster, color in lisa_cmap.items():
 sub = gdf[gdf['LISA_cluster'] == cluster]
 if len(sub) > 0:
  sub.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2)
  legend_patches.append(mpatches.Patch(facecolor=color, edgecolor='#888888',
  label=f'{cluster} (n={len(sub)})'))
ax.set_title('A. LISA Cluster Map — TB-HIV Co-infection', **TITLE_KW)
# geopandas polygon plots build a PatchCollection, which matplotlib's ax.legend()
# cannot auto-derive handles for (silently produces an empty legend box) -- use
# explicit proxy Patch handles instead.
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, frameon=True)
add_map_essentials(ax)

# Panel B: Bivariate LISA (HIV × TB)
ax = axes[0, 1]
legend_patches = []
for cluster, color in lisa_cmap.items():
 sub = gdf[gdf['BvLISA_cluster'] == cluster]
 if len(sub) > 0:
  sub.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2)
  legend_patches.append(mpatches.Patch(facecolor=color, edgecolor='#888888',
  label=f'{cluster} (n={len(sub)})'))
ax.set_title('B. Bivariate LISA — HIV × TB', **TITLE_KW)
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, frameon=True)
add_map_essentials(ax)

# Panel C: Getis-Ord Gi*
ax = axes[1, 0]
gi_cmap = {'Hot Spot 99%': '#d73027', 'Hot Spot 95%': '#fc8d59', 'Hot Spot 90%': '#fee090',
 'Cold Spot 99%': '#313695', 'Cold Spot 95%': '#4575b4', 'Cold Spot 90%': '#91bfdb',
 'Not Significant': '#f0f0f0'}
legend_patches = []
for cluster, color in gi_cmap.items():
 sub = gdf[gdf['Gi_cluster'] == cluster]
 if len(sub) > 0:
  sub.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2)
  legend_patches.append(mpatches.Patch(facecolor=color, edgecolor='#888888',
  label=f'{cluster} (n={len(sub)})'))
ax.set_title('C. Getis-Ord Gi* Hotspots', **TITLE_KW)
ax.legend(handles=legend_patches, loc='lower left', fontsize=9, frameon=True)
add_map_essentials(ax)

# Panel D: GWR local R²
ax = axes[1, 1]
if 'GWR_local_R2' in gdf.columns:
 gdf.plot(column='GWR_local_R2', cmap='viridis', legend=True, ax=ax,
 edgecolor='white', linewidth=0.2, legend_kwds={'shrink': 0.6})
 ax.set_title('D. GWR Local R² (Model Fit)', **TITLE_KW)
 add_map_essentials(ax)
else:
 ax.axis('off')

plt.tight_layout(rect=[0, 0.04, 1, 1])
add_caption(fig, 'Figure 2. Spatial cluster analyses. A: Univariate LISA for TB-HIV co-infection; B: bivariate LISA (HIV x TB); C: Getis-Ord Gi* hotspots; D: GWR local R2 showing spatial non-stationarity.')
save_transparent_figure(fig, 'Figure_2_spatial_clusters.png')
plt.close()

# ============================================================
# FIGURE 3 — Global Moran's I bar plot with significance
# ============================================================
print('[3/8] Fig 3: Global Moran\'s I...')
moran_df = pd.read_csv(OUT / 'tables' / 'global_morans_I.csv')
moran_df = moran_df.sort_values("Moran's I", ascending=True)
fig, ax = plt.subplots(figsize=(7.8, 5.2))
colors = ['#d7191c' if p < 0.001 else ('#fdae61' if p < 0.05 else '#bdbdbd')
 for p in moran_df['p-value']]
bars = ax.barh(moran_df['Variable'], moran_df["Moran's I"], color=colors,
 edgecolor='black', linewidth=0.5)
morans_col = "Moran's I"
for bar, (_, row) in zip(bars, moran_df.iterrows()):
 i_val = row[morans_col]
 p_val = row['p-value']
 ax.text(i_val + 0.01, bar.get_y() + bar.get_height()/2,
 'I={:.3f} (p={:.3f})'.format(i_val, p_val),
 va='center', fontsize=9)
ax.set_xlabel("Global Moran's I", fontsize=12, fontweight='semibold')
ax.set_title("Global Moran's I — Spatial Autocorrelation Across Key Indicators", **TITLE_KW)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlim(-0.1, 1.0)
plt.tight_layout(rect=[0, 0.06, 1, 1])
add_caption(fig, "Figure 3. Global Moran's I values across 11 HIV, TB, and socioeconomic indicators (KNN-5 weight matrix, 999 permutations).")
save_transparent_figure(fig, 'Figure_3_morans_I.png')
plt.close()

# ============================================================
# FIGURE 4 — ML Model Comparison (ROC + Performance bar)
# ============================================================
print('[4/8] Fig 4: ML comparison...')
perf = pd.read_csv(OUT / 'tables' / 'ml_test_set_performance.csv')
cv = pd.read_csv(OUT / 'tables' / 'ml_10fold_cv_results.csv')
spatial_cv = pd.read_csv(OUT / 'tables' / 'ml_spatial_cv_results.csv')

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8))

# Panel A: CV AUC with error bars
ax = axes[0]
x = np.arange(len(cv))
ax.bar(x, cv['AUC_mean'], yerr=cv['AUC_SD'], capsize=5,
 color=['#3182bd', '#e6550d', '#31a354', '#756bb1'][:len(cv)],
 edgecolor='black', linewidth=0.6)
for i, (_, row) in enumerate(cv.iterrows()):
 ax.text(i, row['AUC_mean'] + 0.02, f"{row['AUC_mean']:.3f}",
 ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cv['Model'], rotation=15, ha='right')
ax.set_ylim(0.7, 1.05)
ax.set_ylabel('AUC-ROC (mean ± SD)', fontsize=12, fontweight='semibold')
ax.set_title('A. Random 10-fold CV', **TITLE_KW)
ax.axhline(0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Random')

# Panel B: Spatial (leave-one-region-out) CV AUC with error bars -- directly comparable to Panel A
ax = axes[1]
sp = spatial_cv.dropna(subset=['Spatial_CV_AUC_mean']).reset_index(drop=True)
x = np.arange(len(sp))
ax.bar(x, sp['Spatial_CV_AUC_mean'], yerr=sp['Spatial_CV_AUC_SD'], capsize=5,
 color=['#3182bd', '#e6550d', '#31a354', '#756bb1'][:len(sp)],
 edgecolor='black', linewidth=0.6)
for i, (_, row) in enumerate(sp.iterrows()):
 ax.text(i, row['Spatial_CV_AUC_mean'] + row['Spatial_CV_AUC_SD'] + 0.02,
 f"{row['Spatial_CV_AUC_mean']:.3f}", ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sp['Model'], rotation=15, ha='right')
ax.set_ylim(0.0, 1.05)
ax.set_ylabel('AUC-ROC (mean ± SD)', fontsize=12, fontweight='semibold')
ax.set_title('B. Spatial CV (12 regional folds)', **TITLE_KW)
ax.axhline(0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Random')

plt.tight_layout(rect=[0, 0.13, 1, 1])
add_caption(fig, 'Figure 4. ML model performance. A: 10-fold stratified CV AUC with SD error bars. B: leave-one-region-out spatial CV AUC with SD error bars across 12 regional folds. Spatial CV is the honest generalisation estimate.', width=100)
save_transparent_figure(fig, 'Figure_4_ml_performance.png')
plt.close()

# ============================================================
# FIGURE 5 — SHAP Summary Plot
# ============================================================
print('[5/8] Fig 5: SHAP importance + summary...')
shap_vals = np.load(OUT / 'models' / 'shap_values.npy')
X_test_arr = np.load(OUT / 'models' / 'shap_X_test.npy')
shap_imp = pd.read_csv(OUT / 'tables' / 'shap_feature_importance.csv')
features = shap_imp['Feature'].tolist()

# Clean labels
LABEL_MAP = {
 'HIV_Prev_Total_pct': 'HIV Prevalence (Total)',
 'HIV_Prev_Women_pct': 'HIV Prevalence (Women)',
 'HIV_Prev_Men_pct': 'HIV Prevalence (Men)',
 'HIV_Awareness_Women_pct': 'HIV Awareness',
 'Condom_Use_pct': 'Condom Use',
 'High_Risk_Sex_pct': 'High-Risk Sex',
 'Ever_Tested_HIV_pct': 'Ever Tested for HIV',
 'Know_Where_Test_pct': 'Knows Where to Test',
 'Accepting_Attitudes_pct': 'Accepting Attitudes',
 'Poverty_Incidence_pct': 'Poverty Incidence',
 'Poverty_Intensity_pct': 'Poverty Intensity',
 'Unemployment_Rate_pct': 'Unemployment Rate',
 'Illiteracy_Rate_pct': 'Illiteracy Rate',
 'Uninsurance_Rate_pct': 'Uninsurance Rate',
 'Youth_Dependency_Ratio': 'Youth Dependency',
 'Sex_Ratio_15_64': 'Sex Ratio (15-64)',
 'Sexually_Active_Pop_pct': 'Sexually Active Pop.',
 'TB_Incidence_per100k': 'TB Incidence (/100k)',
 'ART_Coverage_pct': 'ART Coverage',
 'VCT_Uptake_pct': 'VCT Uptake',
 'Doctors_per10k': 'Doctors / 10,000',
 'Nurses_per10k': 'Nurses / 10,000',
 'OOP_Expenditure_pct': 'OOP Expenditure',
 'TB_Treatment_Success_pct': 'TB Treatment Success',
}

# SHAP summary bar plot (top 15)
fig, ax = plt.subplots(figsize=(7.8, 6.0))
top15 = shap_imp.head(15).iloc[::-1]
colors_grad = plt.cm.Blues(np.linspace(0.4, 0.9, len(top15)))
bars = ax.barh([LABEL_MAP.get(f, f) for f in top15['Feature']], top15['Mean_abs_SHAP'],
 color=colors_grad, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, top15['Mean_abs_SHAP']):
 ax.text(val + 0.03, bar.get_y() + bar.get_height()/2,
 f'{val:.3f}', va='center', fontsize=10)
ax.set_xlabel('Mean |SHAP value|', fontsize=13, fontweight='semibold', labelpad=8)
ax.set_title('Top 15 Feature Importance (LightGBM)', **TITLE_KW)
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 1])
add_caption(fig, 'Figure 5. SHAP-derived mean absolute feature importance for HIV-TB co-infection hotspot prediction (LightGBM model, test set).', y=0.018)
save_transparent_figure(fig, 'Figure_5_shap_importance.png')
plt.close()

# SHAP beeswarm (top 10)
fig = plt.figure(figsize=(7.6, 5.8))
top10_idx = [features.index(f) for f in shap_imp['Feature'].head(10)]
shap_top = shap_vals[:, top10_idx]
X_top = X_test_arr[:, top10_idx]
top_labels = [LABEL_MAP.get(f, f) for f in shap_imp['Feature'].head(10)]
shap.summary_plot(shap_top, X_top, feature_names=top_labels, show=False, max_display=10)
plt.title('SHAP Summary — Directional Impact on Hotspot Probability', fontsize=13,
 fontweight='bold', pad=10)
plt.tight_layout()
save_transparent_figure(fig, 'Figure_5b_shap_beeswarm.png', bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 6 — ML Risk Prediction Map
# ============================================================
print('[6/8] Fig 6: Ensemble risk map...')
fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.4))
ax = axes[0]
gdf.plot(column='Ensemble_Risk_Score', cmap='RdYlGn_r', legend=True, ax=ax,
 edgecolor='white', linewidth=0.3,
 legend_kwds={'label': 'Hotspot Probability', 'shrink': 0.6})
ax.set_title('A. Ensemble-Predicted Hotspot Risk', **TITLE_KW)
add_map_essentials(ax, source_note=MAP_SOURCE_NOTE)

ax = axes[1]
# Top 20 high-risk districts
top20 = gdf.nlargest(20, 'Ensemble_Risk_Score')[['District', 'REGION', 'Ensemble_Risk_Score']]
y_pos = np.arange(len(top20))
bars = ax.barh(y_pos, top20['Ensemble_Risk_Score'],
 color=plt.cm.Reds(top20['Ensemble_Risk_Score']),
 edgecolor='black', linewidth=0.4)
for i, (_, row) in enumerate(top20.iterrows()):
 ax.text(row['Ensemble_Risk_Score'] + 0.005,
 i,
 f' ({row["REGION"]})',
 va='center', fontsize=9, color='#555')
ax.set_yticks(y_pos)
ax.set_yticklabels(top20['District'], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('Ensemble Hotspot Probability', fontsize=12, fontweight='semibold')
ax.set_title('B. Top 20 High-Risk Districts', **TITLE_KW)
ax.set_xlim(0, 1.05)

plt.tight_layout(rect=[0, 0.04, 1, 1])
add_caption(fig, 'Figure 6. Machine-learning-derived HIV-TB co-infection hotspot risk. A: ensemble-averaged district-level risk map. B: top-20 predicted high-risk districts.')
save_transparent_figure(fig, 'Figure_6_ml_risk_map.png')
plt.close()

# ============================================================
# FIGURE 7 — Correlation Matrix (FULL, no masking)
# ============================================================
print('[7/8] Fig 7: Correlation matrix...')
corr_vars = [
 'HIV_Prev_Total_pct', 'TB_Incidence_per100k', 'TB_HIV_CoInfection_pct',
 'ART_Coverage_pct', 'VCT_Uptake_pct',
 'Condom_Use_pct', 'Ever_Tested_HIV_pct', 'HIV_Awareness_Women_pct',
 'Poverty_Incidence_pct', 'Poverty_Intensity_pct',
 'Unemployment_Rate_pct', 'Illiteracy_Rate_pct', 'Uninsurance_Rate_pct',
 'Doctors_per10k', 'Nurses_per10k', 'OOP_Expenditure_pct',
 'Youth_Dependency_Ratio', 'Sexually_Active_Pop_pct',
 'TB_Treatment_Success_pct',
]
corr = gdf[corr_vars].corr()
short_labels = [LABEL_MAP.get(v, v) for v in corr_vars]
corr.columns = short_labels
corr.index = short_labels

fig, ax = plt.subplots(figsize=(8.0, 7.4))
cmap = LinearSegmentedColormap.from_list('rdbu', ['#2166ac', '#f7f7f7', '#b2182b'])
sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0,
 vmin=-1, vmax=1, linewidths=0.4, cbar_kws={'shrink': 0.7},
 annot_kws={'size': 8.5}, square=True, ax=ax)
ax.set_title(f'Pearson Correlation Matrix ({len(gdf)} Districts, All {len(corr_vars)} Key Variables)',
 **TITLE_KW)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout(rect=[0, 0.04, 1, 1])
add_caption(fig, f'Figure 7. Pearson correlation matrix of {len(corr_vars)} HIV, TB, healthcare-access, and socioeconomic determinants across {len(gdf)} Ghanaian districts.')
save_transparent_figure(fig, 'Figure_7_correlation.png')
plt.close()

# ============================================================
# FIGURE 8 — GWR Local Coefficient Maps
# ============================================================
print('[8/8] Fig 8: GWR coefficients...')
gwr_vars = [c for c in gdf.columns if c.startswith('GWR_coef_')]
if len(gwr_vars) >= 4:
 fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.8))
 for i, (ax, var) in enumerate(zip(axes.flat, gwr_vars[:4])):
  label = var.replace('GWR_coef_', '')
  label_pretty = LABEL_MAP.get(label, label)
  gdf.plot(column=var, cmap='RdBu_r', legend=True, ax=ax,
  edgecolor='white', linewidth=0.2,
  legend_kwds={'shrink': 0.6})
  ax.set_title(f'GWR β: {label_pretty}', **TITLE_KW)
  add_map_essentials(ax, source_note=MAP_SOURCE_NOTE if i == 0 else None)
 plt.tight_layout(rect=[0, 0.04, 1, 1])
 add_caption(fig, 'Figure 8. GWR local coefficients showing spatial non-stationarity in predictor effects on TB-HIV co-infection prevalence.')
 save_transparent_figure(fig, 'Figure_8_gwr_coefficients.png')
 plt.close()

print('\nAll 8 figures saved at 300 DPI with transparent canvases.')
print(f' Location: {FIG}')
for f in sorted(FIG.glob('*.png')):
 print(f' {f.name}')
