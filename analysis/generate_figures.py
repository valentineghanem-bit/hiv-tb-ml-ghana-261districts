"""
Publication Figures - HIV-TB Co-infection Ghana 261 Districts
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
 'font.family': 'Arial',
 'font.size': 9,
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
TITLE_KW = dict(fontsize=11, fontweight='bold', color='#000000', pad=9)
CAP_KW = dict(ha='center', fontsize=9, style='italic', color='#333333')

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
  font_properties={'size': 7.5}, scale_loc='top', border_pad=0.5, pad=0.4,
 ))

 # North arrow -- upper right, inside the frame.
 x, y0, y1 = 0.94, 0.80, 0.92
 ax.annotate('', xy=(x, y1), xytext=(x, y0), xycoords=ax.transAxes,
  arrowprops=dict(arrowstyle='-|>', color='#1a1a1a', linewidth=1.6,
  mutation_scale=16))
 ax.text(x, y1 + 0.015, 'N', transform=ax.transAxes, ha='center', va='bottom',
  fontsize=11, fontweight='bold', color='#1a1a1a')

 # Optional small source/CRS text box. Placed just below the neatline frame
 # (outside the axes, in the figure margin) rather than inside the map extent:
 # Ghana's outline reaches every corner of the plotted bounding box depending on
 # which districts are coloured for a given variable, so an in-map corner is not
 # a reliably empty position and previously obscured real choropleth data.
 if source_note:
  ax.text(0.0, -0.045, source_note, transform=ax.transAxes, ha='left', va='top',
  fontsize=7.5, color='#333333', style='italic', clip_on=False,
  bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#999999',
  linewidth=0.5, alpha=1.0))


MAP_SOURCE_NOTE = 'Ghana LGA post-2018\n261 districts\nCRS: UTM Zone 30N'


LABEL_MAP = {
 'HIV_Prev_Total_pct': 'HIV prevalence (total)',
 'HIV_Prev_Women_pct': 'HIV prevalence (women)',
 'HIV_Prev_Men_pct': 'HIV prevalence (men)',
 'HIV_Awareness_Women_pct': 'HIV awareness',
 'Condom_Use_pct': 'Condom use',
 'High_Risk_Sex_pct': 'High-risk sex',
 'Ever_Tested_HIV_pct': 'Ever tested for HIV',
 'Know_Where_Test_pct': 'Knows where to test',
 'Accepting_Attitudes_pct': 'Accepting attitudes',
 'Poverty_Incidence_pct': 'Poverty incidence',
 'Poverty_Intensity_pct': 'Poverty intensity',
 'Unemployment_Rate_pct': 'Unemployment rate',
 'Illiteracy_Rate_pct': 'Illiteracy rate',
 'Uninsurance_Rate_pct': 'Uninsurance rate',
 'Youth_Dependency_Ratio': 'Youth dependency',
 'Sex_Ratio_15_64': 'Sex ratio (15-64)',
 'Sexually_Active_Pop_pct': 'Sexually active population',
 'TB_Incidence_per100k': 'TB incidence (/100k)',
 'TB_HIV_CoInfection_pct': 'TB-HIV co-infection',
 'ART_Coverage_pct': 'ART coverage',
 'VCT_Uptake_pct': 'VCT uptake',
 'Doctors_per10k': 'Doctors / 10,000',
 'Nurses_per10k': 'Nurses / 10,000',
 'OOP_Expenditure_pct': 'OOP expenditure',
 'TB_Treatment_Success_pct': 'TB treatment success',
}


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
# FIGURE 1 - Study Area & Disease Burden Choropleths (2x2)
# ============================================================
print('[1/8] Fig 1: Disease burden choropleths...')
fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.0))
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
# Caption intentionally not baked into the image: the manuscript build script
# (build_manuscript.js) supplies the numbered caption as editable docx text,
# per journal convention. Baking a second, differently-worded caption into the
# pixels produced duplicate/divergent captions in the submitted document.
save_transparent_figure(fig, 'Figure_1_disease_burden.png')
plt.close()

# ============================================================
# FIGURE 2 - LISA & Getis-Ord Hotspot Maps
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
ax.set_title('A. LISA cluster: TB-HIV co-infection', **TITLE_KW)
# geopandas polygon plots build a PatchCollection, which matplotlib's ax.legend()
# cannot auto-derive handles for (silently produces an empty legend box) -- use
# explicit proxy Patch handles instead.
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
 fontsize=7.1, frameon=False, ncol=2, columnspacing=0.7,
 labelspacing=0.25, handlelength=1.2)
add_map_essentials(ax)

# Panel B: Bivariate LISA (HIV x TB)
ax = axes[0, 1]
legend_patches = []
for cluster, color in lisa_cmap.items():
 sub = gdf[gdf['BvLISA_cluster'] == cluster]
 if len(sub) > 0:
  sub.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2)
  legend_patches.append(mpatches.Patch(facecolor=color, edgecolor='#888888',
  label=f'{cluster} (n={len(sub)})'))
ax.set_title('B. Bivariate LISA: HIV x TB', **TITLE_KW)
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
 fontsize=7.1, frameon=False, ncol=2, columnspacing=0.7,
 labelspacing=0.25, handlelength=1.2)
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
ax.set_title('C. Getis-Ord Gi* hotspots', **TITLE_KW)
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
 fontsize=7.1, frameon=False, ncol=1, labelspacing=0.25, handlelength=1.2)
add_map_essentials(ax)

# Panel D: GWR local R^2
ax = axes[1, 1]
if 'GWR_local_R2' in gdf.columns:
 gdf.plot(column='GWR_local_R2', cmap='viridis', legend=True, ax=ax,
 edgecolor='white', linewidth=0.2, legend_kwds={'shrink': 0.6})
 ax.set_title('D. GWR local R^2', **TITLE_KW)
 add_map_essentials(ax)
else:
 ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 1], h_pad=3.0, w_pad=2.0)
save_transparent_figure(fig, 'Figure_2_spatial_clusters.png')
plt.close()

# ============================================================
# FIGURE 3 - Global Moran's I bar plot with significance
# ============================================================
print('[3/8] Fig 3: Global Moran\'s I...')
moran_df = pd.read_csv(OUT / 'tables' / 'global_morans_I.csv')
moran_df['Label'] = moran_df['Variable'].map(lambda v: LABEL_MAP.get(v, v.replace('_', ' ')))
moran_df = moran_df.sort_values("Moran's I", ascending=True)
fig, ax = plt.subplots(figsize=(7.8, 5.6))
colors = ['#d7191c' if p < 0.001 else ('#fdae61' if p < 0.05 else '#bdbdbd')
 for p in moran_df['p-value']]
bars = ax.barh(moran_df['Label'], moran_df["Moran's I"], color=colors,
 edgecolor='black', linewidth=0.5)
morans_col = "Moran's I"
for bar, (_, row) in zip(bars, moran_df.iterrows()):
 i_val = row[morans_col]
 p_val = row['p-value']
 ax.text(i_val + 0.012, bar.get_y() + bar.get_height()/2,
 'I={:.3f} (p={:.3f})'.format(i_val, p_val),
 va='center', fontsize=7.8)
ax.set_xlabel("Global Moran's I", fontsize=10, fontweight='semibold')
ax.set_title("All tested indicators show positive spatial clustering", **TITLE_KW)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlim(-0.1, 1.0)
ax.tick_params(axis='y', labelsize=8.4)
ax.grid(axis='x', color='#e6e6e6', linewidth=0.6)
plt.tight_layout(rect=[0, 0.06, 1, 1])
save_transparent_figure(fig, 'Figure_3_morans_I.png')
plt.close()

# ============================================================
# FIGURE 4 - ML Model Comparison (ROC + Performance bar)
# ============================================================
print('[4/8] Fig 4: ML comparison...')
perf = pd.read_csv(OUT / 'tables' / 'ml_test_set_performance.csv')
cv = pd.read_csv(OUT / 'tables' / 'ml_10fold_cv_results.csv')
spatial_cv = pd.read_csv(OUT / 'tables' / 'ml_spatial_cv_results.csv')

fig, ax = plt.subplots(figsize=(7.8, 4.8))
sp = spatial_cv.dropna(subset=['Spatial_CV_AUC_mean']).copy()
plot_df = cv.merge(sp, on='Model', how='inner')
plot_df = plot_df.sort_values('Spatial_CV_AUC_mean', ascending=True).reset_index(drop=True)
y = np.arange(len(plot_df))
for i, row in plot_df.iterrows():
 ax.plot([row['Spatial_CV_AUC_mean'], row['AUC_mean']], [i, i],
  color='#bdbdbd', linewidth=1.5, zorder=1)
 ax.errorbar(row['Spatial_CV_AUC_mean'], i, xerr=row['Spatial_CV_AUC_SD'],
  fmt='o', color='#d55e00', ecolor='#d55e00', elinewidth=1.1, capsize=3,
  markersize=5.5, label='Spatial CV' if i == 0 else '', zorder=3)
 ax.plot(row['AUC_mean'], i, 'o', color='#0072b2', markersize=5.5,
  label='Random 10-fold CV' if i == 0 else '', zorder=4)
 ax.text(row['AUC_mean'] + 0.006, i + 0.08, f"{row['AUC_mean']:.3f}",
  fontsize=7.8, color='#0072b2')
 spatial_label_y = i + 0.18 if i < len(plot_df) - 1 else i - 0.18
 ax.text(row['Spatial_CV_AUC_mean'], spatial_label_y,
  f"{row['Spatial_CV_AUC_mean']:.3f}", fontsize=7.8, color='#8c3b00',
  ha='center')
ax.axvline(0.5, color='#666666', linestyle=':', linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(plot_df['Model'], fontsize=8.8)
ax.set_xlim(0.45, 1.04)
ax.set_ylim(-0.35, len(plot_df) - 0.55)
ax.set_xlabel('AUC-ROC (mean; spatial CV shown with +/- SD)', fontsize=10, fontweight='semibold')
ax.set_title('Spatial validation exposes random-fold optimism', **TITLE_KW)
# 'lower right' previously sat directly on top of the bottom row's data point and
# value label (Random Forest, the lowest spatial-CV model after ascending sort).
# 'upper left' falls in the low-AUC region near the y=0.5 reference line, which is
# empty of data points/labels for every model at every row.
ax.legend(loc='upper left', frameon=False, fontsize=8.4)
ax.grid(axis='x', color='#e6e6e6', linewidth=0.6)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
save_transparent_figure(fig, 'Figure_4_ml_performance.png')
plt.close()

# ============================================================
# FIGURE 5 - SHAP Summary Plot
# ============================================================
print('[5/8] Fig 5: SHAP importance + summary...')
shap_vals = np.load(OUT / 'models' / 'shap_values.npy')
X_test_arr = np.load(OUT / 'models' / 'shap_X_test.npy')
shap_imp = pd.read_csv(OUT / 'tables' / 'shap_feature_importance.csv')
features = shap_imp['Feature'].tolist()

# Clean labels
LABEL_MAP.update({
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
})

# SHAP importance lollipop (top 12): clearer in print than dense bars.
fig, ax = plt.subplots(figsize=(7.4, 5.6))
top12 = shap_imp.head(12).iloc[::-1].reset_index(drop=True)
y = np.arange(len(top12))
vals = top12['Mean_abs_SHAP'].to_numpy()
labels = [LABEL_MAP.get(f, f) for f in top12['Feature']]
ax.hlines(y, 0, vals, color='#c7dcef', linewidth=2.2)
ax.scatter(vals, y, s=42, color='#2166ac', edgecolor='black', linewidth=0.35, zorder=3)
for yi, val in zip(y, vals):
 ax.text(val + 0.045, yi, f'{val:.2f}', va='center', fontsize=8.2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.6)
ax.set_xlabel('Mean |SHAP value|', fontsize=10, fontweight='semibold', labelpad=8)
ax.set_title('HIV prevalence and VCT dominate LightGBM predictions', **TITLE_KW)
ax.set_xlim(0, max(vals) * 1.18)
ax.grid(axis='x', color='#e6e6e6', linewidth=0.6)
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
save_transparent_figure(fig, 'Figure_5_shap_importance.png')
plt.close()

# SHAP beeswarm (top 10)
fig = plt.figure(figsize=(7.6, 5.8))
top10_idx = [features.index(f) for f in shap_imp['Feature'].head(10)]
shap_top = shap_vals[:, top10_idx]
X_top = X_test_arr[:, top10_idx]
top_labels = [LABEL_MAP.get(f, f) for f in shap_imp['Feature'].head(10)]
shap.summary_plot(shap_top, X_top, feature_names=top_labels, show=False, max_display=10)
plt.title('SHAP Summary - Directional Impact on Hotspot Probability', fontsize=11,
 fontweight='bold', pad=10)
plt.tight_layout()
save_transparent_figure(fig, 'Figure_5b_shap_beeswarm.png', bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 6 - ML Risk Prediction Map
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
# Top 20 high-risk districts. Scores are tightly clustered, so a lollipop plot
# communicates rank better than a 0-based bar chart.
top20 = gdf.nlargest(20, 'Ensemble_Risk_Score')[['District', 'REGION', 'Ensemble_Risk_Score']]
y_pos = np.arange(len(top20))
region_colors = {
 'EASTERN': '#6c3483',
 'WESTERN': '#117a65',
 'WESTERN NORTH': '#1a5276',
 'UPPER WEST': '#b9770e',
}
x_min = max(0, top20['Ensemble_Risk_Score'].min() - 0.006)
for i, (_, row) in enumerate(top20.iterrows()):
 color = region_colors.get(str(row['REGION']).upper(), '#555555')
 ax.hlines(i, x_min, row['Ensemble_Risk_Score'], color='#bdbdbd', linewidth=1.1)
 ax.plot(row['Ensemble_Risk_Score'], i, 'o', color=color, markersize=5.5,
 markeredgecolor='black', markeredgewidth=0.35)
for i, (_, row) in enumerate(top20.iterrows()):
 ax.text(row['Ensemble_Risk_Score'] + 0.005,
 i,
 f" {row['Ensemble_Risk_Score']:.3f} ({row['REGION']})",
 va='center', fontsize=7.7, color='#333333')
ax.set_yticks(y_pos)
ax.set_yticklabels(top20['District'], fontsize=8.2)
ax.invert_yaxis()
ax.set_xlabel('Ensemble Hotspot Probability', fontsize=10, fontweight='semibold')
ax.set_title('B. Top 20 High-Risk Districts', **TITLE_KW)
ax.set_xlim(x_min, min(1.01, top20['Ensemble_Risk_Score'].max() + 0.045))
ax.grid(axis='x', color='#d9d9d9', linewidth=0.6)

plt.tight_layout(rect=[0, 0.04, 1, 1])
save_transparent_figure(fig, 'Figure_6_ml_risk_map.png')
plt.close()

# ============================================================
# FIGURE 7 - Correlation Matrix (FULL, no masking)
# ============================================================
print('[7/8] Fig 7: Correlation matrix...')
corr_vars = [
 'HIV_Prev_Total_pct', 'TB_Incidence_per100k', 'TB_HIV_CoInfection_pct',
 'VCT_Uptake_pct', 'ART_Coverage_pct',
 'Poverty_Incidence_pct', 'Poverty_Intensity_pct',
 'Illiteracy_Rate_pct', 'Uninsurance_Rate_pct',
 'Doctors_per10k', 'Nurses_per10k', 'OOP_Expenditure_pct',
]
corr = gdf[corr_vars].corr()
short_labels = [LABEL_MAP.get(v, v) for v in corr_vars]
corr.columns = short_labels
corr.index = short_labels

fig, ax = plt.subplots(figsize=(8.2, 7.4))
cmap = LinearSegmentedColormap.from_list('rdbu', ['#2166ac', '#f7f7f7', '#b2182b'])
sns.heatmap(corr, annot=False, cmap=cmap, center=0, vmin=-1, vmax=1,
 linewidths=0.35, linecolor='white', square=True,
 cbar_kws={'shrink': 0.72, 'label': 'Pearson r', 'pad': 0.025}, ax=ax)
for row_i, row_name in enumerate(corr.index):
 for col_i, col_name in enumerate(corr.columns):
  val = corr.loc[row_name, col_name]
  text_color = 'white' if abs(val) >= 0.58 else '#111111'
  ax.text(col_i + 0.5, row_i + 0.5, f'{val:.2f}',
   ha='center', va='center', fontsize=5.9, color=text_color)
ax.set_title(f'Selected determinant correlations ({len(gdf)} districts)',
 **TITLE_KW)
ax.tick_params(axis='both', length=0)
plt.xticks(rotation=38, ha='right', fontsize=7.0)
plt.yticks(rotation=0, fontsize=7.2)
plt.tight_layout(rect=[0, 0.06, 1, 1])
save_transparent_figure(fig, 'Figure_7_correlation.png')
plt.close()

# ============================================================
# FIGURE 8 - GWR Local Coefficient Maps
# ============================================================
print('[8/8] Fig 8: GWR coefficients...')
gwr_vars = [c for c in gdf.columns if c.startswith('GWR_coef_')]
if len(gwr_vars) >= 4:
 fig, axes = plt.subplots(2, 2, figsize=(8.9, 8.6))
 for i, (ax, var) in enumerate(zip(axes.flat, gwr_vars[:4])):
  label = var.replace('GWR_coef_', '')
  label_pretty = LABEL_MAP.get(label, label)
  gdf.plot(column=var, cmap='RdBu_r', legend=True, ax=ax,
  edgecolor='white', linewidth=0.2,
  legend_kwds={'shrink': 0.50, 'fraction': 0.035, 'pad': 0.02})
  ax.set_title(f'{chr(65+i)}. GWR coefficient: {label_pretty}', fontsize=9.1,
   fontweight='bold', color='#000000', pad=7)
  add_map_essentials(ax, source_note=None)
 fig.text(0.025, 0.018, 'Ghana LGA post-2018; 261 districts; CRS: UTM Zone 30N',
  fontsize=6.7, style='italic', color='#444444')
 plt.tight_layout(rect=[0, 0.045, 1, 1], h_pad=1.7, w_pad=1.2)
 save_transparent_figure(fig, 'Figure_8_gwr_coefficients.png')
 plt.close()

print('\nAll 8 figures saved at 300 DPI with transparent canvases.')
print(f' Location: {FIG}')
for f in sorted(FIG.glob('*.png')):
 print(f' {f.name}')
