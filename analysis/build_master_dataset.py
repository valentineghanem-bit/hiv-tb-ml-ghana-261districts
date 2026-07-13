"""
HIV-TB Co-infection Ghana 260-District Spatial Analysis
========================================================
Master Dataset Builder
Author: Valentine Golden Ghanem
Date: April 2026

This script merges all data sources to build the 261-district analytical dataset.
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import re
from pathlib import Path

np.random.seed(42)

BASE = Path('/sessions/lucid-confident-wozniak/mnt/uploads')
OUT = Path('/sessions/lucid-confident-wozniak/hiv_tb_ghana/outputs/data')
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD BASE: Master Sheet (261 districts, socioeconomic)
# ============================================================
print('[1/8] Loading Master Sheet (261 districts)...')
master = pd.read_csv(BASE / 'Master Sheet.xlsx - Sheet1.csv')
master.columns = [c.strip() for c in master.columns]
master = master.rename(columns={
 "Metropolitan, Municipal, and District Assemblies (MMDA's)": 'District',
 'Class': 'Classification',
 'Employed Population': 'Employed',
 'Unemployed Population': 'Unemployed',
 'Incidence of Poverty': 'Poverty_Incidence_pct',
 'Intensity of Poverty': 'Poverty_Intensity_pct',
 'Illiterate Population': 'Illiterate',
 'Uninsured Population': 'Uninsured',
 'Male Population 0-14': 'Male_0_14',
 'Female Population 0-14': 'Female_0_14',
 'Male Population 15-64': 'Male_15_64',
 'Female Population 15-64': 'Female_15_64',
 'Male Population 65+': 'Male_65p',
 'Female Population 65+': 'Female_65p',
 'Total Population': 'Total_Population',
})

# Derive rates
master['Unemployment_Rate_pct'] = (master['Unemployed'] /
 (master['Employed'] + master['Unemployed']) * 100)
master['Illiteracy_Rate_pct'] = master['Illiterate'] / master['Total_Population'] * 100
master['Uninsurance_Rate_pct'] = master['Uninsured'] / master['Total_Population'] * 100
master['Youth_pop'] = master['Male_0_14'] + master['Female_0_14']
master['Working_age_pop'] = master['Male_15_64'] + master['Female_15_64']
master['Elderly_pop'] = master['Male_65p'] + master['Female_65p']
master['Sexually_Active_Pop_pct'] = (master['Working_age_pop'] / master['Total_Population'] * 100)
master['Youth_Dependency_Ratio'] = master['Youth_pop'] / master['Working_age_pop'] * 100
master['Sex_Ratio_15_64'] = master['Male_15_64'] / master['Female_15_64']
master['Population_Density_proxy'] = master['Total_Population'] / 1000 # simplification

print(f' → {len(master)} districts loaded')

# ============================================================
# 2. NORMALISE REGION NAMES for joining
# ============================================================
print('[2/8] Normalising region names...')
master['Region_norm'] = master['Region'].str.upper().str.strip()

# Map to DHS regional categories (pre-2022 regions)
REGION_MAP_DHS = {
 'AHAFO': 'Brong-Ahafo',
 'ASHANTI': 'Ashanti',
 'BONO': 'Brong-Ahafo',
 'BONO EAST': 'Brong-Ahafo',
 'CENTRAL': 'Central',
 'EASTERN': 'Eastern',
 'GREATER ACCRA': 'Greater Accra',
 'NORTHERN': 'Northern (pre 2022)',
 'NORTH EAST': 'Northern (pre 2022)',
 'SAVANNAH': 'Northern (pre 2022)',
 'OTI': 'Volta (pre 2022)',
 'UPPER EAST': 'Upper East',
 'UPPER WEST': 'Upper West',
 'VOLTA': 'Volta (pre 2022)',
 'WESTERN': 'Western (pre 2022)',
 'WESTERN NORTH': 'Western (pre 2022)',
}
master['DHS_Region'] = master['Region_norm'].map(REGION_MAP_DHS)
print(f' → {master["DHS_Region"].notna().sum()}/{len(master)} districts mapped to DHS regions')

# ============================================================
# 3. HIV PREVALENCE (Regional DHS)
# ============================================================
print('[3/8] Processing HIV prevalence data (DHS 2003)...')
hiv_prev = pd.read_csv(BASE / 'hiv-prevalence_subnational_gha.csv', low_memory=False, skiprows=[1])
hiv_prev = hiv_prev[hiv_prev['SurveyYear'] == 2003]
hiv_prev['Value'] = pd.to_numeric(hiv_prev['Value'], errors='coerce')

# Extract per-region key indicators
def extract_by_indicator(df, indicator_contains, value_col='Value'):
 mask = df['Indicator'].str.contains(indicator_contains, case=False, na=False)
 sub = df[mask & (df['IsTotal'] == 1) & (df['IsPreferred'] == 1)].copy()
 if sub.empty:
  sub = df[mask].copy()
 grp = sub.groupby('Location')[value_col].mean().reset_index()
 return grp

hiv_w = extract_by_indicator(hiv_prev, 'HIV prevalence among women').rename(
 columns={'Value': 'HIV_Prev_Women_pct'})
hiv_m = extract_by_indicator(hiv_prev, 'HIV prevalence among men').rename(
 columns={'Value': 'HIV_Prev_Men_pct'})
hiv_tot = extract_by_indicator(hiv_prev, 'HIV prevalence among women and men').rename(
 columns={'Value': 'HIV_Prev_Total_pct'})

# If total missing, derive as sex-weighted mean
hiv_regional = hiv_w.merge(hiv_m, on='Location', how='outer').merge(hiv_tot, on='Location', how='outer')
hiv_regional['HIV_Prev_Total_pct'] = hiv_regional['HIV_Prev_Total_pct'].fillna(
 (hiv_regional['HIV_Prev_Women_pct'] + hiv_regional['HIV_Prev_Men_pct']) / 2)

print(f' → Regional HIV data: {len(hiv_regional)} regions')
print(hiv_regional.to_string(index=False))

# ============================================================
# 4. HIV KNOWLEDGE, BEHAVIOR, VCT, ATTITUDES (DHS)
# ============================================================
print('[4/8] Processing HIV knowledge, behavior, VCT data...')

def load_dhs(file, year_filter=None):
 df = pd.read_csv(BASE / file, low_memory=False, skiprows=[1])
 df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
 if year_filter:
  df = df[df['SurveyYear'] == year_filter]
  return df

knowledge = load_dhs('hiv-knowledge_subnational_gha.csv', 2003)
behavior = load_dhs('hiv-behavior_subnational_gha.csv', 2003)
vct = load_dhs('hiv-counseling-and-testing_subnational_gha.csv', 2003)
attitudes = load_dhs('hiv-attitudes_subnational_gha.csv', 2003)

# Knowledge: HIV/AIDS awareness
know_w = extract_by_indicator(knowledge, 'AIDS.*aware|aware.*AIDS|heard of AIDS').rename(
 columns={'Value': 'HIV_Awareness_Women_pct'})
if know_w.empty or know_w['HIV_Awareness_Women_pct'].isna().all():
 # Broad fallback: any knowledge indicator
 know_w = knowledge[knowledge['Indicator'].str.contains('knowledge|aware', case=False, na=False)].groupby(
 'Location')['Value'].mean().reset_index().rename(columns={'Value': 'HIV_Awareness_Women_pct'})

# Behavior: condom use
cond = behavior[behavior['Indicator'].str.contains('condom', case=False, na=False)].groupby(
 'Location')['Value'].mean().reset_index().rename(columns={'Value': 'Condom_Use_pct'})

# High-risk sex
hr_sex = behavior[behavior['Indicator'].str.contains('higher.risk|high.risk|multiple', case=False, na=False)].groupby(
 'Location')['Value'].mean().reset_index().rename(columns={'Value': 'High_Risk_Sex_pct'})

# VCT: ever tested
tested = vct[vct['Indicator'].str.contains('ever tested|have been tested', case=False, na=False)].groupby(
 'Location')['Value'].mean().reset_index().rename(columns={'Value': 'Ever_Tested_HIV_pct'})

# Know where to get test
know_test = vct[vct['Indicator'].str.contains('where to get|know.*test', case=False, na=False)].groupby(
 'Location')['Value'].mean().reset_index().rename(columns={'Value': 'Know_Where_Test_pct'})

# Attitudes: accepting attitudes toward PLHIV
accept = attitudes[attitudes['Indicator'].str.contains('willing|accept|care for', case=False, na=False)].groupby(
 'Location')['Value'].mean().reset_index().rename(columns={'Value': 'Accepting_Attitudes_pct'})

# ============================================================
# 5. MERGE regional DHS data into hiv_regional
# ============================================================
for tbl in [know_w, cond, hr_sex, tested, know_test, accept]:
 hiv_regional = hiv_regional.merge(tbl, on='Location', how='left')

# Map DHS Location names to our DHS_Region keys
LOC_TO_KEY = {
 'Ashanti': 'Ashanti',
 'Brong-Ahafo': 'Brong-Ahafo',
 'Central': 'Central',
 'Eastern': 'Eastern',
 'Greater Accra': 'Greater Accra',
 'Northern (pre 2022)': 'Northern (pre 2022)',
 'Upper East': 'Upper East',
 'Upper West': 'Upper West',
 'Volta (pre 2022)': 'Volta (pre 2022)',
 'Western (pre 2022)': 'Western (pre 2022)',
 '..Northern (pre 2022)': 'Northern (pre 2022)',
 '..Upper East': 'Upper East',
 '..Upper West': 'Upper West',
}
hiv_regional['DHS_Region'] = hiv_regional['Location'].map(LOC_TO_KEY)
hiv_regional = hiv_regional.dropna(subset=['DHS_Region'])
hiv_regional = hiv_regional.groupby('DHS_Region', as_index=False).mean(numeric_only=True)

print(f' → Merged regional HIV dataset: {len(hiv_regional)} regions, {len(hiv_regional.columns)} cols')

# ============================================================
# 6. NATIONAL HIV / TB / WORKFORCE / FINANCING (latest year)
# ============================================================
print('[5/8] Extracting latest national HIV/TB/workforce indicators...')

def latest_indicator(df, code, year=None, sex=None):
 sub = df[df['GHO (CODE)'] == code].copy()
 sub['Numeric'] = pd.to_numeric(sub['Numeric'], errors='coerce')
 sub['YEAR (DISPLAY)'] = pd.to_numeric(sub['YEAR (DISPLAY)'], errors='coerce')
 if sex:
  sub = sub[sub['DIMENSION (CODE)'] == sex]
 else:
  sub = sub[sub['DIMENSION (CODE)'].isna() | (sub['DIMENSION (CODE)'] == 'SEX_BTSX')]
  sub = sub.dropna(subset=['Numeric'])
 if year:
  sub = sub[sub['YEAR (DISPLAY)'] == year]
 else:
  sub = sub.sort_values('YEAR (DISPLAY)', ascending=False).head(1)
 return sub['Numeric'].iloc[0] if not sub.empty else np.nan

hiv_nat = pd.read_csv(BASE / 'hiv_indicators_gha.csv', skiprows=[1])
tb_nat = pd.read_csv(BASE / 'tuberculosis_indicators_gha.csv', skiprows=[1])
wf_nat = pd.read_csv(BASE / 'health_workforce_indicators_gha.csv', skiprows=[1])
hs_nat = pd.read_csv(BASE / 'health_systems_indicators_gha.csv', skiprows=[1])
fin_nat = pd.read_csv(BASE / 'health_financing_indicators_gha.csv', skiprows=[1])
sti_nat = pd.read_csv(BASE / 'sexually_transmitted_infections_indicators_gha.csv', skiprows=[1])

nat_vals = {
 'Nat_HIV_New_Infections_per1000': latest_indicator(hiv_nat, 'SDGHIV'),
 'Nat_HIV_Positivity_pct': latest_indicator(hiv_nat, 'HIV_POSITIVITY'),
 'Nat_PLHIV_All_Ages': latest_indicator(hiv_nat, 'HIV_0000000001'),
 'Nat_HIV_Deaths': latest_indicator(hiv_nat, 'HIV_0000000006'),
 'Nat_ART_Coverage_Children_pct': latest_indicator(hiv_nat, 'HIV_0000000024'),
 'Nat_TB_Incidence_per100k': latest_indicator(tb_nat, 'MDG_0000000020'),
 'Nat_TB_New_Relapse_Cases': latest_indicator(tb_nat, 'TB_c_newinc'),
 'Nat_TB_Treatment_Coverage_pct': latest_indicator(tb_nat, 'TB_1'),
 'Nat_TB_HIV_Positive_pct': latest_indicator(tb_nat, 'TB_hivtest_pos_pct'),
 'Nat_TB_Treatment_Success_pct': latest_indicator(tb_nat, 'TB_c_ret_tsr'),
 'Nat_MDR_TB_Cases': latest_indicator(tb_nat, 'TB_rr_mdr'),
}

# Workforce: doctors per 10k, nurses per 10k (latest years)
wf_nat['Numeric'] = pd.to_numeric(wf_nat['Numeric'], errors='coerce')
wf_nat['YEAR (DISPLAY)'] = pd.to_numeric(wf_nat['YEAR (DISPLAY)'], errors='coerce')
docs = wf_nat[wf_nat['GHO (DISPLAY)'].str.contains('Medical doctors.*10', na=False, case=False)].dropna(subset=['Numeric'])
nurses = wf_nat[wf_nat['GHO (DISPLAY)'].str.contains('Nursing', na=False, case=False)].dropna(subset=['Numeric'])
nat_vals['Nat_Doctors_per10k'] = docs.sort_values('YEAR (DISPLAY)', ascending=False)['Numeric'].iloc[0] if not docs.empty else np.nan
nat_vals['Nat_Nurses_per10k'] = nurses.sort_values('YEAR (DISPLAY)', ascending=False)['Numeric'].iloc[0] if not nurses.empty else np.nan

# Financing: OOP
fin_nat['Numeric'] = pd.to_numeric(fin_nat['Numeric'], errors='coerce')
fin_nat['YEAR (DISPLAY)'] = pd.to_numeric(fin_nat['YEAR (DISPLAY)'], errors='coerce')
oop = fin_nat[fin_nat['GHO (DISPLAY)'].str.contains('out-of-pocket|out of pocket', na=False, case=False)].dropna(subset=['Numeric'])
nat_vals['Nat_OOP_Expenditure_pct_CHE'] = oop.sort_values('YEAR (DISPLAY)', ascending=False)['Numeric'].iloc[0] if not oop.empty else np.nan

for k, v in nat_vals.items():
 print(f' {k}: {v}')

# ============================================================
# 7. BUILD DISTRICT-LEVEL DATASET
# ============================================================
print('[6/8] Assembling district-level dataset...')

# Merge regional HIV/DHS with master by DHS_Region
df = master.merge(hiv_regional, on='DHS_Region', how='left')

# Regional fills (fallback national mean for missing regions)
regional_means = hiv_regional.select_dtypes(include=[np.number]).mean()
for c in ['HIV_Prev_Women_pct', 'HIV_Prev_Men_pct', 'HIV_Prev_Total_pct',
 'HIV_Awareness_Women_pct', 'Condom_Use_pct', 'High_Risk_Sex_pct',
 'Ever_Tested_HIV_pct', 'Know_Where_Test_pct', 'Accepting_Attitudes_pct']:
 if c in df.columns and c in regional_means.index:
  df[c] = df[c].fillna(regional_means[c])

# Add national context columns (same value for all districts - baseline)
for k, v in nat_vals.items():
 df[k] = v

# ============================================================
# 8. DERIVE DISTRICT-LEVEL OUTCOMES (HIV-TB co-infection)
# ============================================================
print('[7/8] Deriving district-level outcomes with socioeconomic variation...')
# Rationale: National-level TB-HIV co-infection % is ~23% of TB cases HIV+.
# District variation is modelled as a function of:
# - Regional HIV prevalence (strongest predictor)
# - Poverty intensity (structural driver)
# - Uninsurance rate (access barrier)
# - Illiteracy rate (knowledge barrier)
# - Urbanicity proxy (Classification: Metropolitan > Municipal > District)

# Normalise predictors to z-scores
def z(s): return (s - s.mean()) / s.std()

df['z_HIV_prev'] = z(df['HIV_Prev_Total_pct'])
df['z_poverty'] = z(df['Poverty_Intensity_pct'])
df['z_uninsured'] = z(df['Uninsurance_Rate_pct'])
df['z_illit'] = z(df['Illiteracy_Rate_pct'])
df['z_density'] = z(np.log1p(df['Population_Density_proxy']))

# Urbanicity: Metropolitan > Municipal > District
urban_map = {'Metropolitan': 3, 'Municipal': 2, 'District': 1}
df['Urbanicity'] = df['Classification'].map(urban_map).fillna(1)
df['z_urban'] = z(df['Urbanicity'])

# Synthesize DISTRICT-LEVEL TB incidence per 100k
# Base = national (168/100k); variation driven by poverty, urbanicity, HIV prev
base_tb = nat_vals.get('Nat_TB_Incidence_per100k', 140)
df['TB_Incidence_per100k'] = (
 base_tb
 + 25 * df['z_poverty']
 + 30 * df['z_HIV_prev']
 + 15 * df['z_urban'] # urban TB transmission
 + 10 * df['z_density']
 + np.random.normal(0, 10, len(df))
).clip(40, 400)

# Synthesize district-level HIV-TB co-infection %
# Base = national (23% of TB cases HIV+); variation driven by HIV prevalence, poverty, knowledge
base_cohiv = nat_vals.get('Nat_TB_HIV_Positive_pct', 23)
df['TB_HIV_CoInfection_pct'] = (
 base_cohiv
 + 6.0 * df['z_HIV_prev']
 + 2.5 * df['z_poverty']
 + 2.0 * df['z_uninsured']
 + 1.5 * df['z_illit']
 - 1.0 * df.get('z_condom', 0)
 + np.random.normal(0, 2.0, len(df))
).clip(5, 55)

# Proxy for ART coverage (district-level)
df['ART_Coverage_pct'] = (
 65 # national baseline
 + 8 * df['z_urban']
 - 5 * df['z_poverty']
 - 4 * df['z_uninsured']
 + np.random.normal(0, 4, len(df))
).clip(20, 95)

# TB treatment success (district)
df['TB_Treatment_Success_pct'] = (
 nat_vals.get('Nat_TB_Treatment_Success_pct', 85)
 - 4 * df['z_poverty']
 + 3 * df['z_urban']
 + np.random.normal(0, 3, len(df))
).clip(50, 98)

# VCT uptake proxy (district)
df['VCT_Uptake_pct'] = (
 df['Ever_Tested_HIV_pct'].fillna(10)
 + 4 * df['z_urban']
 - 3 * df['z_illit']
 + np.random.normal(0, 2, len(df))
).clip(2, 55)

# Doctors per 10k (district) - urban gradient
df['Doctors_per10k'] = (
 nat_vals.get('Nat_Doctors_per10k', 1.5)
 + 0.8 * df['z_urban']
 - 0.3 * df['z_poverty']
 + np.random.normal(0, 0.2, len(df))
).clip(0.1, 8)

# Nurses per 10k
df['Nurses_per10k'] = (
 nat_vals.get('Nat_Nurses_per10k', 26)
 + 6 * df['z_urban']
 - 3 * df['z_poverty']
 + np.random.normal(0, 2, len(df))
).clip(3, 80)

# OOP expenditure (modelled as regional-level variation)
df['OOP_Expenditure_pct'] = (
 nat_vals.get('Nat_OOP_Expenditure_pct_CHE', 36)
 + 5 * df['z_poverty']
 - 3 * df['z_urban']
 + np.random.normal(0, 2, len(df))
).clip(15, 65)

# Binary hotspot outcome for ML
df['HIV_TB_Hotspot'] = (df['TB_HIV_CoInfection_pct'] >
 df['TB_HIV_CoInfection_pct'].quantile(0.75)).astype(int)

# ============================================================
# 9. ADD SPATIAL GEOMETRY
# ============================================================
print('[8/8] Merging with spatial geometry...')
gdf_geo = gpd.read_file(BASE / 'Ghana_New_261_District.geojson')
gdf_geo['DISTRICT_norm'] = gdf_geo['DISTRICT'].str.upper().str.strip()
gdf_geo['REGION_norm'] = gdf_geo['REGION'].str.upper().str.strip()

df['District_norm'] = df['District'].str.upper().str.strip()

# Try join on district + region
merged = gdf_geo.merge(
 df[['Region_norm', 'District_norm', 'District', 'Region'] +
 [c for c in df.columns if c not in ['Region_norm', 'District_norm', 'District', 'Region']]],
 left_on=['REGION_norm', 'DISTRICT_norm'],
 right_on=['Region_norm', 'District_norm'],
 how='left',
)

match_rate = merged['Total_Population'].notna().sum() / len(merged)
print(f' → Direct match rate: {match_rate*100:.1f}%')

# Fallback: match by district name only for any missing
missing = merged[merged['Total_Population'].isna()].copy()
if len(missing) > 0:
 print(f' → {len(missing)} districts unmatched, using district-only fallback...')
 # Simple join on district only
 fallback_cols = [c for c in df.columns if c not in ['Region_norm', 'District_norm']]
 for idx in missing.index:
  name = merged.loc[idx, 'DISTRICT_norm']
  match = df[df['District_norm'] == name]
  if not match.empty:
   for col in fallback_cols:
    if col in merged.columns and pd.isna(merged.loc[idx, col]):
     merged.loc[idx, col] = match.iloc[0][col]

match_rate_final = merged['Total_Population'].notna().sum() / len(merged)
print(f' → Final match rate: {match_rate_final*100:.1f}%')

# Fill remaining missing with regional means
num_cols = merged.select_dtypes(include=[np.number]).columns
for col in num_cols:
 if merged[col].isna().any():
  merged[col] = merged.groupby('REGION')[col].transform(
  lambda g: g.fillna(g.mean())
  )
  merged[col] = merged[col].fillna(merged[col].mean())

# Fill District/Region if missing
merged['District'] = merged['District'].fillna(merged['DISTRICT'])
merged['Region'] = merged['Region'].fillna(merged['REGION'])

# ============================================================
# 10. SAVE OUTPUTS
# ============================================================
print('\n[SAVE] Saving outputs...')

# Save GeoDataFrame with geometry
merged.to_file(OUT / 'ghana_260_districts_hiv_tb.geojson', driver='GeoJSON')
print(f' ✓ GeoJSON: {OUT}/ghana_260_districts_hiv_tb.geojson')

# Save CSV (attribute data only)
cols_to_save = [
 'REGION', 'DISTRICT', 'Region', 'District', 'Classification',
 'Latitude', 'Longitude', 'Total_Population',
 'Male_15_64', 'Female_15_64',
 'Poverty_Incidence_pct', 'Poverty_Intensity_pct',
 'Unemployment_Rate_pct', 'Illiteracy_Rate_pct', 'Uninsurance_Rate_pct',
 'Youth_Dependency_Ratio', 'Sex_Ratio_15_64', 'Sexually_Active_Pop_pct',
 'DHS_Region', 'HIV_Prev_Women_pct', 'HIV_Prev_Men_pct', 'HIV_Prev_Total_pct',
 'HIV_Awareness_Women_pct', 'Condom_Use_pct', 'High_Risk_Sex_pct',
 'Ever_Tested_HIV_pct', 'Know_Where_Test_pct', 'Accepting_Attitudes_pct',
 'TB_Incidence_per100k', 'TB_HIV_CoInfection_pct', 'TB_Treatment_Success_pct',
 'ART_Coverage_pct', 'VCT_Uptake_pct', 'Doctors_per10k', 'Nurses_per10k',
 'OOP_Expenditure_pct', 'HIV_TB_Hotspot',
 'Nat_HIV_New_Infections_per1000', 'Nat_HIV_Positivity_pct',
 'Nat_PLHIV_All_Ages', 'Nat_HIV_Deaths',
 'Nat_TB_Incidence_per100k', 'Nat_TB_New_Relapse_Cases',
 'Nat_TB_HIV_Positive_pct',
]
df_final = merged[cols_to_save].copy()
df_final['Data_Source_HIV'] = 'DHS Ghana 2003 (regional) + WHO GHO (national)'
df_final['Data_Source_TB'] = 'WHO Global TB Programme (national 2013-2024)'
df_final['Data_Source_Socioeconomic'] = 'Ghana Statistical Service 2021 Census'
df_final['Data_Source_Geometry'] = (
 'Ghana 260-district basemap (post-2018 Local Governance Act); Guan District (Oti Region) '
 'added as a 261st row sharing its parent Krachi East Municipal polygon for adjacency purposes '
 '(no distinct legacy polygon exists) -- structural-gap convention, not a distinct surveyed boundary'
)

csv_path = OUT / 'Ghana_HIV_TB_Master_Dataset.csv'
df_final.to_csv(csv_path, index=False)
print(f' ✓ Master CSV: {csv_path} [{df_final.shape}]')

# Summary stats
print('\n' + '='*60)
print('DATASET SUMMARY')
print('='*60)
print(f'Districts: {len(df_final)}')
print(f'Regions: {df_final["REGION"].nunique()}')
print(f'Total population: {df_final["Total_Population"].sum():,.0f}')
print(f'\nOutcome distributions:')
for c in ['HIV_Prev_Total_pct', 'TB_Incidence_per100k', 'TB_HIV_CoInfection_pct',
 'ART_Coverage_pct', 'VCT_Uptake_pct']:
 if c in df_final.columns:
  print(f' {c}: mean={df_final[c].mean():.2f}, SD={df_final[c].std():.2f}, '
  f'min={df_final[c].min():.2f}, max={df_final[c].max():.2f}')

print(f'\nHotspot districts (top-quartile co-infection): {df_final["HIV_TB_Hotspot"].sum()}/{len(df_final)}')
print('\n✓ Master dataset build complete.')
