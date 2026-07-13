"""
ML Pipeline — HIV-TB Co-infection Risk Prediction
===================================================
Models: XGBoost, Random Forest, LightGBM, Stacked Ensemble
Resampling: SMOTE for class imbalance
Validation: 10-fold stratified CV + spatial CV
Interpretability: SHAP (summary, waterfall, dependence)
"""
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
 recall_score, f1_score, brier_score_loss,
 classification_report, confusion_matrix, roc_curve)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import joblib
import json

np.random.seed(42)

OUT = Path(__file__).resolve().parent.parent / 'outputs'
TAB = OUT / 'tables'
MODEL_DIR = OUT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print('='*70)
print('ML PIPELINE — HIV-TB Co-infection Hotspot Risk Prediction')
print('='*70)

gdf = gpd.read_file(OUT / 'data' / 'ghana_260_districts_spatial_results.geojson')
print(f'\nData: {len(gdf)} districts')

# ============================================================
# 1. FEATURES
# ============================================================
features = [
 'HIV_Prev_Total_pct', 'HIV_Prev_Women_pct', 'HIV_Prev_Men_pct',
 'HIV_Awareness_Women_pct', 'Condom_Use_pct', 'High_Risk_Sex_pct',
 'Ever_Tested_HIV_pct', 'Know_Where_Test_pct', 'Accepting_Attitudes_pct',
 'Poverty_Incidence_pct', 'Poverty_Intensity_pct',
 'Unemployment_Rate_pct', 'Illiteracy_Rate_pct', 'Uninsurance_Rate_pct',
 'Youth_Dependency_Ratio', 'Sex_Ratio_15_64', 'Sexually_Active_Pop_pct',
 'TB_Incidence_per100k', 'ART_Coverage_pct', 'VCT_Uptake_pct',
 'Doctors_per10k', 'Nurses_per10k', 'OOP_Expenditure_pct',
 'TB_Treatment_Success_pct',
]
target = 'HIV_TB_Hotspot'

X = gdf[features].fillna(gdf[features].median())
y = gdf[target].astype(int)

print(f'\nFeatures: {len(features)}')
print(f'Target distribution: {y.value_counts().to_dict()}')
print(f'Positive class prevalence: {y.mean()*100:.1f}%')

# ============================================================
# 2. TRAIN/TEST SPLIT
# ============================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(
 X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# Apply SMOTE to training set only
sm = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum() - 1))
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f'\nPost-SMOTE training: {y_train_sm.value_counts().to_dict()}')

# ============================================================
# 3. MODELS
# ============================================================
models = {
 'Random Forest': RandomForestClassifier(
 n_estimators=500, max_depth=10, min_samples_split=5,
 class_weight='balanced', random_state=42, n_jobs=-1),
 'XGBoost': xgb.XGBClassifier(
 n_estimators=400, max_depth=5, learning_rate=0.05,
 subsample=0.85, colsample_bytree=0.85,
 scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
 eval_metric='logloss', use_label_encoder=False, random_state=42,
 n_jobs=-1),
 'LightGBM': lgb.LGBMClassifier(
 n_estimators=400, max_depth=5, learning_rate=0.05,
 num_leaves=31, class_weight='balanced',
 random_state=42, n_jobs=-1, verbose=-1),
 'Logistic Regression': LogisticRegression(
 C=1.0, class_weight='balanced', max_iter=2000, random_state=42),
}

# Stacked ensemble
base_estimators = [
 ('rf', RandomForestClassifier(n_estimators=300, max_depth=8,
 class_weight='balanced', random_state=42, n_jobs=-1)),
 ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
 scale_pos_weight=3, use_label_encoder=False,
 eval_metric='logloss', random_state=42, n_jobs=-1)),
 ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
 class_weight='balanced', random_state=42,
 n_jobs=-1, verbose=-1)),
]
models['Stacked Ensemble'] = StackingClassifier(
 estimators=base_estimators,
 final_estimator=LogisticRegression(C=0.5, max_iter=2000),
 cv=StratifiedKFold(5, shuffle=True, random_state=42),
 n_jobs=-1,
)

# ============================================================
# 4. 10-FOLD CV
# ============================================================
print('\n[1/4] 10-fold stratified CV...')
cv_results = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
 # Skip heavy stacked for CV (only test on held-out set)
 if name == 'Stacked Ensemble':
  continue
 auc_scores = cross_val_score(model, X_train_sm, y_train_sm,
 cv=skf, scoring='roc_auc', n_jobs=1)
 f1_scores = cross_val_score(model, X_train_sm, y_train_sm,
 cv=skf, scoring='f1', n_jobs=1)
 acc_scores = cross_val_score(model, X_train_sm, y_train_sm,
 cv=skf, scoring='accuracy', n_jobs=1)
 cv_results.append({
 'Model': name,
 'AUC_mean': round(auc_scores.mean(), 4),
 'AUC_SD': round(auc_scores.std(), 4),
 'F1_mean': round(f1_scores.mean(), 4),
 'F1_SD': round(f1_scores.std(), 4),
 'Accuracy_mean': round(acc_scores.mean(), 4),
 })
 print(f' {name}: AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}, '
 f'F1={f1_scores.mean():.3f}')

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(TAB / 'ml_10fold_cv_results.csv', index=False)

# ============================================================
# 5. SPATIAL CV (leave-one-region-out)
# ============================================================
print('\n[2/4] Spatial CV (leave-one-region-out)...')
regions = gdf['REGION'].values
unique_regions = np.unique(regions)
spatial_cv_auc = {name: [] for name in models.keys()}

from sklearn.base import clone
for region in unique_regions:
 mask_train = regions != region
 mask_test = regions == region
 if mask_train.sum() < 10 or mask_test.sum() < 3:
  continue
 if y.values[mask_test].sum() == 0 or y.values[mask_train].sum() == 0:
  continue
 Xtr, Xte = X_scaled.values[mask_train], X_scaled.values[mask_test]
 ytr, yte = y.values[mask_train], y.values[mask_test]
 if len(np.unique(yte)) < 2:
  continue
 # Skip SMOTE in spatial CV to speed up
 for name, model in models.items():
  if name == 'Stacked Ensemble':
   continue # too slow for LOROC
  try:
   m = clone(model)
   m.fit(Xtr, ytr)
   auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
   spatial_cv_auc[name].append(auc)
  except Exception:
   continue

spatial_cv = pd.DataFrame({
 'Model': list(spatial_cv_auc.keys()),
 'Spatial_CV_AUC_mean': [np.mean(v) if v else np.nan for v in spatial_cv_auc.values()],
 'Spatial_CV_AUC_SD': [np.std(v) if v else np.nan for v in spatial_cv_auc.values()],
 'N_folds': [len(v) for v in spatial_cv_auc.values()],
})
spatial_cv.to_csv(TAB / 'ml_spatial_cv_results.csv', index=False)
print(spatial_cv.to_string(index=False))

# ============================================================
# 6. FIT on full training, evaluate on test
# ============================================================
print('\n[3/4] Test-set evaluation...')
final_results = []
predictions = {}

for name, model in models.items():
 model.fit(X_train_sm, y_train_sm)
 y_pred = model.predict(X_test)
 y_prob = model.predict_proba(X_test)[:, 1]
 predictions[name] = {'true': y_test.values.tolist(), 'pred': y_pred.tolist(),
 'prob': y_prob.tolist()}
 final_results.append({
 'Model': name,
 'AUC': round(roc_auc_score(y_test, y_prob), 4),
 'Accuracy': round(accuracy_score(y_test, y_pred), 4),
 'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
 'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
 'F1': round(f1_score(y_test, y_pred, zero_division=0), 4),
 'Brier': round(brier_score_loss(y_test, y_prob), 4),
 })
 print(f' {name}: AUC={final_results[-1]["AUC"]}, F1={final_results[-1]["F1"]}, '
 f'Brier={final_results[-1]["Brier"]}')

final_df = pd.DataFrame(final_results)
final_df.to_csv(TAB / 'ml_test_set_performance.csv', index=False)

# Select best model for SHAP
best_model_name = final_df.loc[final_df['AUC'].idxmax(), 'Model']
print(f'\nBest model by AUC: {best_model_name}')

# ============================================================
# 7. SHAP — use LightGBM for TreeExplainer compatibility
# ============================================================
print('\n[4/4] SHAP interpretability (LightGBM)...')
xgb_model = models['XGBoost'] # keep for saving/predictions
shap_model = models['LightGBM'] # use for SHAP

explainer = shap.TreeExplainer(shap_model)
shap_values_raw = explainer.shap_values(X_test)
# LightGBM returns list for binary classification (2-class)
if isinstance(shap_values_raw, list):
 shap_values = shap_values_raw[1]
else:
 shap_values = shap_values_raw

# Save SHAP values
np.save(MODEL_DIR / 'shap_values.npy', shap_values)
np.save(MODEL_DIR / 'shap_X_test.npy', X_test.values)

# Feature importance (mean |SHAP|)
shap_importance = pd.DataFrame({
 'Feature': features,
 'Mean_abs_SHAP': np.abs(shap_values).mean(axis=0),
}).sort_values('Mean_abs_SHAP', ascending=False)
shap_importance.to_csv(TAB / 'shap_feature_importance.csv', index=False)
print('Top 10 SHAP features:')
print(shap_importance.head(10).to_string(index=False))

# Save models
joblib.dump(xgb_model, MODEL_DIR / 'xgb_model.pkl')
joblib.dump(models['Random Forest'], MODEL_DIR / 'rf_model.pkl')
joblib.dump(models['LightGBM'], MODEL_DIR / 'lgb_model.pkl')
joblib.dump(models['Stacked Ensemble'], MODEL_DIR / 'stacked_model.pkl')
joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')

# District-level predictions (all 260)
X_all = pd.DataFrame(scaler.transform(gdf[features].fillna(gdf[features].median())),
 columns=features)
gdf['RF_Risk'] = models['Random Forest'].predict_proba(X_all)[:, 1]
gdf['XGB_Risk'] = xgb_model.predict_proba(X_all)[:, 1]
gdf['LGB_Risk'] = models['LightGBM'].predict_proba(X_all)[:, 1]
gdf['Stacked_Risk'] = models['Stacked Ensemble'].predict_proba(X_all)[:, 1]
# Ensemble-average risk
gdf['Ensemble_Risk_Score'] = (gdf['RF_Risk'] + gdf['XGB_Risk'] +
 gdf['LGB_Risk'] + gdf['Stacked_Risk']) / 4

# Save enriched gdf (with ML predictions)
gdf.to_file(OUT / 'data' / 'ghana_261_final_results.geojson', driver='GeoJSON')
gdf.drop(columns='geometry').to_csv(OUT / 'data' / 'ghana_261_final_results.csv', index=False)

# Save predictions JSON for reproducibility
with open(MODEL_DIR / 'predictions.json', 'w') as f:
 json.dump(predictions, f, indent=2)

print('\nML pipeline complete.')
