"""Reviewer-facing sensitivity checks for the E&I submission package."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


OUT = "outputs/tables/submission_sensitivity_checks.csv"
np.random.seed(42)

df = pd.read_csv("outputs/data/ghana_261_final_results.csv")
features = [
    "Poverty_Incidence_pct",
    "Poverty_Intensity_pct",
    "Unemployment_Rate_pct",
    "Illiteracy_Rate_pct",
    "Uninsurance_Rate_pct",
    "Youth_Dependency_Ratio",
    "Sex_Ratio_15_64",
    "Sexually_Active_Pop_pct",
    "TB_Incidence_per100k",
    "ART_Coverage_pct",
    "Doctors_per10k",
    "Nurses_per10k",
    "OOP_Expenditure_pct",
    "TB_Treatment_Success_pct",
]
features = [col for col in features if col in df.columns]
X = pd.DataFrame(
    StandardScaler().fit_transform(df[features].fillna(df[features].median())),
    columns=features,
)
y = df["HIV_TB_Hotspot"].astype(int).to_numpy()
regions = df["REGION"].to_numpy()

models = {
    "Logistic Regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}

rows = []
for name, model in models.items():
    aucs = []
    for region in np.unique(regions):
        train = regions != region
        test = regions == region
        if test.sum() < 3 or len(np.unique(y[test])) < 2 or y[train].sum() == 0:
            continue
        fitted = clone(model)
        fitted.fit(X.loc[train], y[train])
        aucs.append(roc_auc_score(y[test], fitted.predict_proba(X.loc[test])[:, 1]))
    rows.append(
        {
            "Sensitivity": "No DHS behavioural/HIV predictors",
            "Model": name,
            "Spatial_CV_AUC_mean": round(float(np.mean(aucs)), 3),
            "Spatial_CV_AUC_SD": round(float(np.std(aucs)), 3),
            "N_folds": len(aucs),
            "Interpretation": "Structural/TB/system-only signal; weaker than full model but useful as a robustness check.",
        }
    )

guan_pair = df[df["DISTRICT"].str.upper().isin(["GUAN", "KRACHI EAST MUNICIPAL"])]
rows.append(
    {
        "Sensitivity": "Guan/Krachi East shared-polygon check",
        "Model": "Spatial cluster status",
        "Spatial_CV_AUC_mean": "",
        "Spatial_CV_AUC_SD": "",
        "N_folds": "",
        "Interpretation": "; ".join(
            f"{r.DISTRICT}: LISA={r.LISA_cluster}, BvLISA={r.BvLISA_cluster}, Gi={r.Gi_cluster}"
            for r in guan_pair.itertuples()
        ),
    }
)

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"Wrote {OUT}")
