"""
Generate the HIV-TB dashboard and poster through the Bespoke HI-EI generator.

The canonical HI-EI generator lives in the parent research system. This wrapper
creates the HIV-TB source extract from the current final results table, applies
the project-specific corrected repository slug and caveat text to a temporary
copy of the generator, runs it, and copies the generated files into this repo.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = ROOT.parent / "_system" / "bespoke"
CANONICAL_GENERATOR = SYSTEM_ROOT / "bespoke_gen.js"
TEMP = Path(os.environ.get("TEMP", r"C:\Users\VGhanem\AppData\Local\Temp"))
TEMP_GENERATOR = TEMP / "bespoke_gen_hivtb_corrected.js"
TEMP_SOURCE = TEMP / "hiv-tb_regions_source.html"
TEMP_GEO = TEMP / "ghana_districts_compact.geojson"
TEMP_OUT = TEMP / "bespoke_v5" / "hiv-tb"

DASHBOARD_OUT = ROOT / "dashboard" / "HIV_TB_Ghana_Dashboard.html"
POSTER_OUT = ROOT / "poster" / "HIV_TB_Ghana_260_Districts_Poster.html"

FULL_TO_SHORT = {
    "GREATER ACCRA": "Gr.Accra",
    "ASHANTI": "Ashanti",
    "CENTRAL": "Central",
    "EASTERN": "Eastern",
    "WESTERN": "Western",
    "VOLTA": "Volta",
    "BONO": "Bono",
    "AHAFO": "Ahafo",
    "BONO EAST": "Bono E",
    "OTI": "Oti",
    "WESTERN NORTH": "W.North",
    "UPPER EAST": "Upper East",
    "UPPER WEST": "Upper West",
    "NORTHERN": "Northern",
    "SAVANNAH": "Savannah",
    "NORTHERN EAST": "N.East",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def lisa_label(mean_value: float, q25: float, q75: float, hh_count: int) -> str:
    if hh_count:
        return "HH"
    if mean_value <= q25:
        return "LL"
    if mean_value >= q75:
        return "HL"
    return "NS"


def build_source_html() -> None:
    df = pd.read_csv(ROOT / "outputs" / "data" / "ghana_261_final_results.csv")
    moran = pd.read_csv(ROOT / "outputs" / "tables" / "global_morans_I.csv")
    bv = pd.read_csv(ROOT / "outputs" / "tables" / "bivariate_morans_I.csv").iloc[0]
    mlsp = pd.read_csv(ROOT / "outputs" / "tables" / "ml_spatial_cv_results.csv")
    gwr = pd.read_csv(ROOT / "outputs" / "tables" / "gwr_global_fit.csv").iloc[0]

    region = (
        df.groupby("REGION", as_index=False)
        .agg(
            coin=("TB_HIV_CoInfection_pct", "mean"),
            poverty=("Poverty_Incidence_pct", "mean"),
            hh=("LISA_cluster", lambda s: int((s == "High-High").sum())),
        )
        .sort_values("REGION")
    )
    q25, q75 = region["coin"].quantile([0.25, 0.75])

    region_json = []
    scatter_json = []
    for row in region.itertuples(index=False):
        name = row.REGION.upper()
        lisa = lisa_label(float(row.coin), float(q25), float(q75), int(row.hh))
        region_json.append({"name": name, "coin": round(float(row.coin), 3), "lisa": lisa})
        scatter_json.append(
            {
                "x": round(float(row.poverty), 3),
                "y": round(float(row.coin), 3),
                "n": FULL_TO_SHORT.get(name, name.title()),
                "lisa": lisa,
            }
        )

    coin_moran = moran.loc[moran["Variable"] == "TB_HIV_CoInfection_pct"].iloc[0]
    lightgbm = mlsp.loc[mlsp["Model"] == "LightGBM"].iloc[0]
    kpis = [
        (fmt(float(lightgbm["Spatial_CV_AUC_mean"])), "Spatial AUC", "LightGBM LORO-CV"),
        (fmt(float(coin_moran["Moran's I"])), "Moran's I", "co-infection, p=0.001"),
        (str(int((df["LISA_cluster"] == "High-High").sum())), "LISA HH districts", "co-infection clusters"),
        (str(int((df["BvLISA_cluster"] == "High-High").sum())), "Bivariate HH", "HIV x TB co-clusters"),
        (fmt(float(gwr["Global_R2"])), "GWR global R2", f"mean local R2 {fmt(float(gwr['Mean_Local_R2']))}"),
        ("261", "Admin districts", "260 unique polygons"),
    ]

    body = ["<html><body>"]
    body.extend(json.dumps(item, separators=(",", ":")) for item in region_json)
    body.extend(json.dumps(item, separators=(",", ":")) for item in scatter_json)
    for value, label, sub in kpis:
        body.append(f'<div class="kpi-val">{value}</div><div class="kpi-lbl">{label}</div><div class="kpi-sub">{sub}</div>')
    bivariate_moran = float(bv["Bivariate Moran's I"])
    body.append(f"<!-- bivariate_moran={bivariate_moran:.3f} -->")
    body.append("</body></html>")
    TEMP_SOURCE.write_text("\n".join(body), encoding="utf-8")


def prepare_generator() -> None:
    if not CANONICAL_GENERATOR.exists():
        raise FileNotFoundError(f"Missing HI-EI generator: {CANONICAL_GENERATOR}")
    if not (TEMP / "bespoke_dash_tmpl.html").exists() or not (TEMP / "bespoke_poster_tmpl.html").exists():
        raise FileNotFoundError("Missing HI-EI templates in %TEMP%")
    if not TEMP_GEO.exists():
        shutil.copyfile(ROOT / "ghana_districts_compact.geojson", TEMP_GEO)

    src = CANONICAL_GENERATOR.read_text(encoding="utf-8")
    src = src.replace(
        "repo:'hiv-tb-ml-ghana-261districts', src:'hiv-tb-ml-ghana-261districts_dashboard.html'",
        "repo:'hiv-tb-ml-ghana-260districts', srcPath:TMP+'/hiv-tb_regions_source.html'",
    )
    src = src.replace(
        "primary:'#6c3483', thrDefault:3,",
        (
            "primary:'#6c3483', thrDefault:12,"
            " data:'Ghana Census 2021, WHO/GHO TB and HIV indicators, and DHS-derived regional behavioural inputs',"
            " methods:'Global Moran I, LISA, bivariate LISA, Getis-Ord Gi*, GWR, LightGBM and SHAP; leave-one-region-out spatial CV is the main validation result',"
            " caveatLabel:'Spatial validation and data-scale disclosure',"
            " caveat:'This dashboard shows ecological district results. DHS behavioural inputs are regional-era indicators, and Guan is retained as a 261st administrative district using the Krachi East shared-polygon convention. For judging model transfer, use the 0.798 spatial AUC rather than the 0.998 random-fold AUC.',"
        ),
    )
    src = re.sub(
        r"poster:\{ kicker:'HIV.*?takeaway:'.*?'\s*\} \},",
        (
            "poster:{ kicker:'HIV-TB co-epidemic - Ghana',"
            " hook:'HIV-TB co-infection clusters geographically. The stricter test is whether the model still works when a whole region is held out.',"
            " takeaway:'Use the co-cluster map for district prioritisation, and quote the spatial AUC when talking about generalisation.' } },"
        ),
        src,
        count=1,
        flags=re.DOTALL,
    )
    TEMP_GENERATOR.write_text(src, encoding="utf-8")


def patch_generated_html() -> None:
    for path in [TEMP_OUT / "dashboard.html", TEMP_OUT / "poster.html"]:
        text = path.read_text(encoding="utf-8")
        text = text.replace("hiv-tb-ml-ghana-261districts", "hiv-tb-ml-ghana-260districts")
        text = text.replace("HIVâ€“TB", "HIV-TB").replace("HIV–TB", "HIV-TB")
        path.write_text(text, encoding="utf-8")


def main() -> None:
    build_source_html()
    prepare_generator()
    subprocess.run(["node", str(TEMP_GENERATOR), "hiv-tb"], check=True)
    patch_generated_html()
    DASHBOARD_OUT.parent.mkdir(exist_ok=True)
    POSTER_OUT.parent.mkdir(exist_ok=True)
    shutil.copyfile(TEMP_OUT / "dashboard.html", DASHBOARD_OUT)
    shutil.copyfile(TEMP_OUT / "poster.html", POSTER_OUT)
    print(f"Dashboard: {DASHBOARD_OUT} ({DASHBOARD_OUT.stat().st_size / 1024:.1f} KB)")
    print(f"Poster: {POSTER_OUT} ({POSTER_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
