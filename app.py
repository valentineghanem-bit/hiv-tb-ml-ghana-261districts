"""
Machine-Learning Mapping of HIV–TB — Ghana — interactive analytics app (Streamlit + Plotly).
Run:  pip install streamlit plotly pandas  &&  streamlit run app.py
Data: published regional values for this Ghana study. Colourblind-safe palette; works offline.
"""
import json, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="HIV–TB · ML — Ghana", layout="wide", initial_sidebar_state="expanded")

# ---------------- real data ----------------
REGIONS = [
    ("GREATER ACCRA", "Gr.Accra", 8.2, 12.4, "HH"),
    ("EASTERN", "Eastern", 6.1, 29.8, "HH"),
    ("VOLTA", "Volta", 5.8, 36.2, "HH"),
    ("WESTERN", "Western", 4.2, 23.1, "NS"),
    ("CENTRAL", "Central", 4.8, 26.7, "NS"),
    ("ASHANTI", "Ashanti", 5.1, 28.3, "HL"),
    ("OTI", "Oti", 3.6, 48.4, "NS"),
    ("BONO EAST", "Bono E", 3.2, 44.2, "NS"),
    ("AHAFO", "Ahafo", 2.8, 39.1, "NS"),
    ("BONO", "Bono", 2.9, 35.8, "NS"),
    ("WESTERN NORTH", "W.North", 2.4, 42.3, "LH"),
    ("UPPER EAST", "Upper East", 1.4, 70.4, "LL"),
    ("UPPER WEST", "Upper West", 1.2, 68.9, "LL"),
    ("NORTHERN", "Northern", 2.1, 62.1, "LL"),
    ("SAVANNAH", "Savannah", 1.3, 74.3, "LL"),
    ("NORTHERN EAST", "N.East", 0.9, 76.8, "LL"),
]
df = pd.DataFrame(REGIONS, columns=["region", "short", "v", "x", "lisa"])
HASX     = True
OUTCOME  = "HIV–TB burden"
UNIT     = "%"
COV      = "Poverty index"
PRIMARY  = "#6c3483"
SCALE    = ["rgb(241,238,246)", "rgb(215,181,216)", "rgb(167,115,180)", "rgb(118,42,131)", "rgb(77,0,75)"]
KPIS     = [
    ("99.8%", "LightGBM AUC", "LOROCV validated"),
    ("0.449", "Bivariate Moran's I", "HIV × TB"),
    ("31", "HH Co-cluster Dist.", "Southern belt"),
    ("0.916", "GWR R²", "Local spatial fit"),
    ("0.44", "SEM β Poverty→HIV", "p "),
    ("261", "Districts", "All Ghana"),
]
LISA   = {"HH": "#c0392b", "LL": "#2980b9", "HL": "#e67e22", "LH": "#82c0e8", "NS": "#bdc3c7"}
LNAME  = {"HH": "High-High", "LL": "Low-Low", "HL": "High-Low", "LH": "Low-High", "NS": "Not sig."}

@st.cache_data
def load_geo():
    p = os.path.join(os.path.dirname(__file__), "ghana_districts_compact.geojson")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------- sidebar ----------------
st.sidebar.title("HIV–TB · ML — Ghana")
st.sidebar.caption("District HIV–TB co-burden, clusters & ML predictors · 261 districts")
lisa_pick   = st.sidebar.multiselect("Filter by spatial cluster (LISA)", sorted(df.lisa.unique()), default=list(df.lisa.unique()))
region_pick = st.sidebar.multiselect("Filter by region", df.region.tolist(), default=df.region.tolist())
if HASX:
    map_metric = st.sidebar.radio("Map metric", [OUTCOME, COV], index=0)
else:
    map_metric = OUTCOME
fdf = df[df.lisa.isin(lisa_pick) & df.region.isin(region_pick)]

# ---------------- header + KPIs ----------------
st.markdown("### " + "Machine-Learning Mapping of HIV–TB — Ghana")
st.caption("District HIV–TB co-burden, clusters & ML predictors · 261 districts" + f"  ·  {len(fdf)} of {len(df)} regions in view")
if KPIS:
    cols = st.columns(len(KPIS))
    for col, (kv, kl, ks) in zip(cols, KPIS):
        col.metric(kl, kv, ks if ks else None)

# ---------------- row 1: choropleth + ranking ----------------
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f"**{map_metric} by district** — districts coloured by regional value")
    geo = load_geo()
    feat = [{"name": x["properties"]["name"], "region": x["properties"]["region"]} for x in geo["features"]]
    mp = pd.DataFrame(feat).merge(df, on="region", how="left")
    col = "x" if (HASX and map_metric == COV) else "v"
    fig = px.choropleth(mp, geojson=geo, locations="name", featureidkey="properties.name",
                        color=col, color_continuous_scale=SCALE, hover_name="name",
                        hover_data={"region": True, "v": True, "lisa": True, "name": False})
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=440,
                      coloraxis_colorbar=dict(title=(COV if (HASX and map_metric == COV) else OUTCOME)[:14]))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown(f"**Regional ranking — {OUTCOME}** (colour = LISA cluster)")
    r = fdf.sort_values("v")
    fig = go.Figure(go.Bar(x=r.v, y=r.short, orientation="h",
                           marker_color=[LISA[l] for l in r.lisa],
                           text=[f"{v}{UNIT}" for v in r.v], textposition="outside"))
    fig.update_layout(height=440, margin=dict(l=4, r=14, t=4, b=4), xaxis_title=OUTCOME + " (" + UNIT + ")")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- row 2: driver/cluster + composition + parallel ----------------
c3, c4, c5 = st.columns(3)
with c3:
    if HASX:
        st.markdown(f"**{OUTCOME} vs {COV}**")
        xs = fdf.dropna(subset=["x"])
        fig = px.scatter(xs, x="x", y="v", color="lisa", color_discrete_map=LISA, text="short",
                         labels={"x": COV, "v": OUTCOME + " (" + UNIT + ")"})
        fig.update_traces(textposition="top center", marker_size=12)
        if len(xs) >= 2:
            import numpy as np
            b, a = np.polyfit(xs.x.astype(float), xs.v.astype(float), 1)
            xr = [float(xs.x.min()), float(xs.x.max())]
            fig.add_trace(go.Scatter(x=xr, y=[a + b * xr[0], a + b * xr[1]], mode="lines",
                                     line=dict(color="#888", dash="dot", width=2),
                                     showlegend=False, hoverinfo="skip"))
        fig.update_layout(height=360, margin=dict(l=4, r=4, t=4, b=4), showlegend=False)
    else:
        st.markdown("**Burden by spatial-cluster class**")
        g = fdf.groupby("lisa")["v"].mean().reindex(["HH","HL","LH","LL","NS"]).dropna()
        fig = go.Figure(go.Bar(x=g.values, y=[LNAME[i] for i in g.index], orientation="h",
                               marker_color=[LISA[i] for i in g.index],
                               text=[f"{v:.1f}{UNIT}" for v in g.values], textposition="outside"))
        fig.update_layout(height=360, margin=dict(l=4, r=14, t=4, b=4), xaxis_title="Mean " + OUTCOME)
    st.plotly_chart(fig, use_container_width=True)
with c4:
    st.markdown("**Spatial-cluster composition (LISA)**")
    comp = fdf.lisa.value_counts().reindex(["HH","HL","LH","LL","NS"]).dropna()
    fig = go.Figure(go.Bar(x=[LNAME[i] for i in comp.index], y=comp.values,
                           marker_color=[LISA[i] for i in comp.index],
                           text=comp.values, textposition="outside"))
    fig.update_layout(height=360, margin=dict(l=4, r=4, t=4, b=4), yaxis_title="regions")
    st.plotly_chart(fig, use_container_width=True)
with c5:
    st.markdown(f"**{OUTCOME} distribution by cluster**")
    fig = px.box(fdf, x="lisa", y="v", color="lisa", color_discrete_map=LISA, points="all",
                 labels={"lisa": "LISA cluster", "v": OUTCOME + " (" + UNIT + ")"})
    fig.update_layout(height=360, margin=dict(l=4, r=4, t=4, b=4), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.caption("Engine: Streamlit + Plotly · colourblind-safe · regional values for this study · interactive filters in the sidebar.")
