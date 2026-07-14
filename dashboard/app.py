"""
HIV-TB Ghana — Python Dash Application
Run: python3 app.py
Then open: http://127.0.0.1:8050
"""
import json
from pathlib import Path
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'outputs' / 'data' / 'ghana_261_final_results.geojson'
SHAP_FILE = ROOT / 'outputs' / 'tables' / 'shap_feature_importance.csv'

print('Loading data...')
gdf = gpd.read_file(DATA)
gdf['geometry'] = gdf['geometry'].simplify(0.005, preserve_topology=True)
geojson = json.loads(gdf[['DISTRICT', 'REGION', 'geometry']].to_json())
df = gdf.drop(columns='geometry').copy()
shap_imp = pd.read_csv(SHAP_FILE).head(10)

MAP_VARS = {
 'HIV_Prev_Total_pct': 'HIV Prevalence (%)',
 'TB_Incidence_per100k': 'TB Incidence (/100k)',
 'TB_HIV_CoInfection_pct': 'TB-HIV Co-infection (%)',
 'ART_Coverage_pct': 'ART Coverage (%)',
 'VCT_Uptake_pct': 'VCT Uptake (%)',
 'Ensemble_Risk_Score': 'ML Hotspot Risk',
 'Poverty_Incidence_pct': 'Poverty (%)',
 'Illiteracy_Rate_pct': 'Illiteracy (%)',
 'Doctors_per10k': 'Doctors / 10k',
 'Nurses_per10k': 'Nurses / 10k',
}
REGIONS = sorted(df['REGION'].unique().tolist())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
 title='HIV-TB Ghana · 261 Districts')

def kpi_card(value, label, color='danger'):
 return dbc.Card(
 dbc.CardBody([
 html.H3(str(value), className=f'text-{color} mb-0'),
 html.Small(label, className='text-muted'),
 ]),
 className='shadow-sm h-100',
 )

app.layout = dbc.Container([
 html.Div([
 html.H1('HIV-TB Co-infection Dashboard — Ghana 261 Districts',
 className='mt-3 mb-1'),
 html.P('Valentine Golden Ghanem · Biomedical Scientist · April 2026',
 className='text-muted font-italic'),
 ]),
 html.Hr(),

 # KPIs
 dbc.Row([
 dbc.Col(kpi_card(len(df), 'Districts'), width=2),
 dbc.Col(kpi_card(f"{df['TB_HIV_CoInfection_pct'].mean():.1f}%", 'Mean co-infection'),
 width=2),
 dbc.Col(kpi_card(int((df['LISA_cluster'] == 'High-High').sum()), 'LISA HH clusters'),
 width=2),
 dbc.Col(kpi_card(int((df['BvLISA_cluster'] == 'High-High').sum()),
 'Bivariate HH'), width=2),
 dbc.Col(kpi_card('0.998', 'Best ML CV AUC'), width=2),
 dbc.Col(kpi_card('0.525', "Bivariate Moran's I"), width=2),
 ], className='mb-3'),

 # Controls
 dbc.Row([
 dbc.Col([
 html.Label('Variable:', className='fw-bold'),
 dcc.Dropdown(
 id='var-select',
 options=[{'label': v, 'value': k} for k, v in MAP_VARS.items()],
 value='TB_HIV_CoInfection_pct',
 ),
 ], width=4),
 dbc.Col([
 html.Label('Region:', className='fw-bold'),
 dcc.Dropdown(
 id='region-select',
 options=[{'label': 'All regions', 'value': 'all'}] +
 [{'label': r, 'value': r} for r in REGIONS],
 value='all',
 ),
 ], width=4),
 dbc.Col([
 html.Label('Top-N:', className='fw-bold'),
 dcc.Dropdown(
 id='top-n',
 options=[{'label': str(n), 'value': n} for n in [10, 20, 30]],
 value=20,
 ),
 ], width=2),
 ], className='mb-3 p-3 bg-white rounded shadow-sm'),

 # Maps row
 dbc.Row([
 dbc.Col(dbc.Card(dbc.CardBody([
 html.H5('Choropleth Map'), dcc.Graph(id='choropleth-map')]),
 className='shadow-sm'), width=6),
 dbc.Col(dbc.Card(dbc.CardBody([
 html.H5('LISA Cluster Map'), dcc.Graph(id='lisa-map')]),
 className='shadow-sm'), width=6),
 ], className='mb-3'),

 # Bar row
 dbc.Row([
 dbc.Col(dbc.Card(dbc.CardBody([
 html.H5('Top-N Districts'), dcc.Graph(id='top-bar')]),
 className='shadow-sm'), width=6),
 dbc.Col(dbc.Card(dbc.CardBody([
 html.H5('SHAP Feature Importance'), dcc.Graph(id='shap-bar')]),
 className='shadow-sm'), width=6),
 ], className='mb-3'),

 # Bivariate + risk
 dbc.Row([
 dbc.Col(dbc.Card(dbc.CardBody([
 html.H5('Bivariate LISA (HIV × TB)'), dcc.Graph(id='bv-map')]),
 className='shadow-sm'), width=6),
 dbc.Col(dbc.Card(dbc.CardBody([
 html.H5('Ensemble ML Risk'), dcc.Graph(id='risk-map')]),
 className='shadow-sm'), width=6),
 ], className='mb-3'),

 html.Div(className='mb-3'),
 html.Footer(
 html.P('© 2026 Valentine Golden Ghanem · MIT licence · Ghana COCOBOD Cocoa Clinic',
 className='text-center text-muted small'),
 ),
], fluid=True, style={'background-color': '#f7fafc', 'min-height': '100vh'})


@app.callback(
 Output('choropleth-map', 'figure'),
 Input('var-select', 'value'),
 Input('region-select', 'value'),
)
def make_choropleth(var, region):
 dff = df if region == 'all' else df[df['REGION'] == region]
 fig = px.choropleth(
 dff, geojson=geojson, locations='DISTRICT',
 featureidkey='properties.DISTRICT',
 color=var, color_continuous_scale='Reds',
 hover_data=['REGION', 'HIV_Prev_Total_pct', 'TB_Incidence_per100k',
 'TB_HIV_CoInfection_pct'],
 )
 fig.update_geos(fitbounds='locations', visible=False)
 fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520)
 return fig


@app.callback(Output('lisa-map', 'figure'), Input('region-select', 'value'))
def make_lisa(region):
 dff = df if region == 'all' else df[df['REGION'] == region]
 cmap = {'High-High': '#d7191c', 'Low-Low': '#2c7bb6',
 'High-Low': '#fdae61', 'Low-High': '#abd9e9',
 'Not Significant': '#f0f0f0'}
 dff = dff.copy()
 dff['color'] = dff['LISA_cluster'].map(cmap)
 fig = go.Figure()
 for cl, c in cmap.items():
  sub = dff[dff['LISA_cluster'] == cl]
  if len(sub) == 0: continue
  fig.add_trace(go.Choropleth(
  geojson=geojson, locations=sub['DISTRICT'],
  featureidkey='properties.DISTRICT',
  z=[1] * len(sub), colorscale=[[0, c], [1, c]],
  showscale=False, name=f'{cl} (n={len(sub)})',
  showlegend=True,
  hovertemplate='<b>%{location}</b><br>' + cl + '<extra></extra>',
  ))
  fig.update_geos(fitbounds='locations', visible=False)
  fig.update_layout(margin=dict(l=0, r=80, t=10, b=0), height=520,
  legend=dict(x=1.02, y=0.5, font=dict(size=9)))
  return fig


@app.callback(Output('bv-map', 'figure'), Input('region-select', 'value'))
def make_bv(region):
 dff = df if region == 'all' else df[df['REGION'] == region]
 cmap = {'High-High': '#d7191c', 'Low-Low': '#2c7bb6',
 'High-Low': '#fdae61', 'Low-High': '#abd9e9',
 'Not Significant': '#f0f0f0'}
 fig = go.Figure()
 for cl, c in cmap.items():
  sub = dff[dff['BvLISA_cluster'] == cl]
  if len(sub) == 0: continue
  fig.add_trace(go.Choropleth(
  geojson=geojson, locations=sub['DISTRICT'],
  featureidkey='properties.DISTRICT',
  z=[1] * len(sub), colorscale=[[0, c], [1, c]],
  showscale=False, name=f'{cl} (n={len(sub)})',
  showlegend=True,
  hovertemplate='<b>%{location}</b><br>' + cl + '<extra></extra>',
  ))
  fig.update_geos(fitbounds='locations', visible=False)
  fig.update_layout(margin=dict(l=0, r=80, t=10, b=0), height=450,
  legend=dict(x=1.02, y=0.5, font=dict(size=9)))
  return fig


@app.callback(Output('risk-map', 'figure'), Input('region-select', 'value'))
def make_risk(region):
 dff = df if region == 'all' else df[df['REGION'] == region]
 fig = px.choropleth(
 dff, geojson=geojson, locations='DISTRICT',
 featureidkey='properties.DISTRICT',
 color='Ensemble_Risk_Score', color_continuous_scale='RdYlGn_r',
 range_color=[0, 1],
 hover_data=['REGION', 'TB_HIV_CoInfection_pct'],
 )
 fig.update_geos(fitbounds='locations', visible=False)
 fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=450)
 return fig


@app.callback(
 Output('top-bar', 'figure'),
 Input('var-select', 'value'),
 Input('region-select', 'value'),
 Input('top-n', 'value'),
)
def make_topbar(var, region, n):
 dff = df if region == 'all' else df[df['REGION'] == region]
 dff = dff.nlargest(n, var)
 fig = px.bar(
 dff, x=var, y='District', orientation='h',
 color=var, color_continuous_scale='Reds',
 hover_data=['REGION'],
 )
 fig.update_layout(margin=dict(l=180, r=30, t=10, b=30), height=450,
 yaxis=dict(autorange='reversed'))
 return fig


@app.callback(Output('shap-bar', 'figure'), Input('var-select', 'value'))
def make_shap(_):
 LABELS = {
 'HIV_Prev_Total_pct': 'HIV Prevalence',
 'HIV_Prev_Men_pct': 'HIV Prev (Men)',
 'VCT_Uptake_pct': 'VCT Uptake',
 'Poverty_Incidence_pct': 'Poverty Incidence',
 'TB_Incidence_per100k': 'TB Incidence',
 'Illiteracy_Rate_pct': 'Illiteracy',
 'Accepting_Attitudes_pct': 'Accepting Attitudes',
 'Uninsurance_Rate_pct': 'Uninsurance',
 'Unemployment_Rate_pct': 'Unemployment',
 'ART_Coverage_pct': 'ART Coverage',
 }
 df_s = shap_imp.copy()
 df_s['Label'] = df_s['Feature'].map(lambda f: LABELS.get(f, f))
 fig = px.bar(
 df_s, x='Mean_abs_SHAP', y='Label', orientation='h',
 color='Mean_abs_SHAP', color_continuous_scale='Blues',
 text=df_s['Mean_abs_SHAP'].round(2),
 )
 fig.update_layout(margin=dict(l=160, r=30, t=10, b=30), height=450,
 yaxis=dict(autorange='reversed'))
 return fig


if __name__ == '__main__':
 print('Starting dashboard at http://127.0.0.1:8050')
 app.run(debug=False, host='127.0.0.1', port=8050)
