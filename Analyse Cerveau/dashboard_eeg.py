#!/usr/bin/env python3
"""Dashboard EEG en temps réel — lit les Parquet du streaming Spark distribué.

Anti memory-leak : ne charge que les N fichiers les plus récents.
Actualisation toutes les 2 secondes via Dash Interval.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

log = logging.getLogger(__name__)

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
PRED_DIR = BASE / "data" / "stream_eeg" / "predictions"
TOPO_DIR = BASE / "data" / "stream_eeg" / "topomap"

# ── Chargement résilient et optimisé (Anti Memory-Leak) ──────────────────────
def _read_latest_parquets(dir_path: Path, max_files: int) -> pd.DataFrame:
    """Ne lit que les N fichiers Parquet les plus récents pour soulager la RAM."""
    if not dir_path.exists():
        return pd.DataFrame()

    files = list(dir_path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    # Tri par date de modification décroissante (le plus récent en premier)
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    try:
        # Concatène uniquement l'historique récent nécessaire au dashboard
        return pd.concat([pd.read_parquet(f) for f in files[:max_files]], ignore_index=True)
    except Exception as e:
        log.error("Erreur lecture Parquet : %s", e)
        return pd.DataFrame()

def load_predictions() -> pd.DataFrame:
    # On ne lit que les 10 derniers batchs (suffisant pour la timeline temporelle)
    df = _read_latest_parquets(PRED_DIR, max_files=10)
    if df.empty:
        return df
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").drop_duplicates("epoch_id", keep="last")
    return df.tail(100).reset_index(drop=True)

def load_topomap() -> tuple[pd.DataFrame, int | None]:
    # On ne lit que les 2 derniers batchs pour afficher l'image de la topomap actuelle
    df = _read_latest_parquets(TOPO_DIR, max_files=2)
    if df.empty:
        return df, None
    df["event_time"] = pd.to_datetime(df["event_time"])
    last_time = df["event_time"].max()
    last_df = df[df["event_time"] == last_time].copy()
    last_epoch_id = int(last_df["epoch_id"].iloc[0])
    return last_df, last_epoch_id


# ── Palette « Clinique chaleureux » ──────────────────────────────────────────
BG       = '#f7f3ec'   # ivoire chaud
PAPER    = '#fbf8f2'   # crème
CARD_HI  = '#f0eadf'
INK      = '#2b2a26'
TX       = '#3d3a33'
MUTED    = '#8a8276'
BORD     = 'rgba(60, 50, 40, 0.10)'
GRID     = 'rgba(60, 50, 40, 0.06)'

CORAIL   = '#e07a5f'
SAUGE    = '#7a9b76'
OCEAN    = '#5b8baf'
AMBRE    = '#d9a441'
ROSE     = '#c9748a'

CLS_COLORS = {'T0': OCEAN, 'T1': CORAIL, 'T2': SAUGE}
CLS_NAMES  = ['T0', 'T1', 'T2']

WARM_BRAIN = [
    [0.00, '#5b8baf'],
    [0.25, '#a8c5b8'],
    [0.50, '#f4ead5'],
    [0.75, '#e9a87c'],
    [1.00, '#c44536'],
]

GL = dict(
    paper_bgcolor=PAPER, plot_bgcolor=PAPER,
    font=dict(color=TX, family='Inter, system-ui, sans-serif', size=12),
    margin=dict(l=44, r=24, t=52, b=44),
    hoverlabel=dict(bgcolor=PAPER, bordercolor=BORD,
                    font=dict(color=INK, family='Inter', size=12)),
    title=dict(font=dict(size=15, color=INK,
                          family='Fraunces, Georgia, serif'),
               x=0.02, xanchor='left'),
    colorway=[OCEAN, CORAIL, SAUGE, AMBRE, ROSE],
)


# ── Montage EEG 10-05 ────────────────────────────────────────────────────────
PHYSIONET_64 = [
    'Fc5','Fc3','Fc1','Fcz','Fc2','Fc4','Fc6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'Cp5','Cp3','Cp1','Cpz','Cp2','Cp4','Cp6',
    'Fp1','Fpz','Fp2','Af7','Af3','Afz','Af4','Af8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'Ft7','Ft8','T7','T8','T9','T10',
    'Tp7','Tp8','P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'Po7','Po3','Poz','Po4','Po8','O1','Oz','O2','Iz',
]

import mne
mne.set_log_level('ERROR')
_montage = mne.channels.make_standard_montage('standard_1005')
_pos3d_all = _montage.get_positions()['ch_pos']
_lookup = {k.lower(): k for k in _pos3d_all.keys()}
CH_NAMES, CH_POS3D = [], []
for ch in PHYSIONET_64:
    key = _lookup.get(ch.lower())
    if key is not None:
        CH_NAMES.append(ch.upper())
        CH_POS3D.append(_pos3d_all[key])
CH_POS3D = np.array(CH_POS3D)

def _project_2d(pos3d):
    x, y, z = pos3d[:, 0], pos3d[:, 1], pos3d[:, 2]
    r  = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / np.maximum(r, 1e-9))
    phi   = np.arctan2(y, x)
    rho   = np.sin(theta) / (1 + np.cos(theta) + 1e-9)
    return rho * np.cos(phi), rho * np.sin(phi)

CH_X2D, CH_Y2D = _project_2d(CH_POS3D)
MAX_R = max(np.sqrt(CH_X2D**2 + CH_Y2D**2).max() * 1.05, 0.6)


# ── Graphiques ───────────────────────────────────────────────────────────────
def fig_live_accuracy(pred_df: pd.DataFrame) -> go.Figure:
    """Accuracy glissante (rolling window 10)."""
    df = pred_df.copy()
    df["correct"] = df["correct"].astype(int)
    df["rolling_acc"] = df["correct"].rolling(window=min(10, len(df)), min_periods=1).mean()
    df["epoch_idx"] = range(len(df))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["epoch_idx"], y=df["rolling_acc"],
        mode='lines+markers', name='Accuracy glissante',
        line=dict(color=OCEAN, width=2.5),
        marker=dict(color=CORAIL, size=5, opacity=0.6),
        hovertemplate='Epoch %{x}<br>Acc: %{y:.1%}<extra></extra>',
    ))
    fig.add_hline(y=0.333, line_dash='dot', line_color=MUTED,
                  annotation_text='Hasard 33%', annotation_font_color=MUTED)
    fig.update_layout(
        **GL, title_text='Accuracy temps réel · rolling window 10',
        yaxis=dict(tickformat='.0%', range=[0, 1],
                   showgrid=True, gridcolor=GRID, zeroline=False),
        xaxis=dict(title='Epoch #', showgrid=True, gridcolor=GRID, zeroline=False),
        showlegend=False,
    )
    return fig


def fig_prediction_timeline(pred_df: pd.DataFrame) -> go.Figure:
    """Timeline des prédictions : true vs pred, avec erreurs visibles."""
    df = pred_df.copy()
    df["epoch_idx"] = range(len(df))
    df["correct"] = df["correct"].astype(int)

    fig = go.Figure()
    for cls in CLS_NAMES:
        mask = df["true_label"] == cls
        # true label (cercles)
        fig.add_trace(go.Scatter(
            x=df.loc[mask, "epoch_idx"],
            y=[cls] * mask.sum(),
            mode='markers', name=f'Vrai {cls}',
            marker=dict(symbol='circle', size=10, color=CLS_COLORS[cls], opacity=0.4),
            showlegend=True,
        ))
        # correct predictions (petit cercle plein)
        corr = mask & (df["pred_label"] == df["true_label"])
        fig.add_trace(go.Scatter(
            x=df.loc[corr, "epoch_idx"],
            y=[cls] * corr.sum(),
            mode='markers', name=f'✓ {cls}',
            marker=dict(symbol='circle', size=6, color=CLS_COLORS[cls],
                        line=dict(width=1.5, color='white')),
            showlegend=False,
        ))
        # incorrect predictions (croix rouge)
        wrong = mask & (df["pred_label"] != df["true_label"])
        fig.add_trace(go.Scatter(
            x=df.loc[wrong, "epoch_idx"],
            y=[cls] * wrong.sum(),
            mode='markers', name=f'✗ {cls}',
            marker=dict(symbol='x', size=10, color=CORAIL,
                        line=dict(width=2, color=CORAIL)),
            showlegend=False,
        ))

    fig.update_layout(
        **GL, title_text='Prédictions en direct — vrai (◦) vs prédit (✗)',
        yaxis=dict(categoryorder='array', categoryarray=CLS_NAMES, showgrid=False),
        xaxis=dict(title='Epoch #', showgrid=True, gridcolor=GRID, zeroline=False),
        legend=dict(orientation='h', y=-0.18, bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def fig_confusion_live(pred_df: pd.DataFrame) -> go.Figure:
    """Matrice de confusion temps réel."""
    df = pred_df.copy()
    matrix = pd.crosstab(df['true_label'], df['pred_label']) \
               .reindex(index=CLS_NAMES, columns=CLS_NAMES, fill_value=0)
    pct = (matrix.values / matrix.values.sum(axis=1, keepdims=True) * 100).round(1)
    text = [[f'<b>{matrix.values[i,j]}</b><br><span style="font-size:10px;opacity:0.75">{pct[i,j]}%</span>'
             for j in range(3)] for i in range(3)]

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=['T0 prédit','T1 prédit','T2 prédit'],
        y=['T0 réel','T1 réel','T2 réel'],
        colorscale=[[0, '#f0eadf'], [0.5, '#e9a87c'], [1, CORAIL]],
        text=text, texttemplate='%{text}',
        textfont=dict(color=INK, size=14, family='Fraunces, serif'),
        showscale=False,
        hovertemplate='<b>%{y} → %{x}</b><br>%{z} epochs<extra></extra>',
    ))
    fig.update_layout(
        **GL, title_text='Matrice de confusion · live',
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed'),
    )
    return fig


def fig_topomap(topo_df: pd.DataFrame, band: str = 'alpha') -> go.Figure:
    """Topomap alpha en direct à partir des dernières données."""
    if topo_df.empty:
        fig = go.Figure()
        fig.update_layout(**GL, title_text='Topomap — en attente de données...')
        return fig

    # Construire un vecteur de puissance par canal
    ch_power = {}
    for _, row in topo_df.iterrows():
        ch_power[row['channel'].upper()] = row['alpha_power']

    powers = np.array([ch_power.get(ch, np.nan) for ch in CH_NAMES])

    grid_n = 100
    gx = np.linspace(-MAX_R, MAX_R, grid_n)
    gy = np.linspace(-MAX_R, MAX_R, grid_n)
    GX, GY = np.meshgrid(gx, gy)

    sigma = 0.25
    Z = np.zeros_like(GX)
    W = np.zeros_like(GX)
    for i in range(len(CH_NAMES)):
        if np.isnan(powers[i]):
            continue
        d2 = (GX - CH_X2D[i]) ** 2 + (GY - CH_Y2D[i]) ** 2
        w = np.exp(-d2 / (2 * sigma ** 2))
        Z += w * powers[i]
        W += w
    Z = np.where(W > 1e-9, Z / W, np.nan)

    mask = (GX ** 2 + GY ** 2) > (MAX_R * 0.92) ** 2
    Z = np.where(mask, np.nan, Z)

    vmin = float(np.nanmin(Z)) if not np.all(np.isnan(Z)) else 0
    vmax = float(np.nanmax(Z)) if not np.all(np.isnan(Z)) else 1
    if vmax - vmin < 1e-9:
        vmin -= 0.1
        vmax += 0.1

    true_label = topo_df['true_label'].iloc[0] if 'true_label' in topo_df.columns else '?'
    pred_label = topo_df['pred_label'].iloc[0] if 'pred_label' in topo_df.columns else '?'

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z,
        colorscale=WARM_BRAIN,
        zmin=vmin, zmax=vmax,
        contours=dict(showlines=True, coloring='fill',
                      start=vmin, end=vmax,
                      size=(vmax - vmin) / 18),
        line=dict(width=0.5, color='rgba(43,42,38,0.10)'),
        colorbar=dict(
            title=dict(text='Puissance α', font=dict(color=TX, size=11, family='Inter')),
            tickfont=dict(color=TX, size=10, family='Inter'),
            thickness=10, len=0.7, x=1.02,
            outlinewidth=0, bgcolor='rgba(0,0,0,0)',
        ),
        hovertemplate='Puissance : %{z:.2e}<extra></extra>',
    ))

    # Contour tête
    theta = np.linspace(0, 2 * np.pi, 160)
    head_r = MAX_R * 0.92
    fig.add_trace(go.Scatter(
        x=head_r * np.cos(theta), y=head_r * np.sin(theta),
        mode='lines', line=dict(color=INK, width=2.2),
        hoverinfo='skip', showlegend=False,
    ))
    # Nez
    nose_x = [-0.04, 0, 0.04]
    nose_y = [head_r * 0.98, head_r * 1.10, head_r * 0.98]
    fig.add_trace(go.Scatter(
        x=nose_x, y=nose_y, mode='lines',
        line=dict(color=INK, width=2.2),
        hoverinfo='skip', showlegend=False,
    ))

    motor_mask = np.array([n in ('C3', 'CZ', 'C4') for n in CH_NAMES])
    fig.add_trace(go.Scatter(
        x=CH_X2D[~motor_mask], y=CH_Y2D[~motor_mask],
        mode='markers',
        marker=dict(size=4.5, color='rgba(43,42,38,0.45)',
                    line=dict(color='rgba(251,248,242,0.95)', width=0.7)),
        text=[CH_NAMES[i] for i in range(len(CH_NAMES)) if not motor_mask[i]],
        hovertemplate='<b>%{text}</b> (interpolé)<extra></extra>',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=CH_X2D[motor_mask], y=CH_Y2D[motor_mask],
        mode='markers+text',
        marker=dict(size=12, color=PAPER, line=dict(color=CORAIL, width=2.2)),
        text=[CH_NAMES[i] for i in range(len(CH_NAMES)) if motor_mask[i]],
        textposition='top center',
        textfont=dict(color=CORAIL, size=11, family='Fraunces, serif'),
        showlegend=False,
    ))

    fig.update_layout(
        **GL,
        title_text=f'Topographie corticale α · Vrai: {true_label} · Prédit: {pred_label}',
        xaxis=dict(visible=False, range=[-MAX_R * 1.18, MAX_R * 1.18],
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(visible=False, range=[-MAX_R * 1.18, MAX_R * 1.18]),
        autosize=True,
    )
    return fig


# ── App Dash ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='EEG · Streaming Live')

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''

GRAPH_CFG = {'displayModeBar': False, 'responsive': True}
GRAPH_STYLE = {'height': '420px', 'width': '100%'}
GRAPH_STYLE_LG = {'height': '540px', 'width': '100%'}

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div('eeg', className='ns-logo-icon'),
            html.Div([
                html.H1('Motor Imagery · Live', className='ns-title'),
                html.P('Streaming Spark distribué · applyInPandas · Scikit-Learn',
                       className='ns-sub'),
            ]),
        ], className='ns-logo'),
        html.Div([
            html.Span(className='ns-dot', style={'backgroundColor': SAUGE}),
            html.Span('Streaming actif'),
        ], className='ns-live'),
    ], className='ns-header'),

    # Hero
    html.Div([
        html.H2(['Classification temps réel ', html.Em('sur le stream Spark.')]),
        html.P('Prédictions distribuées via applyInPandas. Mise à jour toutes les 2 secondes. '
               'Anti memory-leak : lecture limitée aux derniers fichiers Parquet.'),
    ], className='hero'),

    # KPI row (live)
    html.Div(id='kpi-row', className='kpi-row'),

    # Row 1
    html.Div([
        html.Div([
            html.Div('Précision en direct', className='card-title'),
            html.Div('Accuracy glissante rolling window 10', className='card-sub'),
            dcc.Loading(dcc.Graph(id='acc-graph', config=GRAPH_CFG, style=GRAPH_STYLE),
                        type='circle', color=CORAIL),
        ], className='card'),
        html.Div([
            html.Div('Matrice de confusion', className='card-title'),
            html.Div('Live — s\'enrichit à chaque époque', className='card-sub'),
            dcc.Loading(dcc.Graph(id='conf-graph', config=GRAPH_CFG, style=GRAPH_STYLE),
                        type='circle', color=CORAIL),
        ], className='card'),
    ], className='grid-2'),

    # Row 2 — Timeline
    html.Div([
        html.Div([
            html.Div('Timeline des prédictions', className='card-title'),
            html.Div('Vrai (◦) vs Prédit (✗) — chaque point est une époque', className='card-sub'),
            dcc.Loading(dcc.Graph(id='timeline-graph', config=GRAPH_CFG, style=GRAPH_STYLE_LG),
                        type='circle', color=CORAIL),
        ], className='card'),
    ], className='grid-full'),

    # Row 3 — Topomap
    html.Div([
        html.Div([
            html.Div('Topographie corticale α en direct', className='card-title'),
            html.Div('Interpolation spatiale à partir des électrodes motrices C3, Cz, C4',
                     className='card-sub'),
            dcc.Loading(dcc.Graph(id='topo-graph', config=GRAPH_CFG, style=GRAPH_STYLE_LG),
                        type='circle', color=CORAIL),
        ], className='card'),
    ], className='grid-full'),

    dcc.Interval(id='live-interval', interval=2000, n_intervals=0),

    html.Div(id='last-state', style={'display': 'none'}),  # réservé

], style={'backgroundColor': BG, 'minHeight': '100vh'})


# ── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output('kpi-row', 'children'),
    Output('acc-graph', 'figure'),
    Output('conf-graph', 'figure'),
    Output('timeline-graph', 'figure'),
    Output('topo-graph', 'figure'),
    Input('live-interval', 'n_intervals'),
)
def refresh_all(_n):
    pred_df = load_predictions()
    topo_df, topo_epoch_id = load_topomap()

    if pred_df.empty:
        empty = go.Figure()
        empty.update_layout(**GL, title_text='En attente du stream...')
        kpis = html.Div('Aucune donnée reçue. Lance kedro run --pipeline streaming_eeg.')
        return kpis, empty, empty, empty, empty

    n_epochs = len(pred_df)
    n_correct = int(pred_df['correct'].sum())
    accuracy = n_correct / n_epochs if n_epochs > 0 else 0.0
    n_classes = pred_df['true_label'].nunique()

    def kpi(label, value, sub, color):
        return html.Div([
            html.Div(label, className='kpi-label'),
            html.Div(value, className='kpi-val', style={'color': color}),
            html.Div(sub, className='kpi-sub'),
            html.Div(className='kpi-bar', style={'background': color}),
        ], className='kpi-card')

    kpis = html.Div([
        kpi('Accuracy', f'{accuracy:.1%}',
            f'{n_correct}/{n_epochs} corrects · rolling 10', OCEAN),
        kpi('Époques', f'{n_epochs}',
            f'{n_classes} classes détectées', CORAIL),
        kpi('Dernière époque', str(topo_epoch_id or '—'),
            'ID du dernier batch reçu', SAUGE),
    ], className='kpi-row')

    return (kpis,
            fig_live_accuracy(pred_df),
            fig_confusion_live(pred_df),
            fig_prediction_timeline(pred_df),
            fig_topomap(topo_df))


# ── CSS maison ───────────────────────────────────────────────────────────────
app.index_string = app.index_string.replace(
    '</head>',
    '''<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f7f3ec;font-family:Inter,system-ui,sans-serif;color:#3d3a33;line-height:1.5}
.ns-header{display:flex;justify-content:space-between;align-items:center;
  padding:20px 36px;border-bottom:1px solid rgba(60,50,40,0.10)}
.ns-logo{display:flex;align-items:center;gap:16px}
.ns-logo-icon{background:#2b2a26;color:#fbf8f2;width:44px;height:44px;border-radius:12px;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;
  letter-spacing:0.05em;font-family:Fraunces,Georgia,serif}
.ns-title{font-family:Fraunces,Georgia,serif;font-size:20px;font-weight:600;color:#2b2a26}
.ns-sub{font-size:13px;color:#8a8276;margin-top:1px}
.ns-live{display:flex;align-items:center;gap:8px;font-size:13px;color:#7a9b76;font-weight:500}
.ns-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.hero{padding:28px 36px 8px 36px}
.hero h2{font-family:Fraunces,Georgia,serif;font-size:26px;font-weight:500;color:#2b2a26}
.hero em{color:#e07a5f;font-style:italic}
.hero p{font-size:14px;color:#8a8276;margin-top:6px;max-width:700px}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:14px;padding:16px 36px}
.kpi-card{background:#fbf8f2;border-radius:16px;padding:18px 20px 16px;
  border:1px solid rgba(60,50,40,0.06);position:relative;overflow:hidden}
.kpi-label{font-size:11px;text-transform:uppercase;letter-spacing:0.08em;
  font-weight:600;color:#8a8276;margin-bottom:4px}
.kpi-val{font-family:Fraunces,Georgia,serif;font-size:30px;font-weight:600;line-height:1.1}
.kpi-sub{font-size:12px;color:#8a8276;margin-top:4px}
.kpi-bar{position:absolute;bottom:0;left:0;right:0;height:3px;opacity:0.3}
.grid-2,.grid-full{padding:8px 36px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid-full{display:grid;grid-template-columns:1fr}
.card{background:#fbf8f2;border-radius:20px;padding:22px 24px;
  border:1px solid rgba(60,50,40,0.06)}
.card-title{font-family:Fraunces,Georgia,serif;font-size:16px;font-weight:500;color:#2b2a26}
.card-sub{font-size:12px;color:#8a8276;margin:3px 0 14px 0}
.footer{text-align:center;padding:28px;font-size:12px;color:#8a8276}
.sep{margin:0 10px;opacity:0.4}
</style></head>'''
)


if __name__ == '__main__':
    print('Dashboard live disponible sur http://localhost:8050')
    print(f'  Predictions : {PRED_DIR}')
    print(f'  Topomap     : {TOPO_DIR}')
    app.run(debug=False, host='0.0.0.0', port=8050)
