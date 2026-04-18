import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from scipy.interpolate import griddata

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    "DASHBOARD_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dashboard")
)
PORT = int(os.environ.get("DASHBOARD_PORT", 8050))

# ── Données ──────────────────────────────────────────────────────────────────
def load():
    missing = []
    for f in ['signal_sample.parquet', 'features.parquet', 'predictions.parquet']:
        p = os.path.join(DATA_DIR, f)
        if not os.path.exists(p):
            missing.append(f)
    if missing:
        print(f"\nERREUR : fichiers manquants dans {DATA_DIR}/")
        print(f"  Manquants : {', '.join(missing)}")
        print("  Lance d'abord le notebook (cellule Export), puis relance ce script.\n")
        sys.exit(1)

    signal   = pd.read_parquet(os.path.join(DATA_DIR, 'signal_sample.parquet'))
    features = pd.read_parquet(os.path.join(DATA_DIR, 'features.parquet'))
    preds    = pd.read_parquet(os.path.join(DATA_DIR, 'predictions.parquet'))
    return signal, features, preds

signal_df, features_df, pred_df = load()

# ── Styles ───────────────────────────────────────────────────────────────────
COLORS   = {'T0': '#636EFA', 'T1': '#EF553B', 'T2': '#00CC96'}
BG       = '#0f0f1a'
CARD_BG  = '#1a1a2e'
TEXT     = '#e0e0e0'
MUTED    = '#888888'
GL = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family='Inter, sans-serif'),
    margin=dict(l=40, r=20, t=40, b=40)
)

def card(title, graph):
    return html.Div([
        html.P(title, style={'color': MUTED, 'fontSize': '12px',
                             'margin': '0 0 8px 0', 'textTransform': 'uppercase',
                             'letterSpacing': '1px'}),
        graph
    ], style={
        'backgroundColor': CARD_BG, 'borderRadius': '10px',
        'padding': '16px', 'flex': '1', 'minWidth': '420px'
    })

# ── Graphiques ────────────────────────────────────────────────────────────────
def fig_signal():
    fig = px.line(
        signal_df, x='time', y='Cz', color='task_label',
        color_discrete_map=COLORS,
        labels={'time': 'Temps (s)', 'Cz': 'Amplitude (V)', 'task_label': 'Tâche'}
    )
    fig.update_layout(**GL, title='Signal EEG — Canal Cz (S001 / R03)')
    return fig

def fig_confusion():
    label_map = {0.0: 'T0', 1.0: 'T1', 2.0: 'T2'}
    df = pred_df.copy()
    df['pred_label'] = df['prediction'].map(label_map)
    matrix = pd.crosstab(df['task_label'], df['pred_label']) \
               .reindex(index=['T0','T1','T2'], columns=['T0','T1','T2'], fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=['T0 prédit', 'T1 prédit', 'T2 prédit'],
        y=['T0 réel', 'T1 réel', 'T2 réel'],
        colorscale='Blues',
        text=matrix.values,
        texttemplate='%{text}',
        showscale=False
    ))
    fig.update_layout(**GL, title='Matrice de confusion — RandomForest')
    return fig

def fig_alpha():
    cols = ['C3_alpha', 'Cz_alpha', 'C4_alpha']
    alpha = features_df.groupby('task_label')[cols].mean().reset_index()
    alpha = alpha.melt(id_vars='task_label', value_vars=cols,
                       var_name='canal', value_name='puissance')
    alpha['canal'] = alpha['canal'].str.replace('_alpha', '', regex=False)
    fig = px.bar(
        alpha, x='canal', y='puissance', color='task_label',
        barmode='group', color_discrete_map=COLORS,
        labels={'canal': 'Canal', 'puissance': 'Puissance Alpha (V²)', 'task_label': 'Tâche'}
    )
    fig.update_layout(**GL, title='Puissance Alpha — Cortex moteur (C3 / Cz / C4)')
    return fig

def fig_recall():
    label_map = {0.0: 'T0', 1.0: 'T1', 2.0: 'T2'}
    df = pred_df.copy()
    df['pred_label'] = df['prediction'].map(label_map)
    rows = []
    for cls in ['T0', 'T1', 'T2']:
        total   = (df['task_label'] == cls).sum()
        correct = ((df['task_label'] == cls) & (df['pred_label'] == cls)).sum()
        rows.append({'classe': cls, 'recall': round(correct / total, 3) if total > 0 else 0})
    df_r = pd.DataFrame(rows)
    fig = px.bar(
        df_r, x='recall', y='classe', orientation='h',
        color='classe', color_discrete_map=COLORS,
        labels={'recall': 'Recall', 'classe': 'Classe'}
    )
    fig.add_vline(x=0.333, line_dash='dot', line_color='#aaa',
                  annotation_text='Baseline aléatoire (33%)',
                  annotation_font_color='#aaa')
    fig.update_layout(**GL, title='Recall par classe — vs baseline 33%',
                      showlegend=False)
    return fig

def fig_brain_topo(task='T0', band='alpha'):
    band_cols = {
        'alpha': ['C3_alpha', 'Cz_alpha', 'C4_alpha'],
        'beta':  ['C3_beta',  'Cz_beta',  'C4_beta'],
        'gamma': ['C3_gamma', 'Cz_gamma', 'C4_gamma'],
    }
    cols = band_cols[band]
    task_data = features_df[features_df['task_label'] == task][cols].mean()
    channels  = ['C3', 'Cz', 'C4']

    # Positions 10-20 (système de référence EEG standard)
    elec_pos = {'C3': (-0.50, 0.0), 'Cz': (0.0, 0.12), 'C4': (0.50, 0.0)}
    elec_x   = np.array([elec_pos[c][0] for c in channels])
    elec_y   = np.array([elec_pos[c][1] for c in channels])
    powers   = np.array([task_data[f'{c}_{band}'] for c in channels])

    # Points fantômes sur le pourtour pour que l'interpolation ne diverge pas
    n_bound  = 16
    angles   = np.linspace(0, 2 * np.pi, n_bound, endpoint=False)
    bx = 0.95 * np.cos(angles)
    by = 0.95 * np.sin(angles)
    bv = np.full(n_bound, np.mean(powers))

    all_x = np.concatenate([elec_x, bx])
    all_y = np.concatenate([elec_y, by])
    all_v = np.concatenate([powers, bv])

    # Grille d'interpolation
    res = 180
    gx  = np.linspace(-1.05, 1.05, res)
    gy  = np.linspace(-1.05, 1.05, res)
    GX, GY = np.meshgrid(gx, gy)

    zi = griddata((all_x, all_y), all_v, (GX, GY), method='cubic')

    # Masque circulaire (hors tête = NaN → transparent)
    zi[GX ** 2 + GY ** 2 > 0.97 ** 2] = np.nan

    # ── Figure ──
    fig = go.Figure()

    # Heatmap interpolée
    fig.add_trace(go.Heatmap(
        z=zi,
        x=gx,
        y=gy,
        colorscale='RdBu_r',
        zsmooth='best',
        showscale=True,
        colorbar=dict(
            title=dict(text=f'{band.capitalize()} (V²)', font=dict(color=TEXT, size=11)),
            tickfont=dict(color=TEXT, size=10),
            thickness=14, len=0.75,
        ),
        hoverinfo='skip',
    ))

    # Contour tête
    t = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(go.Scatter(
        x=np.cos(t), y=np.sin(t),
        mode='lines', line=dict(color='rgba(255,255,255,0.6)', width=2),
        showlegend=False, hoverinfo='skip',
    ))

    # Nez
    fig.add_trace(go.Scatter(
        x=[-0.12, 0.0, 0.12, None], y=[0.93, 1.18, 0.93, None],
        mode='lines', line=dict(color='rgba(255,255,255,0.6)', width=2),
        showlegend=False, hoverinfo='skip',
    ))

    # Oreilles
    for ex, ey in [
        ([-1.0, -1.12, -1.12, -1.0, None], [0.12, 0.06, -0.06, -0.12, None]),
        ([ 1.0,  1.12,  1.12,  1.0, None], [0.12, 0.06, -0.06, -0.12, None]),
    ]:
        fig.add_trace(go.Scatter(
            x=ex, y=ey,
            mode='lines', line=dict(color='rgba(255,255,255,0.6)', width=2),
            showlegend=False, hoverinfo='skip',
        ))

    # Points électrodes
    fig.add_trace(go.Scatter(
        x=elec_x, y=elec_y,
        mode='markers+text',
        marker=dict(size=16, color='white',
                    line=dict(color='#111', width=2), symbol='circle'),
        text=channels,
        textposition='top center',
        textfont=dict(color='white', size=12, family='Inter, sans-serif'),
        customdata=powers,
        hovertemplate='<b>%{text}</b><br>%{customdata:.2e} V²<extra></extra>',
        showlegend=False,
    ))

    # Labels latéralisation
    fig.add_annotation(x=-1.3, y=-1.1,
                       text='◀ Hémisphère gauche<br><span style="font-size:10px;color:#EF553B">contrôle main droite</span>',
                       showarrow=False, font=dict(color=TEXT, size=11), align='center')
    fig.add_annotation(x= 1.3, y=-1.1,
                       text='Hémisphère droit ▶<br><span style="font-size:10px;color:#00CC96">contrôle main gauche</span>',
                       showarrow=False, font=dict(color=TEXT, size=11), align='center')

    fig.update_layout(
        **GL,
        title=f'Topomap EEG — {band.capitalize()} · Tâche {task}',
        xaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(range=[-1.35, 1.45], showgrid=False, zeroline=False,
                   showticklabels=False),
        height=460,
    )
    return fig

# ── KPI cards ─────────────────────────────────────────────────────────────────
label_map = {0.0: 'T0', 1.0: 'T1', 2.0: 'T2'}
_df = pred_df.copy()
_df['pred_label'] = _df['prediction'].map(label_map)
accuracy = (_df['task_label'] == _df['pred_label']).mean()
n_epochs = len(features_df)
n_sujets = features_df['subject_id'].nunique() if 'subject_id' in features_df.columns else '—'

def kpi(label, value):
    return html.Div([
        html.P(value, style={'color': TEXT, 'fontSize': '26px',
                              'fontWeight': '700', 'margin': '0'}),
        html.P(label, style={'color': MUTED, 'fontSize': '12px',
                              'margin': '4px 0 0 0', 'textTransform': 'uppercase',
                              'letterSpacing': '1px'})
    ], style={'backgroundColor': CARD_BG, 'borderRadius': '10px',
              'padding': '16px 20px', 'flex': '1'})

# ── Layout ────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='EEG Dashboard — PoC HELMo')

GRAPH_CFG = {'displayModeBar': False}

DROPDOWN_STYLE = {
    'backgroundColor': '#2a2a3e', 'color': TEXT,
    'border': '1px solid #3a3a5e', 'borderRadius': '6px',
    'fontSize': '13px',
}

app.layout = html.Div([

    # Header
    html.Div([
        html.H2('PoC EEG Motor Imagery', style={'color': TEXT, 'margin': '0',
                                                  'fontWeight': '700'}),
        html.P('UE28 Big Data — HELMo 2025-2026  ·  RandomForest sur cortex moteur C3 / Cz / C4',
               style={'color': MUTED, 'margin': '4px 0 0 0', 'fontSize': '13px'}),
    ], style={'padding': '24px 28px 16px', 'borderBottom': '1px solid #2a2a3e'}),

    # KPIs
    html.Div([
        kpi('Accuracy', f'{accuracy:.1%}'),
        kpi('Epochs analysées', f'{n_epochs:,}'),
        kpi('Sujets de test', '14'),
        kpi('Features', '12  (9 bandes + 3 diff C3-C4)'),
    ], style={'display': 'flex', 'gap': '12px', 'padding': '16px 28px'}),

    # Grille 2×2
    html.Div([
        html.Div([
            card('Signal EEG brut', dcc.Graph(figure=fig_signal(), config=GRAPH_CFG)),
            card('Matrice de confusion', dcc.Graph(figure=fig_confusion(), config=GRAPH_CFG)),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}),
        html.Div([
            card('Puissance Alpha par canal', dcc.Graph(figure=fig_alpha(), config=GRAPH_CFG)),
            card('Recall par classe', dcc.Graph(figure=fig_recall(), config=GRAPH_CFG)),
        ], style={'display': 'flex', 'gap': '12px'}),
    ], style={'padding': '12px 28px'}),

    # Topomap interactive
    html.Div([
        html.Div([
            html.P('TOPOMAP EEG — ACTIVITÉ DU CORTEX MOTEUR', style={
                'color': MUTED, 'fontSize': '12px', 'margin': '0 0 12px 0',
                'textTransform': 'uppercase', 'letterSpacing': '1px',
            }),
            html.Div([
                html.Div([
                    html.Label('Tâche', style={'color': MUTED, 'fontSize': '12px',
                                               'marginBottom': '4px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='brain-task',
                        options=[{'label': 'T0 — Repos',         'value': 'T0'},
                                 {'label': 'T1 — Main gauche',   'value': 'T1'},
                                 {'label': 'T2 — Main droite',   'value': 'T2'}],
                        value='T0', clearable=False, style=DROPDOWN_STYLE,
                    ),
                ], style={'width': '210px'}),
                html.Div([
                    html.Label('Bande fréquentielle', style={'color': MUTED, 'fontSize': '12px',
                                               'marginBottom': '4px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='brain-band',
                        options=[{'label': 'Alpha  (8–13 Hz)',  'value': 'alpha'},
                                 {'label': 'Beta   (13–30 Hz)', 'value': 'beta'},
                                 {'label': 'Gamma  (30–80 Hz)', 'value': 'gamma'}],
                        value='alpha', clearable=False, style=DROPDOWN_STYLE,
                    ),
                ], style={'width': '210px'}),
            ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '12px'}),
            dcc.Graph(id='brain-graph', config=GRAPH_CFG),
        ], style={
            'backgroundColor': CARD_BG, 'borderRadius': '10px', 'padding': '16px',
        }),
    ], style={'padding': '12px 28px 28px'}),

], style={'backgroundColor': BG, 'minHeight': '100vh',
          'fontFamily': 'Inter, Segoe UI, sans-serif'})

# ── Callback topomap ──────────────────────────────────────────────────────────
@app.callback(
    Output('brain-graph', 'figure'),
    Input('brain-task', 'value'),
    Input('brain-band', 'value'),
)
def update_brain(task, band):
    return fig_brain_topo(task, band)

if __name__ == '__main__':
    print(f'Dashboard disponible sur http://localhost:{PORT}')
    app.run(debug=False, host='0.0.0.0', port=PORT)
