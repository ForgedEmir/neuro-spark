import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import mne
mne.set_log_level('ERROR')

# ── Données ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'dashboard')

def load():
    signal   = pd.read_parquet(os.path.join(DATA_DIR, 'signal_sample.parquet'))
    features = pd.read_parquet(os.path.join(DATA_DIR, 'features.parquet'))
    preds    = pd.read_parquet(os.path.join(DATA_DIR, 'predictions.parquet'))
    return signal, features, preds

try:
    signal_df, features_df, pred_df = load()
except FileNotFoundError:
    print("\nERREUR : données manquantes dans data/dashboard/")
    print("Lance d'abord le notebook (cellule Export), puis relance ce script.\n")
    raise

feat_imp_path = os.path.join(DATA_DIR, 'feature_importance.parquet')
feat_imp_df   = pd.read_parquet(feat_imp_path) if os.path.exists(feat_imp_path) else None

_all_bands = ['theta', 'alpha', 'beta', 'gamma']
BANDS = [b for b in _all_bands if f'C3_{b}' in features_df.columns]

# ── Plages couleur basées sur les MOYENNES par tâche (pas tous les epochs) ──
# La topomap affiche des moyennes T0/T1/T2 — si la plage vient des epochs individuels,
# toutes les moyennes tombent au milieu et ont la même couleur.
BAND_VMIN, BAND_VMAX = {}, {}
for _b in BANDS:
    _cols = [f'{c}_{_b}' for c in ['C3', 'Cz', 'C4']]
    if all(c in features_df.columns for c in _cols):
        _task_means = features_df.groupby('task_label')[_cols].mean().values.flatten()
        _vmin = float(np.nanmin(_task_means))
        _vmax = float(np.nanmax(_task_means))
        _pad = (_vmax - _vmin) * 0.20 if _vmax > _vmin else abs(_vmax) * 0.10 + 1e-9
        BAND_VMIN[_b] = _vmin - _pad
        BAND_VMAX[_b] = _vmax + _pad

# ── Palette « Clinique chaleureux » (Apple Health / Oura inspired) ───────────
# Tons ivoire, sauge, corail, bleu poudré, ambre doré chaud
BG       = '#f7f3ec'   # ivoire chaud (fond global)
PAPER    = '#fbf8f2'   # crème (cartes)
CARD_HI  = '#f0eadf'   # crème un peu plus profond (hover/inputs)
INK      = '#2b2a26'   # encre brun très sombre (texte principal)
TX       = '#3d3a33'   # texte secondaire
MUTED    = '#8a8276'   # gris-beige doux
BORD     = 'rgba(60, 50, 40, 0.10)'
GRID     = 'rgba(60, 50, 40, 0.06)'

# Accents — palette nature & médicale chaleureuse
CORAIL   = '#e07a5f'   # corail terracotta
SAUGE    = '#7a9b76'   # vert sauge
OCEAN    = '#5b8baf'   # bleu poudré
AMBRE    = '#d9a441'   # ambre doré
ROSE     = '#c9748a'   # rose poudré
LAVANDE  = '#8a7fb8'   # lavande douce

# Mapping classes — nuances chaleureuses qui se distinguent bien
CLS_COLORS = {'T0': OCEAN, 'T1': CORAIL, 'T2': SAUGE}

# Colormap chaleureuse pour le cerveau (sauge → crème → corail → ambre)
WARM_BRAIN = [
    [0.00, '#5b8baf'],   # bleu poudré (faible)
    [0.25, '#a8c5b8'],   # sauge clair
    [0.50, '#f4ead5'],   # crème neutre (centre)
    [0.75, '#e9a87c'],   # pêche
    [1.00, '#c44536'],   # rouge terre cuite (fort)
]

# Colorscale divergente pour la carte différentielle (T2 − T1)
# Bleu → blanc → rouge : négatif = C4 dominant (main droite), positif = C3 dominant (main gauche)
DIFF_BRAIN = [
    [0.00, '#2166ac'],   # bleu profond (T1 domine)
    [0.25, '#74add1'],   # bleu clair
    [0.50, '#f7f7f7'],   # blanc neutre (pas de différence)
    [0.75, '#f4a582'],   # rouge clair
    [1.00, '#c44536'],   # rouge terre cuite (T2 domine)
]

GL = dict(
    paper_bgcolor=PAPER, plot_bgcolor=PAPER,
    font=dict(color=TX, family='Inter, system-ui, sans-serif', size=12),
    # marges réduites : le padding 20px 22px de la carte gère l'espacement extérieur
    margin=dict(l=32, r=16, t=46, b=32),
    hoverlabel=dict(bgcolor=PAPER, bordercolor=BORD,
                    font=dict(color=INK, family='Inter', size=12)),
    title=dict(font=dict(size=15, color=INK,
                          family='Fraunces, Georgia, serif'),
               x=0.02, xanchor='left'),
    colorway=[OCEAN, CORAIL, SAUGE, AMBRE, ROSE, LAVANDE],
    # axes épurés en valeurs par défaut (les figures peuvent override sans conflit)
    template=dict(
        layout=dict(
            xaxis=dict(zeroline=False, ticks='', showline=False),
            yaxis=dict(zeroline=False, ticks='', showline=False),
        )
    ),
)

# ── KPI calculs ──────────────────────────────────────────────────────────────
label_map = {0.0: 'T0', 1.0: 'T1', 2.0: 'T2'}
_df = pred_df.copy()
_df['pred_label'] = _df['prediction'].map(label_map)
accuracy  = (_df['task_label'] == _df['pred_label']).mean()
n_epochs  = len(features_df)
n_sujets_test = pred_df['subject_id'].nunique() if 'subject_id' in pred_df.columns else 14
n_feats   = len([c for c in features_df.columns
                 if c not in ['subject_id', 'run_id', 'epoch_id', 'task_label']])

f1_scores = []
for cls in ['T0', 'T1', 'T2']:
    tp = ((_df['task_label'] == cls) & (_df['pred_label'] == cls)).sum()
    fp = ((_df['task_label'] != cls) & (_df['pred_label'] == cls)).sum()
    fn = ((_df['task_label'] == cls) & (_df['pred_label'] != cls)).sum()
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
f1_macro = np.mean(f1_scores)

per_class_f1 = {'T0': f1_scores[0], 'T1': f1_scores[1], 'T2': f1_scores[2]}
worst_cls = min(per_class_f1, key=per_class_f1.get)

# ── Montage EEG 10-05 ────────────────────────────────────────────────────────
PHYSIONET_64 = [
    'Fc5','Fc3','Fc1','Fcz','Fc2','Fc4','Fc6',
    'C5','C3','C1','Cz','C2','C4','C6',
    'Cp5','Cp3','Cp1','Cpz','Cp2','Cp4','Cp6',
    'Fp1','Fpz','Fp2','Af7','Af3','Afz','Af4','Af8',
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8',
    'Ft7','Ft8','T7','T8','T9','T10',
    'Tp7','Tp8','P7','P5','P3','P1','Pz','P2','P4','P6','P8',
    'Po7','Po3','Poz','Po4','Po8','O1','Oz','O2','Iz'
]

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

def _channel_powers(task, band):
    cols = [f'{c}_{band}' for c in ['C3', 'Cz', 'C4']]
    if not all(c in features_df.columns for c in cols):
        return np.zeros(len(CH_NAMES))
    if task == 'DIFF':
        s_t1 = features_df[features_df['task_label'] == 'T1'][cols].mean()
        s_t2 = features_df[features_df['task_label'] == 'T2'][cols].mean()
        sub = s_t2 - s_t1
    else:
        sub = features_df[features_df['task_label'] == task][cols].mean()
    p_c3, p_cz, p_c4 = float(sub[cols[0]]), float(sub[cols[1]]), float(sub[cols[2]])
    anchors = {'C3': p_c3, 'CZ': p_cz, 'C4': p_c4}
    anchor_pos = np.array([CH_POS3D[CH_NAMES.index(n)] for n in anchors.keys()])
    anchor_val = np.array(list(anchors.values()))
    # Pure spatial interpolation — no baseline blending.
    # Each virtual electrode is the weighted average of the 3 anchors,
    # with wider sigma so the motor-cortex pattern spreads across the scalp.
    sigma = 0.10
    out = np.zeros(len(CH_NAMES))
    for i, p in enumerate(CH_POS3D):
        d = np.linalg.norm(anchor_pos - p, axis=1)
        w = np.exp(-(d ** 2) / (2 * sigma ** 2))
        if w.sum() < 1e-9:
            out[i] = float(np.mean(anchor_val))
        else:
            out[i] = float((w * anchor_val).sum() / w.sum())
    return out

# ── Graphiques ───────────────────────────────────────────────────────────────
def fig_signal():
    fig = px.line(signal_df, x='time', y='Cz', color='task_label',
                  color_discrete_map=CLS_COLORS,
                  labels={'time': 'Temps (s)', 'Cz': 'Amplitude (V)', 'task_label': 'Tâche'})
    fig.update_layout(**GL, title_text='Signal EEG · Canal Cz — sujet S001, run 03',
                      xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
                      yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
                      legend=dict(orientation='h', y=-0.18,
                                  bgcolor='rgba(0,0,0,0)'))
    fig.update_traces(line=dict(width=1.6))
    return fig

def fig_confusion():
    df = pred_df.copy()
    df['pred_label'] = df['prediction'].map(label_map)
    matrix = pd.crosstab(df['task_label'], df['pred_label']) \
               .reindex(index=['T0','T1','T2'], columns=['T0','T1','T2'], fill_value=0)
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
    fig.update_layout(**GL, title_text='Matrice de confusion',
                      xaxis=dict(side='bottom'),
                      yaxis=dict(autorange='reversed'))
    return fig

def fig_bands(band='alpha'):
    cols = [f'C3_{band}', f'Cz_{band}', f'C4_{band}']
    if not all(c in features_df.columns for c in cols):
        return go.Figure()
    data = features_df.groupby('task_label')[cols].mean().reset_index()
    data = data.melt(id_vars='task_label', value_vars=cols,
                     var_name='canal', value_name='puissance')
    data['canal'] = data['canal'].str.replace(f'_{band}', '', regex=False)
    fig = px.bar(data, x='canal', y='puissance', color='task_label',
                 barmode='group', color_discrete_map=CLS_COLORS,
                 labels={'canal': 'Canal', 'puissance': f'Puissance {band.capitalize()}',
                         'task_label': 'Tâche'})
    fig.update_layout(**GL, title_text=f'Puissance spectrale {band.capitalize()} · C3 / Cz / C4',
                      legend=dict(orientation='h', y=-0.18,
                                  bgcolor='rgba(0,0,0,0)'),
                      xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False))
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig

def fig_lateralization():
    diff_cols = [f'diff_{b}' for b in BANDS if f'diff_{b}' in features_df.columns]
    if not diff_cols:
        return go.Figure()
    data = features_df.groupby('task_label')[diff_cols].mean().reset_index()
    data = data.melt(id_vars='task_label', value_vars=diff_cols,
                     var_name='bande', value_name='diff')
    data['bande'] = data['bande'].str.replace('diff_', '', regex=False).str.capitalize()
    fig = px.bar(data, x='bande', y='diff', color='task_label',
                 barmode='group', color_discrete_map=CLS_COLORS,
                 labels={'bande': 'Bande', 'diff': 'C3 − C4', 'task_label': 'Tâche'})
    fig.add_hline(y=0, line_color=BORD, line_width=1)
    fig.update_layout(
        **GL,
        title_text='Latéralisation hémisphérique · C3 − C4',
        legend=dict(orientation='h', y=-0.18, bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        annotations=[dict(
            x=0.5, y=1.10, xref='paper', yref='paper', showarrow=False,
            text=f'<span style="color:{MUTED}">+ → hémisphère gauche dominant · − → hémisphère droit dominant</span>',
            font=dict(size=10, family='Inter')
        )]
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig

def fig_recall():
    df = pred_df.copy()
    df['pred_label'] = df['prediction'].map(label_map)
    rows = []
    for cls in ['T0', 'T1', 'T2']:
        total   = (df['task_label'] == cls).sum()
        correct = ((df['task_label'] == cls) & (df['pred_label'] == cls)).sum()
        tp = correct
        fp = ((df['task_label'] != cls) & (df['pred_label'] == cls)).sum()
        recall = correct / total if total > 0 else 0
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
        rows.append({'classe': cls, 'Recall': recall, 'Précision': prec, 'F1': f1})
    df_r = pd.DataFrame(rows).melt(id_vars='classe', var_name='métrique', value_name='valeur')
    fig = px.bar(df_r, x='valeur', y='classe', color='métrique', orientation='h',
                 barmode='group',
                 color_discrete_map={'Recall': OCEAN, 'Précision': CORAIL, 'F1': SAUGE},
                 labels={'valeur': 'Score', 'classe': '', 'métrique': ''})
    fig.add_vline(x=0.333, line_dash='dot', line_color=MUTED,
                  annotation_text='Hasard 33%', annotation_font_color=MUTED,
                  annotation_font_size=10)
    fig.update_layout(**GL, title_text='Recall · Précision · F1 par classe',
                      legend=dict(orientation='h', y=-0.18, bgcolor='rgba(0,0,0,0)'),
                      xaxis=dict(tickformat='.0%', range=[0, 1],
                                 showgrid=True, gridcolor=GRID, zeroline=False),
                      yaxis=dict(showgrid=False))
    fig.update_traces(marker=dict(line=dict(width=0)))
    return fig

def fig_importance():
    if feat_imp_df is None:
        fig = go.Figure()
        fig.update_layout(**GL, title_text='Feature importance — relancer le notebook')
        return fig
    df = feat_imp_df.sort_values('importance')
    vmax = df['importance'].max()
    # Gradient chaud : terre cuite pour les plus importantes, sauge pour les moindres
    colors = [CORAIL if v/vmax > 0.75 else AMBRE if v/vmax > 0.5
              else OCEAN if v/vmax > 0.3 else SAUGE if v/vmax > 0.15
              else MUTED for v in df['importance']]
    fig = go.Figure(go.Bar(
        x=df['importance'], y=df['feature'], orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>Gini : %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(**GL, title_text='Importance des variables · RandomForest',
                      showlegend=False,
                      yaxis=dict(tickfont=dict(size=10), showgrid=False),
                      xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False))
    return fig

def fig_subject_recall():
    if 'subject_id' not in pred_df.columns:
        fig = go.Figure()
        fig.update_layout(**GL, title_text='Recall par sujet — relancer le notebook')
        return fig
    df = pred_df.copy()
    df['pred_label'] = df['prediction'].map(label_map)
    rows = []
    for subj in sorted(df['subject_id'].unique()):
        s = df[df['subject_id'] == subj]
        total   = len(s)
        correct = (s['task_label'] == s['pred_label']).sum()
        rows.append({'sujet': subj, 'accuracy': correct / total if total > 0 else 0})
    df_s = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    colors = [SAUGE if v >= 0.50 else OCEAN if v >= 0.40 else AMBRE if v >= 0.33 else CORAIL
              for v in df_s['accuracy']]
    fig = go.Figure(go.Bar(
        x=df_s['sujet'], y=df_s['accuracy'],
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate='<b>%{x}</b><br>Accuracy : %{y:.1%}<extra></extra>'
    ))
    fig.add_hline(y=0.333, line_dash='dot', line_color=MUTED,
                  annotation_text='Hasard 33%', annotation_font_color=MUTED,
                  annotation_font_size=10)
    fig.add_hline(y=accuracy, line_dash='dash', line_color=CORAIL,
                  annotation_text=f'Moyenne {accuracy:.1%}', annotation_font_color=CORAIL,
                  annotation_font_size=10)
    fig.update_layout(
        **GL,
        title_text='Lisibilité cérébrale par sujet',
        xaxis=dict(title='', tickfont=dict(size=10), showgrid=False),
        yaxis=dict(title='', tickformat='.0%', range=[0, 1],
                   showgrid=True, gridcolor=GRID, zeroline=False),
    )
    return fig

# ── TOPOMAP 2D ───────────────────────────────────────────────────────────────
def fig_brain_topo_2d(task='T0', band='alpha'):
    powers = _channel_powers(task, band)

    grid_n = 100
    gx = np.linspace(-MAX_R, MAX_R, grid_n)
    gy = np.linspace(-MAX_R, MAX_R, grid_n)
    GX, GY = np.meshgrid(gx, gy)

    sigma = 0.25
    Z = np.zeros_like(GX)
    W = np.zeros_like(GX)
    for i in range(len(CH_NAMES)):
        d2 = (GX - CH_X2D[i]) ** 2 + (GY - CH_Y2D[i]) ** 2
        w  = np.exp(-d2 / (2 * sigma ** 2))
        Z += w * powers[i]
        W += w
    Z = np.where(W > 1e-9, Z / W, np.nan)

    mask = (GX ** 2 + GY ** 2) > (MAX_R * 0.92) ** 2
    Z = np.where(mask, np.nan, Z)

    # For DIFF: symmetric range centered on 0; for normal tasks: fixed per-band range
    if task == 'DIFF':
        abs_max = float(np.nanmax(np.abs(Z[~np.isnan(Z)]))) if not np.all(np.isnan(Z)) else 1e-9
        abs_max = max(abs_max * 1.10, 1e-9)
        vmin, vmax = -abs_max, abs_max
        colorscale = DIFF_BRAIN
    else:
        vmin = BAND_VMIN.get(band, float(np.nanmin(Z)))
        vmax = BAND_VMAX.get(band, float(np.nanmax(Z)))
        colorscale = WARM_BRAIN

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z,
        colorscale=colorscale,
        zmin=vmin, zmax=vmax,
        contours=dict(showlines=True, coloring='fill',
                      start=vmin, end=vmax,
                      size=(vmax - vmin) / 18),
        line=dict(width=0.5, color='rgba(43,42,38,0.10)'),
        colorbar=dict(
            title=dict(text='T2 − T1' if task == 'DIFF' else f'Puissance {band.capitalize()}',
                       font=dict(color=TX, size=11, family='Inter')),
            tickfont=dict(color=TX, size=10, family='Inter'),
            thickness=10, len=0.7, x=1.02,
            outlinewidth=0,
            bgcolor='rgba(0,0,0,0)',
        ),
        hovertemplate='Puissance : %{z:.2e}<extra></extra>',
    ))

    # Contour de la tête
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

    # Oreilles
    for sign in (-1, 1):
        ex = [sign * head_r * 1.02, sign * head_r * 1.09,
              sign * head_r * 1.09, sign * head_r * 1.02]
        ey = [head_r * 0.10, head_r * 0.06, -head_r * 0.06, -head_r * 0.10]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines',
                                 line=dict(color=INK, width=2.2),
                                 hoverinfo='skip', showlegend=False))

    motor_mask = np.array([n in ('C3', 'CZ', 'C4') for n in CH_NAMES])
    fig.add_trace(go.Scatter(
        x=CH_X2D[~motor_mask], y=CH_Y2D[~motor_mask],
        mode='markers',
        marker=dict(size=4.5, color='rgba(43,42,38,0.45)',
                    line=dict(color='rgba(251,248,242,0.95)', width=0.7)),
        text=[CH_NAMES[i] for i in range(len(CH_NAMES)) if not motor_mask[i]],
        customdata=powers[~motor_mask],
        hovertemplate='<b>%{text}</b> (interpolé)<br>%{customdata:.2e}<extra></extra>',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=CH_X2D[motor_mask], y=CH_Y2D[motor_mask],
        mode='markers+text',
        marker=dict(size=12, color=PAPER,
                    line=dict(color=CORAIL, width=2.2)),
        text=[CH_NAMES[i] for i in range(len(CH_NAMES)) if motor_mask[i]],
        textposition='top center',
        textfont=dict(color=CORAIL, size=11, family='Fraunces, serif'),
        customdata=powers[motor_mask],
        hovertemplate='<b>%{text}</b> (mesuré)<br>%{customdata:.2e}<extra></extra>',
        showlegend=False,
    ))

    fig.update_layout(
        **GL,
        title_text=(f'Différentiel cortical · {band.capitalize()} — T2 − T1 (rouge = T2 dominant, bleu = T1 dominant)'
                    if task == 'DIFF'
                    else f'Topographie corticale · {band.capitalize()} — Tâche {task}'),
        xaxis=dict(visible=False, range=[-MAX_R * 1.18, MAX_R * 1.18],
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(visible=False, range=[-MAX_R * 1.18, MAX_R * 1.18]),
        autosize=True,
    )
    return fig

# ── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title='EEG · Motor Imagery')

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

band_options = [{'label': f'{b.capitalize()}  ({lo}–{hi} Hz)', 'value': b}
                for b, lo, hi in [('theta','4','8'),('alpha','8','13'),
                                   ('beta','13','30'),('gamma','30','80')]
                if b in BANDS]

GRAPH_CFG = {'displayModeBar': False, 'responsive': True}

GRAPH_STYLE   = {'height': '420px', 'width': '100%'}
GRAPH_STYLE_LG = {'height': '540px', 'width': '100%'}

def kpi(label, value, sub, color):
    return html.Div([
        html.Div(label, className='kpi-label'),
        html.Div(value, className='kpi-val', style={'color': color}),
        html.Div(sub, className='kpi-sub'),
        html.Div(className='kpi-bar', style={'background': color}),
    ], className='kpi-card')

def card(title, content, sub=None):
    children = [html.Div(title, className='card-title')]
    if sub:
        children.append(html.Div(sub, className='card-sub'))
    children.append(content)
    return html.Div(children, className='card')

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div('eeg', className='ns-logo-icon'),
            html.Div([
                html.H1('Motor Imagery', className='ns-title'),
                html.P(f'UE28 Big Data · HELMo · RandomForest sur {n_sujets_test} sujets de test',
                       className='ns-sub'),
            ]),
        ], className='ns-logo'),
        html.Div([
            html.Span(className='ns-dot'),
            html.Span('Modèle prêt'),
        ], className='ns-live'),
    ], className='ns-header'),

    # Hero intro
    html.Div([
        html.H2(['Lire l\'intention motrice ', html.Em('dans le cerveau.')]),
        html.P('Classification d\'imagerie motrice à partir d\'EEG 64 canaux. '
               'Repos, main gauche, main droite — détectés depuis les ondes cérébrales du cortex moteur.'),
    ], className='hero'),

    # KPI Row
    html.Div([
        kpi('Accuracy', f'{accuracy:.1%}',
            f'Hasard 33% · gain de +{(accuracy - 0.333) * 100:.1f} pp', OCEAN),
        kpi('F1 macro', f'{f1_macro:.1%}',
            'Moyenne harmonique par classe', CORAIL),
        kpi('Epochs', f'{n_epochs:,}',
            f'{n_feats} variables · 4 bandes', SAUGE),
        kpi('Sujets test', f'{n_sujets_test}',
            'Cerveaux jamais vus à l\'entraînement', AMBRE),
        kpi('Classe la plus discrète', worst_cls,
            f'F1 = {per_class_f1[worst_cls]:.1%}', ROSE),
    ], className='kpi-row'),

    # Row 1
    html.Div([
        card('Signal EEG brut',
             dcc.Graph(figure=fig_signal(), config=GRAPH_CFG, style=GRAPH_STYLE_LG),
             sub='Amplitude du canal central Cz au cours du temps, colorée par tâche.'),
        card('Matrice de confusion',
             dcc.Graph(figure=fig_confusion(), config=GRAPH_CFG, style=GRAPH_STYLE),
             sub='Diagonale = prédictions correctes. La couleur s\'intensifie avec le compte.'),
    ], className='grid-2'),

    # Row 2
    html.Div([
        card('Lisibilité cérébrale par sujet',
             dcc.Graph(figure=fig_subject_recall(), config=GRAPH_CFG, style=GRAPH_STYLE),
             sub='Certains cerveaux sont naturellement plus expressifs que d\'autres.'),
    ], className='grid-full'),

    # Row 3
    html.Div([
        card('Recall · Précision · F1 par classe',
             dcc.Graph(figure=fig_recall(), config=GRAPH_CFG, style=GRAPH_STYLE)),
        card('Latéralisation hémisphérique',
             dcc.Graph(figure=fig_lateralization(), config=GRAPH_CFG, style=GRAPH_STYLE),
             sub='Différence de puissance entre C3 (gauche) et C4 (droite).'),
    ], className='grid-2'),

    # Row 4
    html.Div([
        card('Puissance spectrale par bande', html.Div([
            html.Div([
                html.Label('Bande fréquentielle',
                           style={'fontSize': '11px', 'color': MUTED,
                                  'textTransform': 'uppercase',
                                  'letterSpacing': '0.08em',
                                  'fontWeight': 600,
                                  'display': 'block', 'marginBottom': '6px'}),
                dcc.Dropdown(id='band-select', options=band_options,
                             value='alpha', clearable=False,
                             style={'width': '240px'}),
            ], style={'marginBottom': '14px'}),
            dcc.Graph(id='bands-graph', config=GRAPH_CFG, style=GRAPH_STYLE),
        ])),
        card('Importance des variables',
             dcc.Graph(figure=fig_importance(), config=GRAPH_CFG, style=GRAPH_STYLE),
             sub='Variables qui contribuent le plus aux décisions du RandomForest.'),
    ], className='grid-2'),

    # Row 5 — Brain
    html.Div([
        html.Div([
            html.Div('Cartographie corticale', className='card-title'),
            html.Div('Montage international 10-05 · 64 électrodes. C3, Cz et C4 sont mesurés '
                     'directement ; les autres canaux sont extrapolés spatialement.',
                     className='card-sub'),
            html.Div([
                html.Div([
                    html.Label('Tâche'),
                    dcc.Dropdown(id='brain-task',
                                 options=[{'label': 'T0 — Repos',          'value': 'T0'},
                                          {'label': 'T1 — Main gauche',    'value': 'T1'},
                                          {'label': 'T2 — Main droite',    'value': 'T2'},
                                          {'label': 'T2 − T1 · Différentiel', 'value': 'DIFF'}],
                                 value='T0', clearable=False,
                                 style={'width': '240px'}),
                ]),
                html.Div([
                    html.Label('Bande fréquentielle'),
                    dcc.Dropdown(id='brain-band', options=band_options,
                                 value='alpha', clearable=False,
                                 style={'width': '220px'}),
                ]),
                html.Div([
                    html.Label('\u00a0'),
                    html.Button('▶ Play', id='topo-play-btn',
                                n_clicks=0,
                                style={
                                    'padding': '10px 24px',
                                    'borderRadius': '999px',
                                    'border': 'none',
                                    'background': CORAIL,
                                    'color': PAPER,
                                    'fontFamily': 'Inter, sans-serif',
                                    'fontSize': '13px',
                                    'fontWeight': '600',
                                    'letterSpacing': '0.02em',
                                    'cursor': 'pointer',
                                    'boxShadow': '0 4px 12px rgba(224, 122, 95, 0.30), '
                                                 '0 2px 4px rgba(224, 122, 95, 0.15)',
                                    'transition': 'all 0.2s ease',
                                }),
                ]),
                dcc.Interval(id='topo-interval', interval=2000, disabled=True, n_intervals=0),

            ], className='brain-controls'),
            dcc.Loading(
                dcc.Graph(id='brain-graph', config=GRAPH_CFG, style=GRAPH_STYLE_LG),
                type='circle', color=CORAIL,
            ),
        ], className='card'),
    ], className='grid-full'),

    # Footer
    html.Div([
        'Spark MLlib · RandomForest avec CrossValidator',
        html.Span('·', className='sep'),
        'Export Parquet via PySpark',
        html.Span('·', className='sep'),
        'UE28 Big Data — HELMo',
    ], className='footer'),

], style={'backgroundColor': BG, 'minHeight': '100vh'})

# ── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(Output('brain-graph', 'figure'),
              Input('brain-task', 'value'),
              Input('brain-band', 'value'))
def update_brain(task, band):
    return fig_brain_topo_2d(task, band)

@app.callback(Output('bands-graph', 'figure'), Input('band-select', 'value'))
def update_bands(band):
    return fig_bands(band)

# Animate: toggle Play/Pause and enable/disable interval
_ANIM_TASKS = ['T0', 'T1', 'T2']

@app.callback(
    Output('topo-interval', 'disabled'),
    Output('topo-play-btn', 'children'),
    Input('topo-play-btn', 'n_clicks'),
    State('topo-interval', 'disabled'),
)
def toggle_animation(n_clicks, is_disabled):
    if n_clicks == 0:
        return True, '▶ Play'
    if is_disabled:
        return False, '⏸ Pause'
    return True, '▶ Play'

@app.callback(
    Output('brain-task', 'value'),
    Input('topo-interval', 'n_intervals'),
    State('brain-task', 'value'),
    prevent_initial_call=True,
)
def advance_task(n_intervals, current_task):
    if current_task not in _ANIM_TASKS:
        return 'T0'
    idx = _ANIM_TASKS.index(current_task)
    return _ANIM_TASKS[(idx + 1) % len(_ANIM_TASKS)]

if __name__ == '__main__':
    print('Dashboard disponible sur http://localhost:8050')
    app.run(debug=False, host='0.0.0.0', port=8050)
