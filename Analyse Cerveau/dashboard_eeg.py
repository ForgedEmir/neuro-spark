"""Dashboard streaming EEG INTERACTIF.

Mode pilote : l'utilisateur clique sur un bouton (Repos / Mouvement réel /
Mouvement imaginé) et le dashboard envoie un epoch correspondant dans le
file source de Spark. Le système prédit, et le dashboard affiche le résultat
avec **annotations pédagogiques** (qu'est-ce qu'on est censé voir ?).

Lancer :
    make demo        # Spark + dashboard ; toi tu pilotes les epochs
"""
from __future__ import annotations

import logging
import random
import time
import uuid
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html
from scipy.interpolate import griddata

logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────
PARQUET_DIR = Path("data/parquet")
STREAM_INPUT = Path("data/stream_eeg/input")
PRED_DIR = Path("data/stream_eeg/output/predictions")
TOPO_DIR = Path("data/stream_eeg/output/topomap")
REFRESH_MS = 1500

EEG_FS = 160
EPOCH_SAMPLES = EEG_FS * 2

# Palette
BG, PAPER, CARD_HI = "#f5f1e8", "#fbf8f2", "#f0eadf"
INK, TX, MUTED = "#2b2a26", "#3d3a33", "#9a9183"
BORD = "rgba(60, 50, 40, 0.08)"
GRID = "rgba(60, 50, 40, 0.05)"

COLOR_T0 = "#9aa8b3"
COLOR_T1 = "#5b8baf"
COLOR_T2 = "#c44536"
AMBRE = "#d9a441"
CLASS_COLORS = {"T0": COLOR_T0, "T1": COLOR_T1, "T2": COLOR_T2}
CLASS_LABELS = {
    "T0": "Repos",
    "T1": "Mouvement réel",
    "T2": "Mouvement imaginé",
}
CLASS_HINTS = {
    "T0": "Le sujet est au **repos**. Regarde la **zone centrale encerclée** "
          "(cortex moteur C3/Cz/C4) : l'alpha y est **élevé** (rouge) car les "
          "neurones moteurs sont au repos — pas d'intention de mouvement.",
    "T1": "Le sujet **bouge réellement** sa main. La **zone centrale encerclée** "
          "(C3/Cz/C4) devient **bleue** : l'alpha chute car le cortex moteur "
          "s'active — c'est l'**ERD** (Event-Related Desynchronization).",
    "T2": "Le sujet **imagine** bouger sa main, sans mouvement réel. Même "
          "signature : la **zone centrale encerclée** vire au **bleu** (ERD). "
          "La pensée seule suffit à désynchroniser le cortex moteur.",
}

PLOTLY_BASE = dict(
    paper_bgcolor=PAPER, plot_bgcolor=PAPER,
    font=dict(color=TX, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=48, r=24, t=44, b=44),
    hoverlabel=dict(bgcolor=PAPER, bordercolor=BORD, font=dict(color=INK)),
    showlegend=False,
    transition=dict(duration=300, easing="cubic-in-out"),
)


# ── Positions 10-20 ──────────────────────────────────────────────────────────
def _load_positions() -> dict[str, tuple[float, float]]:
    montage = mne.channels.make_standard_montage("standard_1020")
    raw_pos = montage.get_positions()["ch_pos"]
    pos = {n: (float(p[0]), float(p[1])) for n, p in raw_pos.items()}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    mx = max(abs(min(xs)), abs(max(xs))) * 1.05
    my = max(abs(min(ys)), abs(max(ys))) * 1.05
    return {n: (x / mx, y / my) for n, (x, y) in pos.items()}


POSITIONS = _load_positions()
HIGHLIGHT_CHANNELS = {"C3", "Cz", "C4"}  # mises en évidence


# ── Banque d'epochs (chargée au démarrage) ───────────────────────────────────
def _alpha_at(epoch: pd.DataFrame, channel: str) -> float:
    """Puissance alpha (8-13 Hz) d'un canal pour un epoch donné."""
    col = next((c for c in epoch.columns
                if c.replace(".", "") == channel.replace(".", "")), None)
    if col is None:
        return 0.0
    sig = epoch[col].values
    n = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0 / EEG_FS)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    mask = (freqs >= 8) & (freqs < 13)
    return float(np.mean(spec[mask])) if mask.any() else 0.0


# Sujets disponibles + lesquels sont dans le train du sklearn (= 5 premiers)
SUBJECTS = sorted({p.name.split("_")[0] for p in PARQUET_DIR.glob("*.parquet")})
TRAIN_SUBJECTS = set(SUBJECTS[:5])    # les 60 premiers fichiers ≈ S001-S005
# Défaut = S001 (train) → les 3 classes sont bien prédites pour les démos boutons.
# Pour montrer la vraie performance, bascule sur un sujet hors-train (S010+).
DEFAULT_SUBJECT = "S001" if "S001" in SUBJECTS else SUBJECTS[0]
log.info("%d sujets disponibles · train=%s · défaut=%s",
         len(SUBJECTS), sorted(TRAIN_SUBJECTS), DEFAULT_SUBJECT)

EPOCH_COUNTER = {"value": 0}
_bank_cache: dict[str, dict] = {}
# Métriques live : on note l'heure d'envoi de chaque epoch pour mesurer la latence
_send_times: dict[int, float] = {}
_metrics = {"latency": None, "throughput": None, "batch_kb": None,
            "session_start": None}


def get_epoch_bank(subject: str) -> dict[str, list[pd.DataFrame]]:
    """Banque d'epochs (top 20 par classe) pour un sujet donné, avec cache."""
    if subject in _bank_cache:
        return _bank_cache[subject]
    scored: dict[str, list[tuple[float, pd.DataFrame]]] = {"T0": [], "T1": [], "T2": []}
    for p in sorted(PARQUET_DIR.glob(f"{subject}_*.parquet")):
        df = pd.read_parquet(p)
        n = len(df) // EPOCH_SAMPLES
        for i in range(n):
            chunk = df.iloc[i * EPOCH_SAMPLES:(i + 1) * EPOCH_SAMPLES].copy()
            label = chunk["task_label"].mode().iloc[0]
            if label not in scored:
                continue
            chunk["epoch_label"] = label
            motor = (_alpha_at(chunk, "C3") + _alpha_at(chunk, "C4")
                     + _alpha_at(chunk, "Cz"))
            score = motor if label == "T0" else -motor
            scored[label].append((score, chunk))
    top: dict[str, list[pd.DataFrame]] = {}
    for label, lst in scored.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        top[label] = [e for _, e in lst[:20]]
    _bank_cache[subject] = top
    log.info("Bank %s : T0=%d T1=%d T2=%d",
             subject, len(top["T0"]), len(top["T1"]), len(top["T2"]))
    return top


def _load_epoch_bank_for_baseline(parquet_dir: Path, max_files: int = 25) -> dict:
    """Charge un échantillon d'epochs (toutes classes) juste pour la baseline Z-score."""
    full: dict[str, list[pd.DataFrame]] = {"T0": [], "T1": [], "T2": []}
    for p in sorted(parquet_dir.glob("*.parquet"))[:max_files]:
        df = pd.read_parquet(p)
        n = len(df) // EPOCH_SAMPLES
        for i in range(n):
            chunk = df.iloc[i * EPOCH_SAMPLES:(i + 1) * EPOCH_SAMPLES].copy()
            label = chunk["task_label"].mode().iloc[0]
            if label in full:
                full[label].append(chunk)
    return full


def _compute_baseline_stats(bank: dict[str, list[pd.DataFrame]]) -> tuple[dict, dict]:
    """Pour chaque canal, calcule mean et std du log10(alpha power) sur
    *toutes les classes confondues* (T0 + T1 + T2).

    → Un epoch T0 (repos, alpha élevé) aura un Z > 0 sur la plupart des
      électrodes → ROUGE.
    → Un epoch T1/T2 (motor, ERD) aura un Z < 0 sur le motor cortex → BLEU.
    """
    powers: dict[str, list[float]] = {}
    n_total = 0
    for label_epochs in bank.values():
        for epoch in label_epochs:
            n_total += 1
            eeg_cols = [c for c in epoch.columns
                        if c not in {"subject_id", "run_id", "time",
                                     "task_label", "epoch_id", "epoch_label",
                                     "event_time"}]
            for ch in eeg_cols:
                sig = epoch[ch].values
                n = len(sig)
                freqs = np.fft.rfftfreq(n, 1.0 / EEG_FS)
                spec = np.abs(np.fft.rfft(sig)) ** 2
                mask = (freqs >= 8) & (freqs < 13)
                power = float(np.mean(spec[mask])) if mask.any() else 1e-12
                powers.setdefault(ch.replace(".", ""), []).append(
                    np.log10(power + 1e-12)
                )

    mean = {ch: float(np.mean(vals)) for ch, vals in powers.items()}
    std = {ch: float(np.std(vals)) + 1e-6 for ch, vals in powers.items()}
    log.info("Baseline stats globale (T0+T1+T2) sur %d canaux (n=%d epochs)",
             len(mean), n_total)
    return mean, std


BASELINE_MEAN, BASELINE_STD = _compute_baseline_stats(
    _load_epoch_bank_for_baseline(PARQUET_DIR)
)


def _write_epoch(epoch: pd.DataFrame, run_seq: int = 0) -> int:
    """Écrit un epoch en stream input. Renvoie le nouveau epoch_id global."""
    STREAM_INPUT.mkdir(parents=True, exist_ok=True)
    epoch = epoch.copy()
    EPOCH_COUNTER["value"] += 1
    epoch["epoch_id"] = EPOCH_COUNTER["value"]
    epoch["event_time"] = pd.Timestamp.now(tz="UTC").strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )
    if "run_seq" not in epoch.columns:
        epoch["run_seq"] = run_seq

    eeg_cols = [c for c in epoch.columns
                if c not in {"subject_id", "run_id", "time", "task_label",
                             "epoch_id", "epoch_label", "event_time", "run_seq"}]
    epoch[eeg_cols] = epoch[eeg_cols].astype(np.float64)
    epoch["time"] = epoch["time"].astype(np.float64)

    fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.parquet"
    path = STREAM_INPUT / fname
    epoch.to_parquet(path, index=False)
    # métriques : noter l'heure d'envoi + la taille du paquet
    _send_times[EPOCH_COUNTER["value"]] = time.time()
    if _metrics["session_start"] is None:
        _metrics["session_start"] = time.time()
    try:
        _metrics["batch_kb"] = path.stat().st_size / 1024
    except OSError:
        pass
    return EPOCH_COUNTER["value"]


def _send_epoch(label: str, subject: str) -> str:
    """Pioche un epoch du label demandé (sujet donné) et le dépose dans STREAM_INPUT."""
    epochs = get_epoch_bank(subject).get(label, [])
    if not epochs:
        return f"Aucun epoch {label} pour {subject}"
    epoch = random.choice(epochs)
    eid = _write_epoch(epoch, run_seq=0)
    log.info("Sent epoch #%d (%s, label=%s)", eid, subject, label)
    return f"Envoyé : {CLASS_LABELS[label]} · {subject} (epoch #{eid})"




# ── Chargement résilient ─────────────────────────────────────────────────────
def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def load_predictions() -> pd.DataFrame:
    df = _safe_read(PRED_DIR)
    if df.empty:
        return df
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").drop_duplicates("epoch_id", keep="last")
    return df.tail(100).reset_index(drop=True)


def load_topomap() -> tuple[pd.DataFrame, int | None]:
    """Renvoie (df, epoch_id) du DERNIER epoch arrivé. epoch_id sert d'uirevision
    pour forcer Plotly à redessiner la figure même si seules les couleurs changent.
    """
    df = _safe_read(TOPO_DIR)
    if df.empty:
        return df, None
    df["event_time"] = pd.to_datetime(df["event_time"])
    # le DERNIER en event_time, pas le epoch_id max (qui peut être stale si checkpoint reset)
    last_time = df["event_time"].max()
    last_df = df[df["event_time"] == last_time].copy()
    last_epoch_id = int(last_df["epoch_id"].iloc[0])
    return last_df, last_epoch_id


# ── Topomap richement annoté ─────────────────────────────────────────────────
def _add_head_outline(fig: go.Figure) -> None:
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=1.05 * np.cos(theta), y=1.05 * np.sin(theta),
        mode="lines", line=dict(color=MUTED, width=2),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[-0.08, 0, 0.08, -0.08], y=[1.02, 1.18, 1.02, 1.02],
        mode="lines", line=dict(color=MUTED, width=2),
        hoverinfo="skip", fill="toself", fillcolor="rgba(0,0,0,0)",
    ))
    for sign in [-1, 1]:
        fig.add_trace(go.Scatter(
            x=sign * (1.05 + 0.06 * np.cos(theta)), y=0.12 * np.sin(theta),
            mode="lines", line=dict(color=MUTED, width=2),
            hoverinfo="skip",
        ))
    # Annotations textuelles : zones et orientation
    fig.add_annotation(x=0, y=1.30, text="FRONT", showarrow=False,
                       font=dict(color=MUTED, size=10, family="monospace"))
    fig.add_annotation(x=0, y=-1.20, text="ARRIÈRE", showarrow=False,
                       font=dict(color=MUTED, size=10, family="monospace"))
    fig.add_annotation(x=-1.20, y=0, text="GAUCHE", showarrow=False,
                       font=dict(color=MUTED, size=10, family="monospace"),
                       textangle=-90)
    fig.add_annotation(x=1.20, y=0, text="DROITE", showarrow=False,
                       font=dict(color=MUTED, size=10, family="monospace"),
                       textangle=90)


_GRID_RES = 70                     # résolution du heatmap interpolé
_GRID = np.linspace(-1.1, 1.1, _GRID_RES)
_GRID_X, _GRID_Y = np.meshgrid(_GRID, _GRID)
_CIRCLE_MASK = (_GRID_X**2 + _GRID_Y**2) <= 1.0   # n'affiche que l'intérieur du crâne


def topomap_figure(
    df_topo: pd.DataFrame, pred_label: str | None, epoch_id: int | None = None,
) -> go.Figure:
    fig = go.Figure()
    if df_topo.empty:
        fig.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Topographie cérébrale · clique un bouton pour commencer",
                       font=dict(size=14, color=INK,
                                 family="Fraunces, Georgia, serif"), x=0.02),
            height=500, uirevision="topomap_empty",
            xaxis=dict(visible=False, range=[-1.35, 1.35]),
            yaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="x"),
        )
        _add_head_outline(fig)
        return fig

    df = df_topo.copy()
    df["x"] = df["channel"].map(lambda c: POSITIONS.get(c, (np.nan, np.nan))[0])
    df["y"] = df["channel"].map(lambda c: POSITIONS.get(c, (np.nan, np.nan))[1])
    df = df.dropna(subset=["x", "y"])

    # Z-score par canal contre la baseline T0 → indépendant des canaux marginaux
    p_log = np.log10(df["alpha_power"].values + 1e-12)
    means = df["channel"].map(BASELINE_MEAN).fillna(p_log.mean()).values
    stds = df["channel"].map(BASELINE_STD).fillna(1.0).values
    z = (p_log - means) / stds
    z_clip = np.clip(z, -2.5, 2.5)
    df["z"] = z_clip
    df["is_highlight"] = df["channel"].isin(HIGHLIGHT_CHANNELS)

    # ── 1. Interpolation continue (heatmap topographique) ───────────────
    try:
        grid_z = griddata(
            points=df[["x", "y"]].values, values=df["z"].values,
            xi=(_GRID_X, _GRID_Y), method="cubic", fill_value=0.0,
        )
        # masque circulaire : on ne montre que l'intérieur du crâne
        grid_z = np.where(_CIRCLE_MASK, grid_z, np.nan)

        fig.add_trace(go.Heatmap(
            x=_GRID, y=_GRID, z=grid_z,
            colorscale="RdBu_r", zmin=-2.5, zmax=2.5,
            showscale=True,
            colorbar=dict(
                title=dict(text="α power<br>(Z-score)",
                           font=dict(size=10, color=MUTED)),
                tickfont=dict(color=MUTED, size=10),
                tickvals=[-2.5, 0, 2.5],
                ticktext=["BAS<br>activité (ERD)", "MOYENNE", "HAUT<br>repos"],
                len=0.7, thickness=14, x=1.05,
            ),
            hoverinfo="skip",
            zsmooth="best",
        ))
    except Exception as exc:
        log.warning("Interpolation topomap failed: %s", exc)

    # ── 2. Cercles de mise en évidence sur le motor cortex ──────────────
    fig.add_trace(go.Scatter(
        x=df[df["is_highlight"]]["x"], y=df[df["is_highlight"]]["y"],
        mode="markers",
        marker=dict(size=42, color="rgba(255,255,255,0.0)",
                    line=dict(color=INK, width=2.5)),
        hoverinfo="skip",
    ))

    # ── 3. Markers + labels des électrodes ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"], mode="markers+text",
        marker=dict(
            size=df["is_highlight"].map(lambda x: 16 if x else 9),
            color=INK,
            line=dict(color=PAPER, width=1.5),
            opacity=0.7,
        ),
        text=df["channel"].map(
            lambda c: f"<b>{c}</b>" if c in HIGHLIGHT_CHANNELS else c
        ),
        textfont=dict(
            size=df["is_highlight"].map(lambda x: 11 if x else 6),
            color=INK,
        ),
        textposition=df["is_highlight"].map(
            lambda x: "top center" if x else "middle center"
        ),
        hovertemplate="<b>%{text}</b><br>z = %{customdata:.2f}<extra></extra>",
        customdata=df["z"],
    ))

    _add_head_outline(fig)

    title_text = "Topographie cérébrale · α power (8-13 Hz) — Z-score vs moyenne globale"
    if pred_label:
        title_text += f"<br><span style='font-size:12px;color:{CLASS_COLORS.get(pred_label, INK)}'>prédiction : <b>{CLASS_LABELS.get(pred_label, pred_label)}</b></span>"

    # uirevision dépend de l'epoch → force le redraw quand un nouvel epoch arrive
    uirev = f"topomap_{epoch_id}" if epoch_id is not None else "topomap"

    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=title_text,
                   font=dict(size=14, color=INK,
                             family="Fraunces, Georgia, serif"), x=0.02),
        height=500,
        xaxis=dict(visible=False, range=[-1.35, 1.35]),
        yaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="x"),
        uirevision=uirev,
    )
    return fig


def prediction_bars(df_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_pred.empty:
        fig.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Probabilités · clique un bouton pour commencer",
                       font=dict(size=13, color=INK,
                                 family="Fraunces, Georgia, serif"), x=0.02),
            height=440, uirevision="bars",
        )
        return fig

    last = df_pred.iloc[-1]
    probas = [last.get(f"proba_{k}", 0) for k in ["T0", "T1", "T2"]]
    pred = last.get("pred_label", "?")
    true = last.get("true_label", "?")

    fig.add_trace(go.Bar(
        x=[CLASS_LABELS[k] for k in ["T0", "T1", "T2"]],
        y=probas,
        marker=dict(color=[CLASS_COLORS[k] for k in ["T0", "T1", "T2"]],
                    line=dict(color=PAPER, width=2)),
        text=[f"{p*100:.0f}%" for p in probas],
        textposition="outside",
        textfont=dict(color=INK, size=13, family="Fraunces, Georgia, serif"),
        hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
    ))
    correct_mark = " ✓" if pred == true else " ✗"
    title = f"Prédiction : {CLASS_LABELS.get(pred, pred)}{correct_mark}"
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=title, font=dict(size=14, color=INK,
                                          family="Fraunces, Georgia, serif"),
                   x=0.02),
        yaxis=dict(range=[0, 1.15], gridcolor=GRID, tickformat=".0%",
                   tickfont=dict(color=MUTED)),
        xaxis=dict(tickfont=dict(color=MUTED, size=11)),
        height=440, uirevision="bars",
    )
    return fig


def timeline_figure(df_pred: pd.DataFrame) -> go.Figure:
    """Moniteur temps réel : chaque epoch = une barre colorée placée dans le
    temps (event_time). Style moniteur hôpital — on voit le flux défiler.
    """
    fig = go.Figure()
    if df_pred.empty or "event_time" not in df_pred.columns:
        fig.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Moniteur temps réel · en attente du flux…",
                       font=dict(size=13, color=INK,
                                 family="Fraunces, Georgia, serif"), x=0.02),
            height=170, uirevision="timeline",
        )
        return fig

    df = df_pred.tail(40).copy()
    for cls in ["T0", "T1", "T2"]:
        sub = df[df["pred_label"] == cls]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["event_time"], y=[1] * len(sub),
            marker=dict(color=CLASS_COLORS[cls],
                        line=dict(color=PAPER, width=1)),
            name=CLASS_LABELS[cls], width=1500,  # ms
            hovertemplate=f"<b>{CLASS_LABELS[cls]}</b><br>%{{x|%H:%M:%S}}<extra></extra>",
        ))

    layout = {k: v for k, v in PLOTLY_BASE.items()
              if k not in {"showlegend", "margin"}}
    fig.update_layout(
        **layout,
        showlegend=True,
        legend=dict(orientation="h", y=-0.35, x=0, font=dict(size=10, color=TX)),
        margin=dict(l=24, r=24, t=46, b=58),
        title=dict(
            text="Moniteur temps réel · flux des prédictions (chaque barre = 1 epoch)",
            font=dict(size=13, color=INK, family="Fraunces, Georgia, serif"),
            x=0.02, y=0.95,
        ),
        barmode="overlay",
        yaxis=dict(visible=False, range=[0, 1.2]),
        xaxis=dict(gridcolor=GRID, tickfont=dict(color=MUTED)),
        height=200, uirevision="timeline",
    )
    return fig


# ── Cards & UI ───────────────────────────────────────────────────────────────
CARD_STYLE = {
    "padding": "18px 22px", "background": PAPER,
    "border": f"1px solid {BORD}", "borderRadius": "18px",
    "boxShadow": "0 1px 3px rgba(60,50,40,0.04), 0 2px 12px rgba(60,50,40,0.03)",
    "flex": "1", "minWidth": "160px",
}


def big_button(label: str, icon: str, color: str, btn_id: str,
               description: str) -> html.Button:
    return html.Button(
        children=[
            html.Div(icon, style={"fontSize": "30px", "marginBottom": "6px"}),
            html.Div(label, style={
                "fontSize": "15px", "fontWeight": "600",
                "fontFamily": "Fraunces, Georgia, serif",
            }),
            html.Div(description, style={
                "fontSize": "10.5px", "color": MUTED, "marginTop": "4px",
                "lineHeight": "1.3",
            }),
        ],
        id=btn_id, n_clicks=0,
        style={
            "padding": "18px 16px", "background": PAPER,
            "border": f"2px solid {color}", "borderRadius": "16px",
            "color": color, "cursor": "pointer", "minWidth": "180px",
            "flex": "1", "transition": "all 200ms cubic-bezier(0.4, 0, 0.2, 1)",
        },
    )


def update_live_metrics(df_pred: pd.DataFrame) -> None:
    """Met à jour les métriques globales : latence vraie (mtime du fichier
    Spark - send time) + débit observé."""
    if df_pred.empty or "epoch_id" not in df_pred.columns:
        return
    now = time.time()

    # Latence : on prend le mtime du dernier fichier prédiction écrit par Spark
    # = moment où Spark a vraiment fini ce batch
    if PRED_DIR.exists():
        files = sorted(PRED_DIR.glob("*.parquet"),
                       key=lambda f: f.stat().st_mtime)
        if files:
            latest_mtime = files[-1].stat().st_mtime
            max_eid = int(df_pred["epoch_id"].max())
            send_t = _send_times.get(max_eid)
            if send_t is not None:
                lat = latest_mtime - send_t
                if 0 < lat < 30:   # filtre les valeurs aberrantes
                    prev = _metrics["latency"]
                    _metrics["latency"] = (
                        lat if prev is None else 0.4 * prev + 0.6 * lat
                    )
    # purge les send_times anciens pour ne pas accumuler de la mémoire
    for eid in list(_send_times.keys()):
        if now - _send_times[eid] > 60:
            _send_times.pop(eid, None)

    # Débit : nombre d'epochs / temps de session
    start = _metrics["session_start"]
    if start is not None:
        elapsed = max(now - start, 1.0)
        _metrics["throughput"] = len(df_pred) / elapsed


def metrics_panel() -> html.Div:
    """Affiche les 4 métriques clés du streaming, façon dashboard ops."""
    def chip(icon: str, label: str, value: str, color: str = INK) -> html.Div:
        return html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "15px", "marginRight": "6px"}),
                html.Span(label, style={
                    "color": MUTED, "fontSize": "10px",
                    "letterSpacing": "0.08em", "textTransform": "uppercase",
                    "fontWeight": "600",
                }),
            ], style={"marginBottom": "4px", "display": "flex", "alignItems": "center"}),
            html.Div(value, style={
                "color": color, "fontSize": "20px", "fontWeight": "600",
                "fontFamily": "Fraunces, Georgia, serif", "lineHeight": "1",
            }),
        ], style={
            "padding": "10px 14px", "background": PAPER,
            "border": f"1px solid {BORD}", "borderRadius": "12px",
            "flex": "1", "minWidth": "130px",
        })

    lat = _metrics["latency"]
    thr = _metrics["throughput"]
    batch = _metrics["batch_kb"]
    return html.Div([
        chip("⏱️", "Latence",
             f"{lat:.2f} s" if lat is not None else "— s",
             color=COLOR_T2),
        chip("🚀", "Débit",
             f"{thr:.2f} epoch/s" if thr is not None else "— epoch/s",
             color=COLOR_T1),
        chip("📦", "Taille paquet",
             f"{batch:.0f} Ko" if batch is not None else "— Ko"),
        chip("🔢", "Signaux traités",
             f"{int(EPOCH_COUNTER['value'])}"),
    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"})


def hint_panel(last_label: str | None) -> html.Div:
    if last_label is None:
        return html.Div([
            html.Div("👈  Choisis un epoch à envoyer", style={
                "fontSize": "14px", "color": INK, "fontWeight": "500",
                "marginBottom": "6px",
            }),
            html.Div(
                "Le dashboard est en mode interactif. Clique sur un des trois "
                "boutons ci-dessus pour envoyer un epoch de ce type au pipeline "
                "Spark. Tu verras la prédiction et la topographie cérébrale.",
                style={"color": MUTED, "fontSize": "12px", "lineHeight": "1.5"},
            ),
        ], style={**CARD_STYLE, "padding": "16px"})

    hint = CLASS_HINTS[last_label]
    color = CLASS_COLORS[last_label]
    return html.Div([
        html.Div([
            html.Span("Ce que tu dois voir : ", style={
                "color": MUTED, "fontSize": "11px",
                "letterSpacing": "0.08em", "textTransform": "uppercase",
                "fontWeight": "500",
            }),
            html.Span(CLASS_LABELS[last_label], style={
                "color": color, "fontSize": "13px", "fontWeight": "600",
                "marginLeft": "8px",
            }),
        ], style={"marginBottom": "8px"}),
        html.Div(dcc.Markdown(hint, link_target="_blank"),
                 style={"color": INK, "fontSize": "13px", "lineHeight": "1.5"}),
    ], style={**CARD_STYLE, "padding": "16px",
              "borderLeft": f"4px solid {color}"})


# ── App Dash ─────────────────────────────────────────────────────────────────
EXTERNAL_FONTS = [
    "https://fonts.googleapis.com/css2?family=Fraunces:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap",
]
app = Dash(__name__, title="NeuroSpark · EEG BCI Live",
           external_stylesheets=EXTERNAL_FONTS, update_title=None)

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div(style={
                "width": "10px", "height": "10px", "borderRadius": "50%",
                "background": COLOR_T2,
                "boxShadow": "0 0 0 4px rgba(196,69,54,0.20)",
                "marginRight": "10px",
                "animation": "pulse 1.6s ease-in-out infinite",
            }),
            html.Span("LIVE BCI · Mode interactif", style={
                "color": COLOR_T2, "fontSize": "11px", "fontWeight": "700",
                "letterSpacing": "0.18em",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.H1("NeuroSpark · Brain-Computer Interface", style={
            "margin": "8px 0 6px 0", "fontFamily": "Fraunces, Georgia, serif",
            "color": INK, "fontSize": "30px", "fontWeight": "600",
        }),
        html.Div(
            "Clique sur un bouton pour envoyer un epoch EEG dans Spark. "
            "Le modèle prédira la classe et la topographie cérébrale "
            "montrera l'activité alpha sur les 64 électrodes.",
            style={"color": TX, "fontSize": "13px", "maxWidth": "820px",
                   "lineHeight": "1.5"},
        ),
        # Sélecteur de sujet
        html.Div([
            html.Span("Sujet testé : ", style={
                "color": MUTED, "fontSize": "12px", "fontWeight": "600",
                "marginRight": "8px",
            }),
            dcc.Dropdown(
                id="subject-select",
                options=[
                    {"label": f"{s}  {'(train ⚠)' if s in TRAIN_SUBJECTS else '(hors-train ✓)'}",
                     "value": s}
                    for s in SUBJECTS
                ],
                value=DEFAULT_SUBJECT,
                clearable=False,
                style={"width": "240px", "fontSize": "13px"},
            ),
            html.Span(
                "  ← un sujet HORS-train donne de vraies variations (les sujets "
                "train = 100 % par data leakage)",
                style={"color": MUTED, "fontSize": "11px", "marginLeft": "10px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "12px"}),
    ], style={"padding": "28px 32px 12px 32px"}),

    # Boutons interactifs
    html.Div([
        big_button("Repos", "🧘", COLOR_T0, "btn-T0",
                   "Le sujet ne fait rien. Alpha élevé attendu."),
        big_button("Mouvement réel", "✋", COLOR_T1, "btn-T1",
                   "Le sujet bouge sa main. Alpha bas sur C3/C4."),
        big_button("Mouvement imaginé", "🧠", COLOR_T2, "btn-T2",
                   "Le sujet imagine bouger. ERD attendu sur cortex moteur."),
    ], style={"display": "flex", "gap": "12px",
              "padding": "12px 32px 0 32px"}),

    # Panneau métriques live (latence / débit / taille / count)
    html.Div(id="metrics-panel", style={"padding": "16px 32px 0 32px"}),

    # Animation pipeline (s'allume étape par étape quand un epoch est envoyé)
    html.Div(id="pipeline-anim", style={"padding": "16px 32px 0 32px"}),

    # Toggle flux continu + statut
    html.Div([
        html.Div([
            dcc.Checklist(
                id="auto-toggle",
                options=[{"label": "  ▶ Démarrer le flux continu (1 epoch / 3 s, simule un sujet réel)",
                          "value": "auto"}],
                value=[],
                inputStyle={"marginRight": "6px"},
                style={"color": INK, "fontSize": "13px", "fontWeight": "500"},
            ),
            html.Div(
                "Active ce mode pour alimenter le moniteur temps réel et la "
                "fenêtre glissante Spark ci-dessous — c'est le vrai streaming continu.",
                style={"color": MUTED, "fontSize": "11.5px", "marginTop": "2px",
                       "marginLeft": "22px"},
            ),
        ], style={"marginBottom": "10px"}),

        html.Div(id="status", children="", style={
            "color": MUTED, "fontSize": "12px", "marginTop": "12px",
            "fontFamily": "monospace",
        }),
    ], style={"padding": "16px 32px 0 32px"}),

    html.Div(id="hint-panel", style={"padding": "16px 32px 0 32px"}),

    # Topomap + Bars/Accuracy
    html.Div([
        html.Div([
            dcc.Graph(id="topomap",
                      config={"displayModeBar": False},
                      style={"height": "500px"}),
        ], style={**CARD_STYLE, "flex": 2, "padding": "12px"}),
        html.Div([
            dcc.Graph(id="pred-bars",
                      config={"displayModeBar": False},
                      style={"height": "440px"}),
        ], style={**CARD_STYLE, "flex": 1, "padding": "12px"}),
    ], style={"display": "flex", "gap": "12px",
              "padding": "20px 32px 0 32px"}),

    # Moniteur temps réel (timeline du flux des prédictions)
    html.Div([
        html.Div([
            dcc.Graph(id="timeline",
                      config={"displayModeBar": False},
                      style={"height": "200px"}),
        ], style={**CARD_STYLE, "padding": "12px"}),
    ], style={"padding": "12px 32px 32px 32px"}),

    dcc.Store(id="last-label", data=None),
    dcc.Store(id="pipeline-start", data=None),  # timestamp ms du dernier clic
    dcc.Interval(id="tick", interval=REFRESH_MS, n_intervals=0),
    dcc.Interval(id="auto-tick", interval=3000, n_intervals=0),
    dcc.Interval(id="anim-tick", interval=300, n_intervals=0),

    html.Footer([
        html.Span("NeuroSpark BCI", style={"fontWeight": 600}),
        html.Span(" · UE28 Big Data · 2026", style={"color": MUTED}),
    ], style={"padding": "14px 32px 30px 32px", "color": TX,
              "fontSize": "11px", "textAlign": "center",
              "fontFamily": "monospace"}),
], style={"background": BG, "minHeight": "100vh",
          "fontFamily": "Inter, system-ui, sans-serif", "color": INK})


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50%      { transform: scale(1.15); opacity: 0.85; }
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(60,50,40,0.10);
            }
            button:active { transform: translateY(0); }
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #f5f1e8; }
            ::-webkit-scrollbar-thumb { background: #c9bfae; border-radius: 4px; }
            body { margin: 0; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


AUTO_ROTATION = ["T0", "T1", "T2", "T0", "T2", "T1"]   # séquence cyclique

# Animation pipeline : 5 étapes, chacune dure ~500 ms
PIPELINE_STEPS = [
    ("📥", "Epoch envoyé", "Dashboard écrit un Parquet dans data/stream_eeg/input/"),
    ("🔍", "Spark detect", "readStream voit le nouveau fichier (file source)"),
    ("📊", "FFT extraction", "Calcul des band powers theta/alpha/beta/gamma sur C3/Cz/C4"),
    ("🤖", "ML predict", "sklearn RandomForest classifie : T0 / T1 / T2"),
    ("💾", "Sink Parquet", "Résultat écrit + dashboard rafraîchit en 1.5 s"),
]
STEP_DURATION_MS = 500   # chaque étape dure 500 ms (5 étapes = 2.5 s total)


def pipeline_animation(start_ts: int | None) -> html.Div:
    """5 boîtes alignées qui s'allument une après l'autre après un clic."""
    if start_ts is None:
        elapsed_ms = -1
    else:
        elapsed_ms = int(time.time() * 1000) - start_ts

    current_step = elapsed_ms // STEP_DURATION_MS if elapsed_ms >= 0 else -1
    is_idle = current_step >= len(PIPELINE_STEPS) or current_step < 0

    boxes = []
    for i, (icon, title, desc) in enumerate(PIPELINE_STEPS):
        if is_idle:
            bg, border, color = PAPER, BORD, MUTED
            icon_opacity = 0.4
        elif i < current_step:
            bg, border, color = "rgba(122,155,118,0.10)", "#7a9b76", "#5a7556"
            icon_opacity = 1.0
        elif i == current_step:
            bg, border, color = "rgba(217,164,65,0.18)", "#d9a441", "#a87520"
            icon_opacity = 1.0
        else:
            bg, border, color = PAPER, BORD, MUTED
            icon_opacity = 0.4

        boxes.append(html.Div([
            html.Div(icon, style={
                "fontSize": "22px", "marginBottom": "4px",
                "opacity": icon_opacity,
                "transition": "all 300ms cubic-bezier(0.4, 0, 0.2, 1)",
            }),
            html.Div(title, style={
                "color": color, "fontSize": "11px", "fontWeight": "600",
                "fontFamily": "Fraunces, Georgia, serif",
            }),
            html.Div(desc, style={
                "color": MUTED, "fontSize": "9.5px", "marginTop": "2px",
                "lineHeight": "1.3", "minHeight": "26px",
            }),
        ], style={
            "padding": "10px 12px", "background": bg,
            "border": f"1.5px solid {border}", "borderRadius": "12px",
            "flex": "1", "textAlign": "center",
            "transition": "all 300ms cubic-bezier(0.4, 0, 0.2, 1)",
        }))

        # connecteur entre étapes (sauf après la dernière)
        if i < len(PIPELINE_STEPS) - 1:
            arrow_color = "#7a9b76" if i < current_step and not is_idle else BORD
            boxes.append(html.Div("→", style={
                "color": arrow_color, "fontSize": "16px",
                "alignSelf": "center", "padding": "0 4px",
                "transition": "all 300ms",
            }))

    label_strip = html.Div([
        html.Span("PIPELINE STREAMING ", style={
            "color": MUTED, "fontSize": "10px", "fontWeight": "600",
            "letterSpacing": "0.12em",
        }),
        html.Span(
            "· idle" if is_idle else f"· étape {current_step + 1}/{len(PIPELINE_STEPS)}",
            style={"color": COLOR_T2 if not is_idle else MUTED, "fontSize": "10px",
                   "fontFamily": "monospace"},
        ),
    ], style={"marginBottom": "8px"})

    return html.Div([
        label_strip,
        html.Div(boxes, style={"display": "flex", "alignItems": "stretch"}),
    ])


# ── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output("status", "children"),
    Output("last-label", "data"),
    Output("pipeline-start", "data"),
    Input("btn-T0", "n_clicks"),
    Input("btn-T1", "n_clicks"),
    Input("btn-T2", "n_clicks"),
    Input("auto-tick", "n_intervals"),
    State("auto-toggle", "value"),
    State("last-label", "data"),
    State("subject-select", "value"),
    prevent_initial_call=True,
)
def on_trigger(_n0, _n1, _n2, auto_n, auto_value, current, subject):
    ctx = callback_context
    subject = subject or DEFAULT_SUBJECT
    if not ctx.triggered:
        return "", current, None
    trig = ctx.triggered[0]["prop_id"].split(".")[0]

    # Flux continu (mode auto)
    if trig == "auto-tick":
        if auto_value and "auto" in auto_value:
            label = AUTO_ROTATION[auto_n % len(AUTO_ROTATION)]
            msg = _send_epoch(label, subject)
            return f"[FLUX] {msg}", label, int(time.time() * 1000)
        return "", current, None

    # Bouton manuel T0/T1/T2
    label = trig.replace("btn-", "")
    msg = _send_epoch(label, subject)
    return f"→ {msg} · suis le pipeline ci-dessous", label, int(time.time() * 1000)


@app.callback(
    Output("pipeline-anim", "children"),
    Input("anim-tick", "n_intervals"),
    State("pipeline-start", "data"),
)
def refresh_pipeline_anim(_n, start_ts):
    return pipeline_animation(start_ts)


@app.callback(
    Output("status", "children", allow_duplicate=True),
    Output("last-label", "data", allow_duplicate=True),
    Input("subject-select", "value"),
    prevent_initial_call=True,
)
def on_subject_change(subject):
    """Au changement de sujet : efface les sorties affichées et remet
    le compteur + les métriques à zéro. Page blanche.
    """
    for d in (PRED_DIR, TOPO_DIR):
        if d.exists():
            for f in d.glob("*.parquet"):
                try:
                    f.unlink()
                except OSError:
                    pass
    EPOCH_COUNTER["value"] = 0
    _send_times.clear()
    _metrics.update({"latency": None, "throughput": None,
                     "batch_kb": None, "session_start": None})
    log.info("Sujet → %s : reset complet", subject)
    return (
        f"→ Sujet changé : {subject}. Page réinitialisée, prêt pour un nouveau test.",
        None,
    )


@app.callback(
    Output("topomap", "figure"),
    Output("pred-bars", "figure"),
    Output("timeline", "figure"),
    Output("hint-panel", "children"),
    Output("metrics-panel", "children"),
    Input("tick", "n_intervals"),
    State("last-label", "data"),
)
def refresh(_n, last_label):
    pred = load_predictions()
    topo, topo_eid = load_topomap()
    update_live_metrics(pred)
    last_pred = pred["pred_label"].iloc[-1] if not pred.empty else None
    hint_label = None
    if not topo.empty and "true_label" in topo.columns:
        hint_label = topo["true_label"].iloc[0]
    elif last_label:
        hint_label = last_label
    return (
        topomap_figure(topo, last_pred, epoch_id=topo_eid),
        prediction_bars(pred),
        timeline_figure(pred),
        hint_panel(hint_label),
        metrics_panel(),
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8052)
