"""Dashboard Streamlit — outil décisionnel de rétention client (dark mode).

Le dashboard consomme l'API FastAPI (il ne charge jamais le modèle directement).
Design « Signal » sombre : le risque du client est le point focal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from churn.config import load_config, resolve  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration & jetons de design (palette sombre)
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Rétention Client", page_icon="📉", layout="wide")

CFG = load_config()
API_URL = CFG["api"]["url"]
TARGET = CFG["data"]["target"]
ID_COL = CFG["data"]["id_column"]

BG, SURFACE, BORDER = "#0B0E14", "#151A23", "#262E3D"
INK, MUTED, PRIMARY = "#E7EAF0", "#8B93A7", "#818CF8"
GREEN, AMBER, RED = "#34D399", "#FBBF24", "#FB7185"
RISK_COLOR = {"faible": GREEN, "modéré": AMBER, "élevé": RED}
RISK_BG = {"faible": "#10241C", "modéré": "#2A2310", "élevé": "#2A1620"}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stApp, [class*="css"] { font-family:'Inter',-apple-system,Segoe UI,sans-serif; }
    .stApp { background:#0B0E14; }
    #MainMenu, footer, header { visibility:hidden; }
    .block-container { padding:1.8rem 2.5rem 3rem; max-width:1600px; }
    .hdr { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.6rem; }
    .hdr h1 { font-size:1.8rem; font-weight:700; letter-spacing:-.02em; color:#E7EAF0; margin:0; }
    .hdr p { color:#8B93A7; margin:.3rem 0 0; font-size:.96rem; }
    .eyebrow { font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:#818CF8; font-weight:700; margin:.2rem 0 .8rem; }
    .kpi-row { display:flex; gap:18px; margin-bottom:.5rem; }
    .kpi { flex:1; background:#151A23; border:1px solid #262E3D; border-radius:16px; padding:1.2rem 1.4rem; }
    .kpi .label { font-size:.72rem; color:#8B93A7; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
    .kpi .value { font-size:2.1rem; font-weight:700; color:#E7EAF0; line-height:1.1; margin-top:.35rem; }
    .chip { display:inline-flex; align-items:center; gap:.5rem; padding:.4rem .9rem; border-radius:999px; font-weight:600; font-size:.9rem; }
    .status { display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .8rem; border-radius:999px; font-size:.8rem; font-weight:600; border:1px solid #262E3D; background:#151A23; color:#8B93A7; }
    .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
    .panel { background:#151A23; border:1px solid #262E3D; border-radius:18px; padding:1.4rem 1.6rem; }
    .sig { display:flex; justify-content:space-between; align-items:center; padding:.62rem 0; border-bottom:1px solid #1F2733; }
    .sig:last-child { border-bottom:none; }
    .sig .name { color:#E7EAF0; font-weight:500; font-size:.95rem; }
    .sig .val { display:flex; align-items:center; gap:.55rem; color:#8B93A7; font-weight:600; font-size:.95rem; }
    .empty { text-align:center; color:#6B7488; padding:3.6rem 1rem; border:1px dashed #2A3344; border-radius:18px; background:#0F141D; }
    .empty .big { font-size:2.4rem; margin-bottom:.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; }
    .stTabs [data-baseweb="tab"] { font-weight:600; }
    div[data-testid="stImage"] img { border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig, height):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), font={"family": "Inter", "color": INK},
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
                     tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
                     tickfont={"color": MUTED})
    return fig


# --------------------------------------------------------------------------- #
# Données & API (mis en cache)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(resolve(CFG["data"]["path"]))


@st.cache_data
def load_model_comparison():
    p = resolve(CFG["paths"]["reports_dir"], "metrics", "model_comparison.csv")
    return pd.read_csv(p) if p.exists() else None


def default_profile(df: pd.DataFrame) -> dict:
    feats = [c for c in df.columns if c not in {TARGET, ID_COL}]
    prof = {}
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]):
            v = df[c].median()
            prof[c] = int(v) if df[c].dtype.kind == "i" else float(v)
        else:
            m = df[c].mode(dropna=True)
            prof[c] = m.iloc[0] if not m.empty else None
    return prof


def call_predict(payload: dict):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"Erreur API {r.status_code} : {r.text}")
    except requests.exceptions.RequestException:
        st.error(f"API injoignable. Lancez `make api` (endpoint : {API_URL}).")
    return None


def api_online() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json().get("model_loaded", False)
    except requests.exceptions.RequestException:
        return False


def risk_gauge(proba: float, threshold: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 52, "color": INK, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                     "tickfont": {"color": MUTED, "size": 11}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#10241C"},
                {"range": [33, 66], "color": "#2A2310"},
                {"range": [66, 100], "color": "#2A1620"},
            ],
            "threshold": {"line": {"color": INK, "width": 2}, "thickness": 0.8,
                          "value": threshold * 100},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=24, r=24, t=24, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
    return fig


def profile_signals(p: dict):
    """Lecture métier rapide du profil (heuristiques issues de l'analyse des données)."""
    s = []
    pf = p["payment_failures"]
    s.append(("Paiements ratés", str(pf), "risk" if pf >= 2 else "ok"))
    cs = p["csat_score"]
    s.append(("Satisfaction (CSAT)", f"{cs:.1f}/5", "risk" if cs <= 2 else ("ok" if cs >= 4 else "warn")))
    nps = p["nps_score"]
    s.append(("NPS", str(nps), "risk" if nps < 0 else ("ok" if nps >= 50 else "warn")))
    tn = p["tenure_months"]
    s.append(("Ancienneté", f"{tn} mois", "risk" if tn <= 6 else "ok"))
    ug = p["usage_growth_rate"]
    s.append(("Croissance d'usage", f"{ug:+.0%}", "risk" if ug < 0 else "ok"))
    ll = p["last_login_days_ago"]
    s.append(("Dernière connexion", f"il y a {ll} j", "risk" if ll >= 21 else "ok"))
    return s


SIG_DOT = {"risk": RED, "warn": AMBER, "ok": GREEN}


# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #
df = load_dataset()
online = api_online()
status_dot = GREEN if online else RED
status_txt = "API connectée" if online else "API hors ligne"

st.markdown(
    f"""
    <div class="hdr">
      <div>
        <h1>Pilotage de la rétention client</h1>
        <p>Évaluez le risque de départ d'un client, comprenez-le, et simulez une action.</p>
      </div>
      <span class="status"><span class="dot" style="background:{status_dot}"></span>{status_txt}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_score, tab_kpi, tab_models = st.tabs(["  Évaluer un client  ", "  Vue d'ensemble  ", "  Modèles  "])

# --------------------------------------------------------------------------- #
# Onglet 1 — Évaluer un client (HÉRO)
# --------------------------------------------------------------------------- #
with tab_score:
    base = default_profile(df)
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.markdown('<div class="eyebrow">Profil du client</div>', unsafe_allow_html=True)
        with st.form("scoring"):
            base["csat_score"] = st.slider("Satisfaction (CSAT)", 1.0, 5.0,
                                           float(base["csat_score"]), step=0.5)
            base["nps_score"] = st.slider("NPS", -100, 100, int(base["nps_score"]))
            c1, c2 = st.columns(2)
            base["payment_failures"] = c1.number_input("Paiements ratés", 0, 10,
                                                        int(base["payment_failures"]))
            base["support_tickets"] = c2.number_input("Tickets support", 0, 20,
                                                       int(base["support_tickets"]))
            base["tenure_months"] = st.slider("Ancienneté (mois)", 0, 72,
                                               int(base["tenure_months"]))
            base["last_login_days_ago"] = st.slider("Dernière connexion (jours)", 0, 90,
                                                     int(base["last_login_days_ago"]))
            base["usage_growth_rate"] = st.slider("Croissance d'usage", -0.6, 0.6,
                                                  float(base["usage_growth_rate"]), step=0.01)
            base["contract_type"] = st.selectbox("Type de contrat",
                                                 sorted(df["contract_type"].unique()))
            submitted = st.form_submit_button("Évaluer le risque", type="primary",
                                              use_container_width=True)
        if submitted:
            res = call_predict(base)
            if res:
                st.session_state["result"] = res
                st.session_state["profile"] = dict(base)

    with right:
        if "result" not in st.session_state:
            st.markdown(
                '<div class="empty"><div class="big">📉</div>'
                'Renseignez un profil à gauche puis lancez l\'évaluation.<br>'
                'Le risque de churn s\'affichera ici.</div>',
                unsafe_allow_html=True,
            )
        else:
            res = st.session_state["result"]
            prof = st.session_state["profile"]
            level = res["risk_level"]
            color = RISK_COLOR[level]
            verdict = "À risque de départ" if res["churn_prediction"] == 1 else "Probablement fidèle"

            st.markdown('<div class="eyebrow">Risque de churn</div>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(res["churn_probability"], res["threshold"], color),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div style="text-align:center;margin-top:-.6rem">'
                f'<span class="chip" style="background:{RISK_BG[level]};color:{color}">'
                f'<span class="dot" style="background:{color}"></span>Risque {level} · {verdict}</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="eyebrow" style="margin-top:1.5rem">Signaux du profil</div>',
                        unsafe_allow_html=True)
            rows = "".join(
                f'<div class="sig"><span class="name">{name}</span>'
                f'<span class="val">{val}<span class="dot" style="background:{SIG_DOT[lvl]}"></span></span></div>'
                for name, val, lvl in profile_signals(prof)
            )
            st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

            if st.button("⟳ Simuler une action de rétention", use_container_width=True):
                scen = dict(prof)
                scen["nps_score"] = min(100, prof["nps_score"] + 40)
                scen["csat_score"] = min(5.0, prof["csat_score"] + 1.0)
                scen["usage_growth_rate"] = round(prof["usage_growth_rate"] + 0.2, 2)
                sim = call_predict(scen)
                if sim:
                    delta = sim["churn_probability"] - res["churn_probability"]
                    st.metric("Risque après action (satisfaction + engagement)",
                              f"{sim['churn_probability']:.0%}", f"{delta:+.0%}",
                              delta_color="inverse")

    with st.expander("Importance globale des variables (modèle)"):
        perm_img = resolve(CFG["paths"]["reports_dir"], "figures", "permutation_importance.png")
        shap_img = resolve(CFG["paths"]["reports_dir"], "figures", "shap_summary.png")
        if perm_img.exists():
            st.markdown("**Quelles variables pèsent le plus** — plus la barre est longue, "
                        "plus la variable est déterminante.")
            st.image(str(perm_img))
        if shap_img.exists():
            st.markdown("**Comment chaque variable pousse le risque** — un point = un client ; "
                        "rouge = valeur élevée, bleu = faible ; à droite = augmente le churn.")
            st.image(str(shap_img))
        if not perm_img.exists() and not shap_img.exists():
            st.caption("Lancez `make train` pour générer ces graphiques.")

# --------------------------------------------------------------------------- #
# Onglet 2 — Vue d'ensemble
# --------------------------------------------------------------------------- #
with tab_kpi:
    rate = df[TARGET].mean()
    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi"><div class="label">Clients</div><div class="value">{len(df):,}</div></div>
          <div class="kpi"><div class="label">Taux de churn</div><div class="value">{rate:.1%}</div></div>
          <div class="kpi"><div class="label">Clients perdus</div><div class="value">{int(df[TARGET].sum()):,}</div></div>
        </div>
        """.replace(",", " "),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="eyebrow" style="margin-top:1.5rem">Le signal le plus fort : les échecs de paiement</div>',
                unsafe_allow_html=True)
    pf = (df.groupby("payment_failures")[TARGET].mean() * 100).reset_index()
    pf.columns = ["payment_failures", "churn_pct"]
    fig = px.bar(pf, x="payment_failures", y="churn_pct",
                 labels={"payment_failures": "Paiements ratés", "churn_pct": "Taux de churn (%)"})
    fig.update_traces(marker_color=PRIMARY)
    fig.add_hline(y=rate * 100, line_dash="dot", line_color=MUTED,
                  annotation_text=f"Moyenne {rate:.0%}", annotation_position="top left",
                  annotation_font_color=MUTED)
    style_fig(fig, 360)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Au-delà de 2 paiements ratés, le risque de départ est 2 à 3 fois la moyenne.")

    with st.expander("Explorer d'autres facteurs"):
        col = st.selectbox("Variable", ["customer_segment", "contract_type", "survey_response"])
        seg = (df.groupby(col)[TARGET].mean() * 100).reset_index()
        seg.columns = [col, "churn_pct"]
        f2 = px.bar(seg, x=col, y="churn_pct", labels={"churn_pct": "Taux de churn (%)"})
        f2.update_traces(marker_color="#6366F1")
        style_fig(f2, 340)
        st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})

# --------------------------------------------------------------------------- #
# Onglet 3 — Modèles
# --------------------------------------------------------------------------- #
with tab_models:
    st.markdown('<div class="eyebrow">Performances comparées (validation croisée)</div>',
                unsafe_allow_html=True)
    comp = load_model_comparison()
    if comp is not None:
        view = comp[["modele", "pr_auc", "roc_auc", "recall"]].copy()
        best = view["pr_auc"].idxmax()
        ordered = view.sort_values("pr_auc")
        fig = px.bar(ordered, x="pr_auc", y="modele", orientation="h",
                     labels={"pr_auc": "PR-AUC", "modele": ""})
        colors = [PRIMARY if i == best else "#3A4358" for i in ordered.index]
        fig.update_traces(marker_color=colors)
        style_fig(fig, 320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(view.round(3), use_container_width=True, hide_index=True)
        st.caption("Le Random Forest offre le meilleur PR-AUC ; le réseau de neurones (MLP) "
                   "est ici le moins performant — le Deep Learning n'est pas toujours supérieur.")
    else:
        st.caption("Lancez `make train` pour générer la comparaison des modèles.")
