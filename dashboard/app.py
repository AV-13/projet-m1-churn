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
from churn.config import load_config, resolve  


st.set_page_config(page_title="Rétention Client", page_icon="📉", layout="wide")

CFG = load_config()
API_URL = CFG["api"]["url"]
TARGET = CFG["data"]["target"]
ID_COL = CFG["data"]["id_column"]

INK, MUTED, PRIMARY = "#1A2233", "#6B7385", "#4F46E5"
GREEN, AMBER, RED, BORDER = "#16A34A", "#F59E0B", "#E11D48", "#E6E9EF"
RISK_COLOR = {"faible": GREEN, "modéré": AMBER, "élevé": RED}
RISK_BG = {"faible": "#ECFDF3", "modéré": "#FEF6E7", "élevé": "#FEECEF"}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stApp, [class*="css"] { font-family:'Inter',-apple-system,Segoe UI,sans-serif; }
    .stApp { background:#F7F8FA; }
    #MainMenu, footer, header { visibility:hidden; }
    .block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1120px; }
    .hdr { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.6rem; }
    .hdr h1 { font-size:1.7rem; font-weight:700; letter-spacing:-.02em; color:#1A2233; margin:0; }
    .hdr p { color:#6B7385; margin:.25rem 0 0; font-size:.95rem; }
    .eyebrow { font-size:.7rem; letter-spacing:.13em; text-transform:uppercase; color:#6B7385; font-weight:600; margin:.2rem 0 .7rem; }
    .kpi-row { display:flex; gap:16px; margin-bottom:.5rem; }
    .kpi { flex:1; background:#fff; border:1px solid #E6E9EF; border-radius:16px; padding:1.1rem 1.25rem; }
    .kpi .label { font-size:.72rem; color:#6B7385; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
    .kpi .value { font-size:2rem; font-weight:700; color:#1A2233; line-height:1.1; margin-top:.3rem; }
    .chip { display:inline-flex; align-items:center; gap:.45rem; padding:.32rem .8rem; border-radius:999px; font-weight:600; font-size:.85rem; }
    .status { display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .7rem; border-radius:999px; font-size:.8rem; font-weight:600; border:1px solid #E6E9EF; background:#fff; color:#6B7385; }
    .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
    .panel { background:#fff; border:1px solid #E6E9EF; border-radius:18px; padding:1.4rem 1.5rem; }
    .sig { display:flex; justify-content:space-between; align-items:center; padding:.6rem 0; border-bottom:1px solid #F0F2F5; }
    .sig:last-child { border-bottom:none; }
    .sig .name { color:#1A2233; font-weight:500; font-size:.92rem; }
    .sig .val { display:flex; align-items:center; gap:.5rem; color:#6B7385; font-weight:600; font-size:.92rem; }
    .empty { text-align:center; color:#9aa1b1; padding:3.2rem 1rem; border:1px dashed #D9DEE7; border-radius:18px; }
    .empty .big { font-size:2.2rem; margin-bottom:.4rem; }
    .stTabs [data-baseweb="tab-list"] { gap:6px; }
    .stTabs [data-baseweb="tab"] { font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)



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
        number={"suffix": "%", "font": {"size": 46, "color": INK, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)",
                     "tickfont": {"color": MUTED, "size": 11}},
            "bar": {"color": color, "thickness": 0.30},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#ECFDF3"},
                {"range": [33, 66], "color": "#FEF6E7"},
                {"range": [66, 100], "color": "#FEECEF"},
            ],
            "threshold": {"line": {"color": MUTED, "width": 2}, "thickness": 0.8,
                          "value": threshold * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=24, r=24, t=20, b=0),
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
                f'<div style="text-align:center;margin-top:-.5rem">'
                f'<span class="chip" style="background:{RISK_BG[level]};color:{color}">'
                f'<span class="dot" style="background:{color}"></span>Risque {level} · {verdict}</span></div>',
                unsafe_allow_html=True,
            )

            # Signaux du profil
            st.markdown('<div class="eyebrow" style="margin-top:1.4rem">Signaux du profil</div>',
                        unsafe_allow_html=True)
            rows = "".join(
                f'<div class="sig"><span class="name">{name}</span>'
                f'<span class="val">{val}<span class="dot" style="background:{SIG_DOT[lvl]}"></span></span></div>'
                for name, val, lvl in profile_signals(prof)
            )
            st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

            # Simulation d'action de rétention
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
        shap_img = resolve(CFG["paths"]["reports_dir"], "figures", "shap_summary.png")
        perm_img = resolve(CFG["paths"]["reports_dir"], "figures", "permutation_importance.png")
        if shap_img.exists():
            st.image(str(shap_img))
        elif perm_img.exists():
            st.image(str(perm_img))
        else:
            st.caption("Lancez `make train` pour générer ces graphiques.")


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

    st.markdown('<div class="eyebrow" style="margin-top:1.4rem">Le signal le plus fort : les échecs de paiement</div>',
                unsafe_allow_html=True)
    pf = (df.groupby("payment_failures")[TARGET].mean() * 100).reset_index()
    pf.columns = ["payment_failures", "churn_pct"]
    fig = px.bar(pf, x="payment_failures", y="churn_pct",
                 labels={"payment_failures": "Paiements ratés", "churn_pct": "Taux de churn (%)"})
    fig.update_traces(marker_color=PRIMARY)
    fig.add_hline(y=rate * 100, line_dash="dot", line_color=MUTED,
                  annotation_text=f"Moyenne {rate:.0%}", annotation_position="top left")
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=10, b=10), font={"family": "Inter", "color": INK})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Au-delà de 2 paiements ratés, le risque de départ est 2 à 3 fois la moyenne.")

    with st.expander("Explorer d'autres facteurs"):
        col = st.selectbox("Variable", ["customer_segment", "contract_type", "survey_response"])
        seg = (df.groupby(col)[TARGET].mean() * 100).reset_index()
        seg.columns = [col, "churn_pct"]
        f2 = px.bar(seg, x=col, y="churn_pct", labels={"churn_pct": "Taux de churn (%)"})
        f2.update_traces(marker_color="#A5B4FC")
        f2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(l=10, r=10, t=10, b=10), font={"family": "Inter", "color": INK})
        st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})


with tab_models:
    st.markdown('<div class="eyebrow">Performances comparées (validation croisée)</div>',
                unsafe_allow_html=True)
    comp = load_model_comparison()
    if comp is not None:
        view = comp[["modele", "pr_auc", "roc_auc", "recall"]].copy()
        best = view["pr_auc"].idxmax()
        fig = px.bar(view.sort_values("pr_auc"), x="pr_auc", y="modele", orientation="h",
                     labels={"pr_auc": "PR-AUC", "modele": ""})
        colors = [PRIMARY if i == best else "#C7CDD9" for i in view.sort_values("pr_auc").index]
        fig.update_traces(marker_color=colors)
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=10, b=10), font={"family": "Inter", "color": INK})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(view.round(3), use_container_width=True, hide_index=True)
        st.caption("Le Random Forest offre le meilleur PR-AUC ; le réseau de neurones (MLP) "
                   "est ici le moins performant — le Deep Learning n'est pas toujours supérieur.")
    else:
        st.caption("Lancez `make train` pour générer la comparaison des modèles.")
