from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Optional: calibration makes probabilities less extreme (more realistic)
try:
    from sklearn.calibration import CalibratedClassifierCV
    HAS_CALIBRATION = True
except Exception:
    HAS_CALIBRATION = False


# =========================
# Config + Styling
# =========================
st.set_page_config(page_title="Underwriter Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1250px; padding-top: 1.1rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

.card {
  border: 1px solid rgba(49, 51, 63, 0.18);
  border-radius: 16px;
  padding: 16px 18px;
  background: rgba(255,255,255,0.02);
}

.subtle { color: rgba(255,255,255,0.72); font-size: 0.92rem; }
.smallcap { color: rgba(255,255,255,0.70); font-size: 0.85rem; margin-bottom: 6px; }

.pill {
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: 650;
  border: 1px solid rgba(49, 51, 63, 0.25);
  background: rgba(255,255,255,0.03);
}

.kv {
  display:flex;
  justify-content:space-between;
  gap: 18px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}
.kv:last-child { border-bottom: none; }

hr.soft { border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 12px 0; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================
# Paths + Schema
# =========================
APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR / "data" / "past_cases.csv"

CAT_COLS = ["occupation_risk", "location_risk", "income_stability", "prior_insurance"]
NUM_COLS = ["age", "data_completeness"]
TARGET_COL = "claim_outcome"
REQUIRED_COLS = set(CAT_COLS + NUM_COLS + [TARGET_COL])

FRIENDLY = {
    "occupation_risk=high": "High-risk occupation",
    "occupation_risk=med": "Moderate-risk occupation",
    "occupation_risk=low": "Low-risk occupation",
    "location_risk=high": "High-risk area",
    "location_risk=med": "Moderate-risk area",
    "location_risk=low": "Low-risk area",
    "income_stability=low": "Unstable income",
    "income_stability=med": "Moderately stable income",
    "income_stability=high": "Stable income",
    "prior_insurance=yes": "Has prior insurance history",
    "prior_insurance=no": "No prior insurance history",
    "age": "Age factor",
    "data_completeness": "Data completeness"
}


# =========================
# Utils
# =========================
def clamp_int(x: float, lo: int = 0, hi: int = 100) -> int:
    return int(round(float(np.clip(x, lo, hi))))

def df_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.md5(payload.tobytes()).hexdigest()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"CSV missing required columns: {sorted(list(missing))}")
        st.stop()

    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip().str.lower()

    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int).clip(0, 120)
    df["data_completeness"] = pd.to_numeric(df["data_completeness"], errors="coerce").fillna(0).astype(int).clip(0, 100)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int).clip(0, 1)

    return df

def build_model(df: pd.DataFrame):
    """
    Fix for 0/100 risk:
    - Stronger regularization (C smaller)
    - Balanced class weights
    - Optional calibration (if available + data allows)
    - Probability clipping + smoothing
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", "passthrough", NUM_COLS)
    ])

    base_lr = LogisticRegression(
        max_iter=3000,
        C=0.25,                # stronger regularization (reduces extreme probs)
        class_weight="balanced",
        solver="lbfgs"
    )

    pipe = Pipeline([
        ("prep", pre),
        ("model", base_lr)
    ])

    # Calibration makes probabilities less extreme (more realistic)
    if HAS_CALIBRATION:
        # calibration requires enough samples in each class
        class_counts = y.value_counts().to_dict()
        min_class = min(class_counts.get(0, 0), class_counts.get(1, 0))
        if min_class >= 6:
            # Train preprocessor first, then calibrate the estimator on transformed data
            Xt = pre.fit_transform(X, y)
            lr = LogisticRegression(max_iter=3000, C=0.25, class_weight="balanced", solver="lbfgs")
            cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
            cal.fit(Xt, y)

            class CalibratedPipe:
                def __init__(self, preprocessor, calibrated_model):
                    self.pre = preprocessor
                    self.cal = calibrated_model
                def predict_proba(self, Xnew):
                    Xnew_t = self.pre.transform(Xnew)
                    return self.cal.predict_proba(Xnew_t)

            return CalibratedPipe(pre, cal), X

    # fallback: normal pipeline
    pipe.fit(X, y)
    return pipe, X

def premium_gauge(title: str, value: int, bands: List[Tuple[int,int,str]]) -> go.Figure:
    """
    Filled-arc gauge (bar shows the measurement).
    No threshold line.
    """
    steps = [{"range": [a, b], "color": c} for a, b, c in bands]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "/100", "font": {"size": 34}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0},
            "bar": {"thickness": 0.35, "color": "rgba(255,255,255,0.90)"},  # filled measurement arc
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def compute_confidence(data_completeness: int, top_similarity: float) -> int:
    top_similarity = float(np.clip(top_similarity, 0, 1))
    raw = 0.7 * (data_completeness / 100.0) + 0.3 * top_similarity
    return clamp_int(raw * 100)

def compute_success(risk: int, confidence: int) -> int:
    base = 100 - risk
    uncertainty_penalty = (100 - confidence) * 0.4
    return clamp_int(base - uncertainty_penalty)

def recommend_action(risk: int, confidence: int) -> str:
    if confidence < 40:
        return "ESCALATE (Human Review Required)"
    if risk > 70:
        return "LIMITED COVERAGE (Higher Deductible / Short Term)"
    return "APPROVE ELIGIBLE (Standard Terms)"


# =========================
# Header
# =========================
st.markdown("<h1 style='text-align:center;'>Underwriter Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtle' style='text-align:center;'>AI-assisted decision support with Risk + Confidence + Explainability.</p>", unsafe_allow_html=True)
st.write("")


# =========================
# Sidebar: Data
# =========================
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload past_cases.csv", type=["csv"])
    st.caption("Tip: upload a bigger dataset for smoother scoring (more variety reduces extremes).")

# Load + normalize dataset
if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    if not CSV_PATH.exists():
        st.error(f"Missing file: {CSV_PATH}. Create data/past_cases.csv")
        st.stop()
    df_raw = pd.read_csv(CSV_PATH)

df = normalize_df(df_raw)

# Cache model by dataset fingerprint
fp = df_fingerprint(df)
if st.session_state.get("fp") != fp:
    model, X_train = build_model(df)
    st.session_state["model"] = model
    st.session_state["X_train"] = X_train
    st.session_state["fp"] = fp

model = st.session_state["model"]
X_train = st.session_state["X_train"]


# =========================
# Input Card
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<div class='smallcap'>Applicant Input</div>", unsafe_allow_html=True)

with st.form("applicant_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.2, 1.5])

    with c1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
        prior = st.selectbox("Prior Insurance", ["no", "yes"], index=0)

    with c2:
        occupation = st.selectbox("Occupation Risk", ["low", "med", "high"], index=0)
        income = st.selectbox("Income Stability", ["low", "med", "high"], index=1)

    with c3:
        location = st.selectbox("Location Risk", ["low", "med", "high"], index=1)
        completeness = st.slider("Data Completeness (%)", 0, 100, 60)

    with c4:
        st.markdown("<div class='smallcap'>Actions</div>", unsafe_allow_html=True)
        score_btn = st.form_submit_button("Score Applicant ✅")
        st.caption("Scores update on submit.")

st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Score
# =========================
if score_btn:
    applicant = {
        "age": int(age),
        "occupation_risk": occupation,
        "location_risk": location,
        "income_stability": income,
        "prior_insurance": prior,
        "data_completeness": int(completeness),
    }
    app_df = pd.DataFrame([applicant])

    # --- Risk probability ---
    prob_claim = float(model.predict_proba(app_df)[0, 1])

    # IMPORTANT: prevent extreme 0/1 outputs ruining the dashboard
    # This is a common real-world technique (probability clipping + smoothing)
    prob_claim = float(np.clip(prob_claim, 0.03, 0.97))
    prob_claim = 0.90 * prob_claim + 0.10 * 0.5  # slight smoothing toward 0.5

    risk = clamp_int(prob_claim * 100)

    # --- Similarity coverage ---
    # For calibrated model wrapper we don't have prep stored the same way, so handle both.
    if hasattr(model, "named_steps"):
        prep = model.named_steps["prep"]
        Xt = prep.transform(X_train)
        nt = prep.transform(app_df)
    else:
        # CalibratedPipe uses .pre
        prep = model.pre
        Xt = prep.transform(X_train)
        nt = prep.transform(app_df)

    sims = cosine_similarity(nt, Xt)[0]
    top_idx = np.argsort(sims)[::-1][:3]
    top_sim = float(np.clip(sims[top_idx[0]], 0, 1))

    confidence = compute_confidence(int(completeness), top_sim)
    success = compute_success(risk, confidence)
    rec = recommend_action(risk, confidence)

    st.session_state["result"] = {
        "applicant": applicant,
        "risk": risk,
        "confidence": confidence,
        "success": success,
        "recommendation": rec,
        "similar_idx": top_idx.tolist(),
        "similar_sims": sims[top_idx].round(3).tolist()
    }


# =========================
# Dashboard Output
# =========================
if "result" not in st.session_state:
    st.info("Fill applicant data above and click **Score Applicant** to view results.")
else:
    res = st.session_state["result"]
    a = res["applicant"]

    left, mid, right = st.columns([1.05, 1.2, 1.2], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Applicant Summary</h3>", unsafe_allow_html=True)

        def kv(k, v):
            st.markdown(f"<div class='kv'><div class='subtle'>{k}</div><div><b>{v}</b></div></div>", unsafe_allow_html=True)

        kv("Age", a["age"])
        kv("Occupation Risk", a["occupation_risk"].upper())
        kv("Income Stability", a["income_stability"].upper())
        kv("Location Risk", a["location_risk"].upper())
        kv("Prior Insurance", a["prior_insurance"].upper())
        kv("Data Completeness", f'{a["data_completeness"]}%')
        st.markdown("</div>", unsafe_allow_html=True)

    bands_risk = [(0, 40, "#2ecc71"), (40, 70, "#f1c40f"), (70, 100, "#e74c3c")]
    bands_conf = [(0, 40, "#e74c3c"), (40, 70, "#f1c40f"), (70, 100, "#2ecc71")]
    bands_succ = [(0, 40, "#e74c3c"), (40, 70, "#f1c40f"), (70, 100, "#2ecc71")]

    with mid:
        st.plotly_chart(premium_gauge("Risk Score", res["risk"], bands_risk), use_container_width=True)

    with right:
        st.plotly_chart(premium_gauge("Confidence Score", res["confidence"], bands_conf), use_container_width=True)
        st.plotly_chart(premium_gauge("Success Likelihood", res["success"], bands_succ), use_container_width=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.6, 1.0], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Key Factors</h3>", unsafe_allow_html=True)
        st.write("• Data completeness influences confidence.")
        st.write("• Similar past cases influence risk estimate.")
        st.write("• Guardrails prevent unsafe automation.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>System Recommendation</h3>", unsafe_allow_html=True)
        icon = "✅" if "APPROVE" in res["recommendation"] else "⚠️" if "LIMITED" in res["recommendation"] else "🛑"
        st.markdown(f"<div class='pill'>{icon} {res['recommendation']}</div>", unsafe_allow_html=True)

        st.write("")
        b1, b2, b3 = st.columns(3)
        with b1: st.button("Approve", use_container_width=True)
        with b2: st.button("Adjust", use_container_width=True)
        with b3: st.button("Escalate", use_container_width=True)

        st.caption("Buttons represent workflow steps. Final decision remains human-led.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Similar Past Cases</h3>", unsafe_allow_html=True)
    sim_df = df.iloc[res["similar_idx"]].copy()
    sim_df["similarity"] = res["similar_sims"]

    st.dataframe(
        sim_df[["age","occupation_risk","location_risk","income_stability","prior_insurance","data_completeness","claim_outcome","similarity"]],
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

