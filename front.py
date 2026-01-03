import streamlit as st
import numpy as np

# Backend model import (trains/loads model from app.py)
from app import predict_heart_disease

# --------------------------------------------------
# Page Configuration (remove default header effect)
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# HARD OVERRIDE STREAMLIT DEFAULT HEADER
# --------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit default top padding/header gap */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* ---------- DARK THEME COLORS ---------- */
:root {
    --bg-main: #020617;
    --bg-sidebar: #020617;
    --bg-card: #0f172a;
    --text-main: #e5e7eb;
    --text-muted: #94a3b8;
    --border-soft: rgba(255,255,255,0.08);
}

/* ---------- MAIN APP ---------- */
.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

/* ---------- FORCE TEXT COLOR ---------- */
label, span, p, div {
    color: var(--text-main) !important;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-soft);
}
[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}

/* ---------- TITLE CARD ---------- */
.title-card {
    background: linear-gradient(180deg, #0f172a, #020617);
    padding: 38px;
    border-radius: 22px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.9);
    margin-bottom: 30px;
}
.main-title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
}
.main-title span {
    background: linear-gradient(90deg, #ff6b6b, #fca5a5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: var(--text-muted);
    font-size: 18px;
    margin-top: 8px;
}

/* ---------- CARD ---------- */
.card {
    background: var(--bg-card);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 24px 48px rgba(0,0,0,0.85);
    margin-bottom: 20px;
}

/* ---------- INPUTS ---------- */
input, textarea {
    background-color: #020617 !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-soft) !important;
}

/* ---------- SELECTBOX ---------- */
[data-baseweb="select"] > div {
    background-color: #020617 !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-soft) !important;
}
[role="listbox"] {
    background-color: #0f172a !important;
}
[role="option"] {
    color: var(--text-main) !important;
    background-color: #0f172a !important;
}
[role="option"][aria-selected="true"],
[role="option"]:hover {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
}

/* ---------- TOOLTIP (help popovers) ---------- */
[data-testid="stTooltip"] {
    background: #0f172a !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-soft) !important;
}
[data-testid="stTooltip"] > div {
    background: transparent !important;
    color: var(--text-main) !important;
}

/* ---------- EXPANDER (lifestyle heading) ---------- */
[data-testid="stExpander"] > details {
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-soft) !important;
}
[data-testid="stExpander"] summary {
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
}
[data-testid="stExpander"] summary span {
    color: var(--text-main) !important;
}
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:hover {
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
}

/* ---------- BUTTON ---------- */
div.stButton > button {
    background: linear-gradient(90deg, #ff6b6b, #fca5a5);
    color: #020617 !important;
    font-weight: 800;
    border-radius: 10px;
    padding: 10px 16px;
    border: none;
}

/* ---------- RESULT ---------- */
.result-high {
    background: rgba(255, 80, 80, 0.15);
    color: #fecaca !important;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #ff6b6b;
}
.result-low {
    background: rgba(34, 197, 94, 0.15);
    color: #bbf7d0 !important;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #22c55e;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CUSTOM TITLE (ONLY HEADER THAT EXISTS)
# --------------------------------------------------
st.markdown("""
<div class="title-card">
    <div class="main-title">
        ❤️ <span>Heart Disease Risk Predictor</span>
    </div>
    <div class="subtitle">
        AI-assisted clinical risk assessment system
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SMALL HELPER: CIRCULAR GAUGE (no extra deps)
# --------------------------------------------------
def render_gauge(percent: float):
    pct = max(0, min(100, percent))
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:16px;margin:12px 0;">
            <div style="
                width:120px;
                height:120px;
                border-radius:50%;
                background: conic-gradient(#fca5a5 {pct}%, #1f2937 {pct}% 100%);
                display:flex;
                align-items:center;
                justify-content:center;
                color:#e5e7eb;
                font-weight:800;
                font-size:20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.45);
            ">
                {pct:.0f}%
            </div>
            <div style="color:#e5e7eb;">Estimated probability</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left, right = st.columns([2, 1])

# --------------------------------------------------
# LEFT COLUMN — MANUAL INPUT ONLY
# --------------------------------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Details")

    # Age and Sex side by side
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 20, 80, 45, help="Age of the patient in years.")
    with c2:
        sex = st.selectbox("Sex", ["Male", "Female"], help="Biological sex of the patient.")

    # Cholesterol, BP, Sugar, Heart Rate stacked
    chol = st.slider(
        "Serum Cholesterol (mg/dl)",
        100, 400, 200,
        help="Cholesterol level in blood."
    )
    bp = st.slider(
        "Resting Blood Pressure (mm Hg)",
        80, 200, 120,
        help="Blood pressure measured at rest."
    )
    sugar = st.slider(
        "Fasting Blood Sugar (mg/dl)",
        50, 300, 100,
        help="Fasting blood sugar level in mg/dl."
    )
    heartrate = st.slider(
        "Max Heart Rate Achieved",
        60, 200, 140,
        help="Maximum heart rate during exercise test."
    )

    # ECG with only Normal/Abnormal options
    ecg_choice = st.selectbox(
        "Resting ECG",
        ["Normal", "Abnormal"],
        help="Resting ECG result."
    )
    ecg = 0 if ecg_choice == "Normal" else 1
    with st.expander("➕ Lifestyle Factors (optional)"):
        exercise = st.selectbox(
            "Regular Exercise",
            ["No", "Yes"],
            help="Does the person exercise regularly?"
        )
        smoking = st.selectbox(
            "Smoking",
            ["No", "Yes"],
            help="Current smoking status."
        )
        alcohol = st.selectbox(
            "Alcohol Intake",
            ["No", "Yes"],
            help="Regular alcohol consumption."
        )

    # Encode
    sex = 1 if sex == "Male" else 0
    sugar_flag = 1 if sugar > 120 else 0
    exercise_val = 1 if exercise == "Yes" else 0
    smoking_val = 1 if smoking == "Yes" else 0
    alcohol_val = 1 if alcohol == "Yes" else 0

    if st.button("🔍 Predict Risk"):
        # Call backend model
        percentage = predict_heart_disease(
            age=age,
            sex=sex,
            bp=bp,
            chol=chol,
            sugar=sugar_flag,
            ecg=ecg,
            heartrate=heartrate,
            exercise=exercise_val,
            smoking=smoking_val,
            alcohol=alcohol_val
        )

        if percentage > 50:
            st.markdown(
                f"<div class='result-high'>⚠️ High Risk of Heart Disease<br><b>Estimated Risk:</b> {percentage}%</div>",
                unsafe_allow_html=True
            )
            render_gauge(percentage)
        else:
            st.markdown(
                f"<div class='result-low'>✅ Low Risk of Heart Disease<br><b>Estimated Risk:</b> {percentage}%</div>",
                unsafe_allow_html=True
            )
            render_gauge(percentage)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# RIGHT COLUMN — EXPLANATION
# --------------------------------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ What does this result mean?")
    st.write("""
• **Low Risk** → Lower likelihood of heart disease  
• **High Risk** → Higher likelihood; consult a doctor  
• **Risk %** → Model confidence, not a diagnosis  
    """)
    st.write("""
⚠️ **Disclaimer:**  
This tool is for educational and decision-support purposes only.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<p style='text-align:center;opacity:0.6;'>Built for Kaggle Royale • Educational Use Only</p>",
    unsafe_allow_html=True
)