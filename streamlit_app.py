# psa_future_ready_app.py
# PSA Future-Ready Workforce — Complete AI + ML Solution
# Place in same folder as:
#   Employee_Profiles.json
#   Functions & Skills.xlsx
# Run:
#   streamlit run psa_future_ready_app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
from typing import Tuple, Dict, List
from openai import OpenAI
from pathlib import Path

# ===============================
# CONFIGURATION
# ===============================
st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — Full Solution")
BASE_DIR = Path(__file__).resolve().parent
EMP_PATH = BASE_DIR / "Employee_Profiles.json"
FUNC_PATH = BASE_DIR / "Functions & Skills.xlsx"
MODEL_PATH = BASE_DIR / "psa_leadership_model.joblib"

# ===============================
# SIDEBAR CONFIG
# ===============================
st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input("🔑 OpenAI API Key", type="password", help="Paste your OpenAI key here for AI features.")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

st.sidebar.markdown("### 🧭 Accessibility")
if st.sidebar.checkbox("High contrast mode"):
    st.markdown("<style>.stApp{background-color:black;color:white;}</style>", unsafe_allow_html=True)
if st.sidebar.checkbox("Large font size"):
    st.markdown("<style>.stApp *{font-size:18px !important;}</style>", unsafe_allow_html=True)

# ===============================
# HELPER FUNCTIONS
# ===============================
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

@st.cache_data
def load_employee_json():
    if EMP_PATH.exists():
        try:
            return json.loads(EMP_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not parse Employee_Profiles.json: {e}")
            return []
    else:
        st.error("Employee_Profiles.json not found.")
        return []

@st.cache_data
def load_functions_skills():
    if FUNC_PATH.exists():
        try:
            xl = pd.read_excel(FUNC_PATH, sheet_name=None, engine="openpyxl")
            frames = []
            for name, df in xl.items():
                df["__sheet__"] = name
                frames.append(df)
            return pd.concat(frames, ignore_index=True)
        except Exception as e:
            st.warning(f"Error reading Functions & Skills.xlsx: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def contains_lead_word(title):
    if not title:
        return False
    t = title.lower()
    for kw in ["manager", "lead", "head", "director", "chief"]:
        if kw in t:
            return True
    return False

def derive_is_leader(profile):
    for pos in profile.get("positions_history", []) or []:
        if contains_lead_word(pos.get("role_title", "")):
            return 1
    for c in profile.get("competencies", []) or []:
        if "leadership" in (c.get("name") or "").lower():
            return 1
    return 0

def parse_date(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def profile_to_features(profile):
    ei = profile.get("employment_info", {}) or {}
    now = datetime.utcnow()
    hire = parse_date(ei.get("hire_date"))
    in_role = parse_date(ei.get("in_role_since"))
    years_total = (now - hire).days / 365.25 if hire else 0
    years_in_role = (now - in_role).days / 365.25 if in_role else 0
    skills = profile.get("skills", []) or []
    comps = profile.get("competencies", []) or []
    projs = profile.get("projects", []) or []
    return {
        "years_total": years_total,
        "years_in_role": years_in_role,
        "num_skills": len(skills),
        "num_competencies": len(comps),
        "num_projects": len(projs),
    }

@st.cache_resource
def build_dataset_and_train():
    employees = load_employee_json()
    if not employees:
        return None, {}
    rows = []
    for p in employees:
        feat = profile_to_features(p)
        feat["label"] = derive_is_leader(p)
        feat["employee_id"] = p.get("employee_id")
        rows.append(feat)
    df = pd.DataFrame(rows).set_index("employee_id")
    X = df[[c for c in df.columns if c != "label"]].fillna(0).astype(float)
    y = df["label"].astype(int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    if len(df) > 5 and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
        clf.fit(X_train, y_train)
        acc = float(accuracy_score(y_test, clf.predict(X_test)))
        auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    else:
        clf.fit(Xs, y)
        acc, auc = None, None
    joblib.dump({"model": clf, "scaler": scaler, "feature_cols": list(X.columns)}, MODEL_PATH)
    return {"model": clf, "scaler": scaler, "feature_cols": list(X.columns)}, {"df": df, "accuracy": acc, "auc": auc}

def predict_for_profile(model_pack, profile):
    feat = profile_to_features(profile)
    X_row = np.array([[feat[c] for c in model_pack["feature_cols"]]], dtype=float)
    Xs = model_pack["scaler"].transform(X_row)
    prob = float(model_pack["model"].predict_proba(Xs)[0, 1])
    return prob

# ===============================
# AI FUNCTIONS
# ===============================
def handle_conversation(profile, message):
    client = get_openai_client()
    name = profile.get("personal_info", {}).get("name", "the employee")
    role = profile.get("employment_info", {}).get("job_title", "employee")
    if not client:
        return {"reply": f"Hi {name}, I'm your career assistant. I can help with career growth, stress management, or upskilling."}
    prompt = f"""
    You are PSA's Career & Wellbeing Assistant.
    Employee name: {name}
    Current role: {role}.
    Task: Respond concisely and empathetically to the employee's message below.
    Message: {message}
    Provide advice related to career growth, wellbeing, or skill development.
    """
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a supportive HR wellbeing assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=250
        )
        return {"reply": r.choices[0].message.content.strip()}
    except Exception as e:
        return {"reply": f"(AI error: {e})"}

def generate_career_pathway(profile, leadership_prob):
    client = get_openai_client()
    name = profile.get("personal_info", {}).get("name", "the employee")
    role = profile.get("employment_info", {}).get("job_title", "employee")
    skills = [s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]
    if not client:
        return {"ai_reply": f"As {role}, consider next steps like Senior {role}, Team Lead, or Manager roles. Strengthen skills: leadership, stakeholder management, and innovation."}
    sys_prompt = f"""
    You are PSA's AI Career Advisor.
    Employee: {name}, current role: {role}.
    Leadership potential: {leadership_prob:.2%}.
    Skills: {', '.join(skills[:10])}.
    Suggest 2–3 future roles, each with 2–3 upskilling areas and internal mobility options.
    Keep concise, actionable, and aligned with PSA's values.
    """
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Generate a personalised PSA career pathway."},
            ],
            temperature=0.8,
            max_tokens=400
        )
        return {"ai_reply": r.choices[0].message.content.strip()}
    except Exception as e:
        return {"ai_reply": f"(Error generating pathway: {e})"}

# ===============================
# UI
# ===============================
st.title("🚀 PSA Future-Ready Workforce — Complete AI Platform")
st.write("AI-driven leadership prediction, personalised career pathways, conversational assistant, mentorship, and recognition system.")

employees = load_employee_json()
functions_df = load_functions_skills()
model_pack, info = build_dataset_and_train()

if not employees:
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.header("📊 Model Overview")
    if info.get("accuracy"):
        st.metric("Model Accuracy", f"{info['accuracy']:.3f}")
    if info.get("auc"):
        st.metric("ROC AUC", f"{info['auc']:.3f}")
    st.caption("Model trained using heuristic leadership labels and feature engineering.")

with right:
    st.header("👥 Employee Explorer")
    emp_map = {e.get("employee_id"): e for e in employees}
    selected_id = st.selectbox("Select Employee", [""] + list(emp_map.keys()))
    if selected_id:
        profile = emp_map[selected_id]
        name = profile.get("personal_info", {}).get("name", "")
        role = profile.get("employment_info", {}).get("job_title", "")
        dept = profile.get("employment_info", {}).get("department", "")
        hire = profile.get("employment_info", {}).get("hire_date", "")
        st.subheader(f"{name} — {role}")
        st.markdown(f"**Department:** {dept} | **Hire Date:** {hire}")

        st.markdown("### 🧩 Background & Skills")
        st.write("**Skills:**", ", ".join([s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]))
        st.write("**Competencies:**", ", ".join([c.get("name") for c in profile.get("competencies", []) if c.get("name")]))

        st.markdown("### 🎯 Leadership Prediction")
        prob = predict_for_profile(model_pack, profile)
        st.metric("Predicted Leadership Potential", f"{prob:.2%}")
        if prob > 0.6:
            st.success("High leadership potential — consider mentorship or strategic leadership training.")
        elif prob > 0.3:
            st.info("Emerging leader — develop leadership through projects or coaching.")
        else:
            st.warning("Focus on technical upskilling and cross-functional exposure.")

        st.markdown("### 🚀 Personalised Career Pathway")
        if st.button("Generate Career Pathway"):
            with st.spinner("Generating AI pathway..."):
                plan = generate_career_pathway(profile, prob)
            st.write(plan["ai_reply"])

        st.markdown("### 💬 Conversational Career Assistant")
        user_msg = st.text_input("Ask about your career, wellbeing, or growth:")
        if st.button("Ask Assistant"):
            with st.spinner("Thinking..."):
                res = handle_conversation(profile, user_msg)
            st.write(res["reply"])

st.markdown("---")
st.caption("Built with ❤️ for PSA — empowering a future-ready workforce through AI and inclusivity.")
