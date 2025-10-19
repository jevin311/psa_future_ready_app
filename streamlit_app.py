# psa_future_ready_app.py
# PSA Future-Ready Workforce — AI & ML Integrated Streamlit App
# Run with: streamlit run psa_future_ready_app.py
# Place in same folder as: Employee_Profiles.json and Functions & Skills.xlsx

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
from typing import Tuple, Dict
from openai import OpenAI

# -------------------------
# Setup
# -------------------------
st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — AI & ML Demo")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sidebar for OpenAI key
api_key_input = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input


def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


# -------------------------
# Data Loading
# -------------------------
@st.cache_data
def load_employee_json():
    p = os.path.join(BASE_DIR, "Employee_Profiles.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            st.error(f"Could not load {p}: {e}")
            return []
    st.error(f"File not found: {p}")
    return []


@st.cache_data
def load_functions_skills():
    p = os.path.join(BASE_DIR, "Functions & Skills.xlsx")
    if os.path.isfile(p):
        try:
            xl = pd.read_excel(p, sheet_name=None, engine="openpyxl")
            dfs = []
            for name, df in xl.items():
                df["__sheet__"] = name
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)
        except Exception as e:
            st.error(f"Error reading Excel: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# -------------------------
# Helper Functions
# -------------------------
def derive_is_leader(profile: dict) -> int:
    def contains_lead_word(title):
        if not title:
            return False
        for kw in ["manager", "lead", "head", "director", "chief"]:
            if kw in title.lower():
                return True
        return False

    for pos in profile.get("positions_history", []) or []:
        per = pos.get("period", {}) or {}
        end = per.get("end")
        if end in (None, "", "null"):
            if contains_lead_word(pos.get("role_title", "")):
                return 1
    for pos in profile.get("positions_history", []) or []:
        if contains_lead_word(pos.get("role_title", "")):
            return 1
    for comp in profile.get("competencies", []) or []:
        name = (comp.get("name", "") or "").lower()
        level = (comp.get("level", "") or "").lower()
        if "leadership" in name or "stakeholder" in name:
            if level in ("advanced", "expert"):
                return 1
    for proj in profile.get("projects", []) or []:
        if contains_lead_word(proj.get("role", "")):
            return 1
    return 0


def profile_to_features(profile: dict) -> Dict:
    ei = profile.get("employment_info", {}) or {}
    now_str = ei.get("last_updated") or datetime.now().strftime("%Y-%m-%d")
    try:
        now = datetime.fromisoformat(now_str)
    except Exception:
        now = datetime.utcnow()

    def parse_date(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s)
        except Exception:
            try:
                return datetime.strptime(s, "%Y-%m-%d")
            except Exception:
                return None

    hire = parse_date(ei.get("hire_date"))
    in_role_since = parse_date(ei.get("in_role_since"))
    years_total = (now - hire).days / 365.25 if hire else 0.0
    years_in_role = (now - in_role_since).days / 365.25 if in_role_since else 0.0

    skills = profile.get("skills", []) or []
    comps = profile.get("competencies", []) or []
    projs = profile.get("projects", []) or []
    positions = profile.get("positions_history", []) or []
    languages = profile.get("personal_info", {}).get("languages", []) or []

    num_trainings = len(profile.get("training_history", []) or [])
    has_lead_comp = int(any("lead" in (c.get("name", "").lower()) for c in comps))

    return {
        "years_total": years_total,
        "years_in_role": years_in_role,
        "num_skills": len(skills),
        "num_competencies": len(comps),
        "num_projects": len(projs),
        "num_positions": len(positions),
        "num_languages": len(languages),
        "has_lead_comp": has_lead_comp,
        "num_trainings": num_trainings,
    }


# -------------------------
# Model Training
# -------------------------
@st.cache_resource
def build_dataset_and_train():
    employees = load_employee_json()
    if not employees:
        return None, {}

    rows = []
    for p in employees:
        feat = profile_to_features(p)
        label = derive_is_leader(p)
        feat["employee_id"] = p.get("employee_id")
        feat["label"] = label
        rows.append(feat)

    df = pd.DataFrame(rows).set_index("employee_id")
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)

    if len(df) >= 6 and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        except Exception:
            auc = None
    else:
        clf.fit(Xs, y)
        acc, auc = None, None

    return {"model": clf, "scaler": scaler, "feature_cols": feature_cols}, {"df": df, "accuracy": acc, "auc": auc, "n": len(df)}


def predict_for_profile(model_pack: dict, profile: dict) -> Dict:
    feat = profile_to_features(profile)
    feature_cols = model_pack["feature_cols"]
    X_row = np.array([feat[c] for c in feature_cols]).reshape(1, -1)
    Xs = model_pack["scaler"].transform(X_row)
    prob = model_pack["model"].predict_proba(Xs)[0, 1]
    pred = int(model_pack["model"].predict(Xs)[0])

    coefs = model_pack["model"].coef_[0]
    contrib = {feature_cols[i]: float(coefs[i] * Xs[0, i]) for i in range(len(feature_cols))}
    contrib_sorted = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    return {"probability": float(prob), "prediction": pred, "features": feat, "contributions": contrib_sorted}


# -------------------------
# Conversational Assistant (AI or fallback)
# -------------------------
def handle_conversation(profile: dict, message: str) -> Dict:
    client = get_openai_client()
    if not client:
        # fallback simple logic
        m = (message or "").lower()
        if any(w in m for w in ["stress", "tired", "anxiety", "burnout"]):
            return {"reply": "I'm sorry you're feeling this way. Try a 5-min pause or contact EAP for support."}
        if "career" in m or "promotion" in m:
            return {"reply": "You can explore internal mobility or leadership upskilling. Try 'Generate Career Pathway' below!"}
        return {"reply": "I can help with training, career planning, or wellbeing — ask about 'career' or 'stress'."}

    name = profile.get("personal_info", {}).get("name", "the employee")
    role = profile.get("employment_info", {}).get("job_title", "employee")
    skills = [s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]

    system_prompt = f"""
    You are PSA's AI Career & Wellbeing Assistant.
    You give empathetic, inclusive, and growth-focused guidance to employees.
    Employee: {name}, Role: {role}, Skills: {', '.join(skills[:10])}.
    Respond as a friendly HR coach.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        max_tokens=200,
        temperature=0.8,
    )
    return {"reply": response.choices[0].message.content.strip()}


# -------------------------
# Career Pathway Generator
# -------------------------
def generate_career_pathway(profile: dict, leadership_prob: float) -> Dict:
    client = get_openai_client()
    role = profile.get("employment_info", {}).get("job_title", "employee")

    if not client:
        if "analyst" in role.lower():
            return {"ai_reply": "Suggested path: Senior Analyst → Principal Analyst → Team Lead. Upskill in analytics, stakeholder engagement, and leadership."}
        elif "engineer" in role.lower():
            return {"ai_reply": "Suggested path: Senior Engineer → Tech Lead → Engineering Manager. Focus on cloud, system design, and people leadership."}
        return {"ai_reply": "Consider broadening skills through PSA Academy and mentorship to explore new mobility paths."}

    name = profile.get("personal_info", {}).get("name", "the employee")
    skills = [s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]
    competencies = [c.get("name") for c in profile.get("competencies", []) if c.get("name")]

    system_prompt = f"""
    You are PSA's Workforce Development Advisor.
    Generate inclusive, realistic internal mobility and upskilling plans.
    Employee: {name}, Role: {role}, Leadership potential: {leadership_prob:.1%}
    Skills: {', '.join(skills[:12])}
    Competencies: {', '.join(competencies[:8])}
    Provide 2-3 career pathways and specific upskilling/reskilling recommendations.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate a personalized career roadmap and training plan."},
        ],
        max_tokens=350,
        temperature=0.8,
    )
    return {"ai_reply": response.choices[0].message.content.strip()}


# -------------------------
# Streamlit App Layout
# -------------------------
st.title("🌟 PSA Future-Ready Workforce (AI & ML Platform)")

employees = load_employee_json()
functions_df = load_functions_skills()
model_pack, info = build_dataset_and_train()

if not employees:
    st.error("No employee profiles loaded. Place Employee_Profiles.json in the same folder.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📊 Model & Data Overview")
    st.write(f"Profiles loaded: **{len(employees)}**")
    if info.get("accuracy") is not None:
        st.metric("Accuracy", f"{info['accuracy']:.3f}")
    st.write("Heuristic labels mark employees with leadership indicators (manager titles, competencies).")

with col2:
    st.header("👤 Employee Insights")
    emp_map = {e.get("employee_id"): e for e in employees}
    emp_id = st.selectbox("Select employee", [""] + list(emp_map.keys()))
    if emp_id:
        profile = emp_map[emp_id]
        st.subheader(f"{profile.get('personal_info', {}).get('name', '')} — {profile.get('employment_info', {}).get('job_title', '')}")

        res = predict_for_profile(model_pack, profile)
        st.write(f"**Leadership potential:** {res['probability']:.1%}")
        st.write(f"Prediction: {'Leader' if res['prediction']==1 else 'Individual Contributor'}")
        st.write("Top contributing features:")
        for f, v in res["contributions"][:5]:
            st.write(f"- {f}: {v:.3f}")

        st.markdown("---")
        st.subheader("🚀 Generate Personalised Career Pathway")
        if st.button("Generate Career Pathway"):
            with st.spinner("AI is crafting your career roadmap..."):
                plan = generate_career_pathway(profile, res["probability"])
            st.write(plan.get("ai_reply"))

        st.markdown("---")
        st.subheader("💬 Assistant (Conversational Support)")
        msg = st.text_input("Ask about career, training, or wellbeing:")
        if st.button("Ask Assistant"):
            if msg.strip():
                reply = handle_conversation(profile, msg)
                st.write(reply["reply"])
            else:
                st.warning("Please type a question first.")

st.markdown("---")
st.subheader("📈 Derived Dataset")
if info.get("df") is not None:
    st.dataframe(info["df"].reset_index())
