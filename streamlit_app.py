# psa_future_ready_app.py
# Streamlit app: PSA Future-Ready Workforce (AI Career Growth, Engagement & Leadership Predictor)
# Run: streamlit run psa_future_ready_app.py

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

st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — AI Career Assistant")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# OpenAI setup (lazy init)
# -------------------------
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

# -------------------------
# Sidebar: optional key entry
# -------------------------
st.sidebar.header("🔐 AI Assistant Configuration")
api_key_input = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input
    st.sidebar.success("✅ API key loaded for this session")

# -------------------------
# Loaders
# -------------------------
@st.cache_data
def load_employee_json():
    p = os.path.join(BASE_DIR, "Employee_Profiles.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load JSON: {e}")
    st.error(f"Missing Employee_Profiles.json at {p}")
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
            st.warning(f"Could not read Excel: {e}")
    st.warning("No Functions & Skills.xlsx found.")
    return pd.DataFrame()

# -------------------------
# Leadership label heuristic
# -------------------------
def derive_is_leader(profile: dict) -> int:
    def contains_lead(title):
        if not title:
            return False
        title_l = title.lower()
        for kw in ["manager", "lead", "head", "director", "chief"]:
            if kw in title_l:
                return True
        return False

    for pos in profile.get("positions_history", []):
        if contains_lead(pos.get("role_title", "")):
            return 1

    for c in profile.get("competencies", []):
        name = (c.get("name", "") or "").lower()
        level = (c.get("level", "") or "").lower()
        if "leadership" in name and level in ("advanced", "expert"):
            return 1
    return 0

# -------------------------
# Feature engineering
# -------------------------
def profile_to_features(profile: dict) -> Dict:
    ei = profile.get("employment_info", {}) or {}
    now = datetime.now()

    def parse_date(s):
        try:
            return datetime.fromisoformat(s)
        except Exception:
            try:
                return datetime.strptime(s, "%Y-%m-%d")
            except Exception:
                return None

    hire = parse_date(ei.get("hire_date"))
    in_role_since = parse_date(ei.get("in_role_since"))
    years_total = (now - hire).days / 365.25 if hire else 0
    years_in_role = (now - in_role_since).days / 365.25 if in_role_since else 0

    return {
        "years_total": years_total,
        "years_in_role": years_in_role,
        "num_skills": len(profile.get("skills", [])),
        "num_competencies": len(profile.get("competencies", [])),
        "num_projects": len(profile.get("projects", [])),
        "num_positions": len(profile.get("positions_history", [])),
        "num_languages": len(profile.get("personal_info", {}).get("languages", [])),
        "has_lead_comp": int(any("lead" in (c.get("name","").lower()) for c in profile.get("competencies", []))),
        "num_trainings": len(profile.get("training_history", [])) if "training_history" in profile else 0,
    }

# -------------------------
# Model training
# -------------------------
@st.cache_resource
def build_dataset_and_train():
    employees = load_employee_json()
    if not employees:
        return None, {}

    rows = []
    for p in employees:
        feat = profile_to_features(p)
        feat["employee_id"] = p.get("employee_id")
        feat["label"] = derive_is_leader(p)
        rows.append(feat)
    df = pd.DataFrame(rows).set_index("employee_id")

    X = df.drop(columns=["label"])
    y = df["label"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)

    if len(df) >= 6 and len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    else:
        clf.fit(Xs, y)
        acc, auc = None, None

    model_pack = {"model": clf, "scaler": scaler, "cols": list(X.columns)}
    info = {"df": df, "accuracy": acc, "auc": auc, "n": len(df)}
    joblib.dump(model_pack, os.path.join(BASE_DIR, "psa_leadership_model.joblib"))
    return model_pack, info

# -------------------------
# Prediction
# -------------------------
def predict_for_profile(model_pack: dict, profile: dict) -> Dict:
    feat = profile_to_features(profile)
    X = np.array([feat[c] for c in model_pack["cols"]]).reshape(1, -1)
    Xs = model_pack["scaler"].transform(X)
    prob = float(model_pack["model"].predict_proba(Xs)[0, 1])
    pred = int(model_pack["model"].predict(Xs)[0])
    return {"probability": prob, "prediction": pred, "features": feat}

# -------------------------
# Simple training recommender
# -------------------------
DEFAULT_TRAINING_DB = {
    "cloud": ["Cloud Architecture Masterclass", "Infrastructure as Code (IaC)"],
    "leadership": ["Leading High Performance Teams", "Coaching Skills for Managers"],
    "analytics": ["Data Analytics Foundations", "Advanced Excel for Business"],
    "finance": ["Strategic Finance", "Financial Modeling Bootcamp"]
}

def recommend_trainings_from_skills(profile: dict) -> Tuple[list, list]:
    skills = [(s.get("skill_name") or "").lower() for s in profile.get("skills", [])]
    matched, trainings = [], []
    for sk in skills:
        for key in DEFAULT_TRAINING_DB:
            if key in sk:
                trainings += DEFAULT_TRAINING_DB[key]
                matched.append(sk)
    return list(dict.fromkeys(trainings)), matched

# -------------------------
# Conversational assistant (OpenAI + fallback)
# -------------------------
def handle_conversation(profile: dict, message: str, leadership_prob: float) -> Dict:
    client = get_openai_client()
    if not client:
        # rule-based fallback
        msg = message.lower()
        if "stress" in msg or "burnout" in msg:
            return {"reply": "I'm sorry you're feeling stressed. Try taking a break or reaching out to our Employee Assistance Programme (EAP). Would you like a link to EAP resources?"}
        if "career" in msg or "promotion" in msg:
            return {"reply": "I can suggest training and mentorship options to help your career growth. Try asking: 'What skills should I build for leadership?'"}
        if "training" in msg or "learn" in msg:
            t, matched = recommend_trainings_from_skills(profile)
            if t:
                return {"reply": f"Based on your skills ({', '.join(matched)}), you might explore: {', '.join(t)}"}
            return {"reply": "Tell me which skill you'd like to learn, and I'll recommend courses."}
        return {"reply": "I can help with wellbeing, career, and training guidance. Try asking about 'career growth' or 'stress'."}

    try:
        name = profile.get("personal_info", {}).get("name", "the employee")
        role = profile.get("employment_info", {}).get("job_title", "employee")
        skills = [s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]

        system_prompt = f"""
        You are PSA's AI Career Assistant — a compassionate and insightful advisor helping employees
        thrive in a fast-evolving logistics and technology environment.

        Context:
        - PSA aims to future-proof its workforce through continuous upskilling, well-being, and inclusivity.
        - Your role is to help {name}, who currently works as {role}, identify development goals, career
          pathways, and mental wellness practices.
        - Leadership potential score (predicted): {leadership_prob:.2%}
        - Employee’s key skills: {', '.join(skills[:10])}

        Objectives:
        1. Offer empathetic, professional advice aligned with PSA's values (inclusivity, safety, innovation).
        2. Provide tailored suggestions for:
           • Career advancement or role transitions
           • Relevant upskilling or internal mobility opportunities
           • Mental well-being and engagement
           • Mentorship and inclusive leadership support
        3. Keep responses concise (≤ 6 sentences) and actionable.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=300,
            temperature=0.8,
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"reply": f"(AI assistant error: {e})"}

# -------------------------
# Streamlit layout
# -------------------------
st.title("🚢 PSA — Future-Ready Workforce (AI for Growth & Engagement)")
st.markdown("""
Empower PSA employees with AI-driven career insights, leadership prediction, and personalised well-being support.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data & Leadership Model")
    employees = load_employee_json()
    st.write(f"Loaded {len(employees)} profiles.")
    if st.button("Train / Re-train Model"):
        with st.spinner("Training leadership model..."):
            model_pack, info = build_dataset_and_train()
        st.success("✅ Model trained.")
    else:
        model_pack, info = build_dataset_and_train()

with col2:
    st.header("Employee Insights & Career Assistant")
    emp_map = {e.get("employee_id"): e for e in employees}
    emp_id = st.selectbox("Select employee", [""] + list(emp_map.keys()))

    if emp_id:
        profile = emp_map[emp_id]
        name = profile.get("personal_info", {}).get("name", "")
        title = profile.get("employment_info", {}).get("job_title", "")
        st.subheader(f"{name} — {title}")

        res = predict_for_profile(model_pack, profile)
        prob = res["probability"]
        st.write(f"**Leadership Potential:** {prob:.1%}")
        st.caption("Based on role history, competencies, and experience patterns.")

        if prob > 0.6:
            st.success("High leadership potential — consider targeted mentorship or stretch assignments.")
        elif prob > 0.3:
            st.info("Emerging leader — explore leadership development programs.")
        else:
            st.warning("Potential for growth — consider skill broadening and coaching opportunities.")

        st.markdown("---")
        st.subheader("💬 AI Career Assistant")
        message = st.text_input("Ask about your career, training, or well-being:")
        if st.button("Ask Assistant"):
            reply = handle_conversation(profile, message, prob)
            st.write(reply["reply"])

st.markdown("---")
st.header("Dataset Overview")
if "df" in info and not info["df"].empty:
    st.dataframe(info["df"].reset_index())
