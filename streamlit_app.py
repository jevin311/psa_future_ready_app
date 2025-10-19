# psa_future_ready_app.py
# Streamlit app: PSA Future-Ready Workforce (ML leadership predictor + AI conversational assistant)
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
from openai import OpenAI  # ✅ for AI assistant

st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — ML + AI Assistant")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# OpenAI client initialization (lazy, handles missing key)
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
# Sidebar: optional API key input
# -------------------------
st.sidebar.header("🔐 AI Assistant Configuration")
api_key_input = st.sidebar.text_input("Enter your OpenAI API key", type="password")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input
    st.sidebar.success("✅ API key loaded for this session")

# -------------------------
# File loaders
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
# Label derivation
# -------------------------
def derive_is_leader(profile: dict) -> int:
    def contains_lead_word(title):
        if not title:
            return False
        t = title.lower()
        for kw in ["manager", "lead", "head", "director", "chief"]:
            if kw in t:
                return True
        return False

    ph = profile.get("positions_history", []) or []
    for pos in ph:
        end = (pos.get("period", {}) or {}).get("end")
        if end in (None, "", "null"):
            if contains_lead_word(pos.get("role_title", "")):
                return 1

    for pos in ph:
        if contains_lead_word(pos.get("role_title", "")):
            return 1

    for comp in profile.get("competencies", []) or []:
        if "leadership" in (comp.get("name", "") or "").lower():
            if (comp.get("level", "") or "").lower() in ("advanced", "expert"):
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

    skills = profile.get("skills", [])
    competencies = profile.get("competencies", [])
    projects = profile.get("projects", [])
    positions = profile.get("positions_history", [])
    languages = profile.get("personal_info", {}).get("languages", [])

    return {
        "years_total": years_total,
        "years_in_role": years_in_role,
        "num_skills": len(skills),
        "num_competencies": len(competencies),
        "num_projects": len(projects),
        "num_positions": len(positions),
        "num_languages": len(languages),
        "has_lead_comp": int(any("lead" in (c.get("name", "") or "").lower() for c in competencies)),
        "num_trainings": len(profile.get("training_history", [])) if "training_history" in profile else 0,
    }

# -------------------------
# Train model
# -------------------------
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

    joblib.dump({"model": clf, "scaler": scaler, "cols": list(X.columns)}, os.path.join(BASE_DIR, "psa_model.joblib"))
    return {"model": clf, "scaler": scaler, "cols": list(X.columns)}, {"df": df, "accuracy": acc, "auc": auc, "n": len(df)}

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
# Simple recommender
# -------------------------
DEFAULT_TRAINING_DB = {
    "cloud": ["Cloud Architecture Masterclass", "Infrastructure as Code (IaC)"],
    "leadership": ["Leading High Performance Teams", "Coaching Skills for Managers"],
}

def recommend_trainings_from_skills(profile: dict) -> Tuple[list, list]:
    skills = [(s.get("skill_name") or "").lower() for s in profile.get("skills", [])]
    matches, train = [], []
    for sk in skills:
        for key in DEFAULT_TRAINING_DB:
            if key in sk:
                train += DEFAULT_TRAINING_DB[key]
                matches.append(sk)
    return list(dict.fromkeys(train)), matches

# -------------------------
# Conversational assistant (OpenAI + fallback)
# -------------------------
def handle_conversation(profile: dict, message: str) -> Dict:
    client = get_openai_client()
    if not client:
        # fallback rule-based
        m = message.lower()
        if "stress" in m or "burnout" in m:
            return {"reply": "I'm sorry to hear that. Consider short breaks or contacting EAP for support."}
        if "career" in m or "promotion" in m:
            return {"reply": "I can help suggest career growth opportunities. Try asking about leadership or new skills."}
        if "training" in m or "learn" in m:
            t, matched = recommend_trainings_from_skills(profile)
            if t:
                return {"reply": f"Based on your skills ({', '.join(matched)}), you can explore: {', '.join(t)}"}
            return {"reply": "Tell me which skill you’d like to learn, and I’ll recommend a course."}
        return {"reply": "I can help with training, career advice, or wellbeing. Try asking about 'career' or 'training'."}

    try:
        name = profile.get("personal_info", {}).get("name", "the employee")
        role = profile.get("employment_info", {}).get("job_title", "employee")
        skills = [s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]

        system_prompt = f"""
        You are PSA's AI Career Assistant.
        Help employees like {name} ({role}) with wellbeing, growth, and skill development.
        Be empathetic, concise (max 6 sentences), and actionable.
        Skills: {', '.join(skills[:10])}.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"reply": f"(AI assistant error: {e})"}

# -------------------------
# Streamlit layout
# -------------------------
st.title("🚢 PSA — Future-Ready Workforce (ML + AI Assistant)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data & Model")
    employees = load_employee_json()
    st.write(f"Loaded {len(employees)} profiles.")
    if st.button("Train / Re-train Model"):
        with st.spinner("Training..."):
            model_pack, info = build_dataset_and_train()
        st.success("✅ Model trained.")
    else:
        model_pack, info = build_dataset_and_train()

with col2:
    st.header("Employee Browser & Assistant")
    emp_map = {e.get("employee_id"): e for e in employees}
    selected = st.selectbox("Select employee", [""] + list(emp_map.keys()))
    if selected:
        profile = emp_map[selected]
        name = profile.get("personal_info", {}).get("name", "")
        title = profile.get("employment_info", {}).get("job_title", "")
        st.subheader(f"{name} — {title}")

        res = predict_for_profile(model_pack, profile)
        st.write(f"Leadership potential: **{res['probability']:.1%}** (Predicted: {res['prediction']})")

        st.markdown("---")
        st.subheader("💬 Conversational Assistant")
        msg = st.text_input("Ask about career, training, or wellbeing:")
        if st.button("Ask Assistant"):
            reply = handle_conversation(profile, msg)
            st.write(reply["reply"])
