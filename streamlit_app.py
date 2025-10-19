# psa_future_ready_app.py
# Streamlit app: PSA Future-Ready Workforce (ML leadership predictor + recommender + AI assistant)
# Place this file in the same repository as:
#   Employee_Profiles.json
#   Functions & Skills.xlsx
#
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
from openai import OpenAI  # ✅ new import for AI assistant

# initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — ML + AI Assistant")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Robust file loaders
# -------------------------
@st.cache_data
def load_employee_json():
    """Load Employee_Profiles.json from BASE_DIR"""
    p = os.path.join(BASE_DIR, "Employee_Profiles.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            st.success(f"Loaded employee profiles from {p}")
            return data
        except Exception as e:
            st.warning(f"Found {p} but could not load JSON: {e}")
    st.error(f"Could not find Employee_Profiles.json at {p}")
    return []

@st.cache_data
def load_functions_skills():
    """Load Functions & Skills.xlsx from BASE_DIR"""
    p = os.path.join(BASE_DIR, "Functions & Skills.xlsx")
    if os.path.isfile(p):
        try:
            xl = pd.read_excel(p, sheet_name=None, engine="openpyxl")
            dfs = []
            for name, df in xl.items():
                df["__sheet__"] = name
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)
            st.success(f"Loaded Functions & Skills taxonomy from {p}")
            return df_all
        except Exception as e:
            st.warning(f"Found {p} but could not read Excel: {e}")
    st.warning(f"Could not find Functions & Skills.xlsx at {p}. Continuing without it.")
    return pd.DataFrame()

# -------------------------
# Heuristic label creation
# -------------------------
def derive_is_leader(profile: dict) -> int:
    """Heuristic to derive whether employee is a leader (1) or not (0)"""
    def contains_lead_word(title):
        if not title:
            return False
        title_l = title.lower()
        for kw in ["manager", "lead", "head", "director", "chief"]:
            if kw in title_l:
                return True
        return False

    ph = profile.get("positions_history", []) or []
    for pos in ph:
        per = pos.get("period", {}) or {}
        end = per.get("end")
        if end in (None, "", "null"):
            if contains_lead_word(pos.get("role_title", "")):
                return 1

    for pos in ph:
        if contains_lead_word(pos.get("role_title", "")):
            return 1

    for comp in profile.get("competencies", []) or []:
        name = (comp.get("name", "") or "").lower()
        level = (comp.get("level", "") or "").lower()
        if "leadership" in name or "stakeholder" in name:
            if level in ("advanced", "expert"):
                return 1

    for proj in profile.get("projects", []) or []:
        role = proj.get("role", "") or ""
        if contains_lead_word(role):
            return 1

    return 0

# -------------------------
# Feature engineering
# -------------------------
def profile_to_features(profile: dict) -> Dict:
    ei = profile.get("employment_info", {}) or {}
    now_str = ei.get("last_updated") or datetime.now().strftime("%Y-%m-%d")
    try:
        now = datetime.fromisoformat(now_str)
    except Exception:
        try:
            now = datetime.strptime(now_str, "%Y-%m-%d")
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
    competencies = profile.get("competencies", []) or []
    projects = profile.get("projects", []) or []
    positions = profile.get("positions_history", []) or []
    languages = profile.get("personal_info", {}).get("languages", []) or []

    features = {
        "years_total": (now - hire).days / 365.25 if hire else 0.0,
        "years_in_role": (now - in_role_since).days / 365.25 if in_role_since else 0.0,
        "num_skills": len(skills),
        "num_competencies": len(competencies),
        "num_projects": len(projects),
        "num_positions": len(positions),
        "num_languages": len(languages),
        "avg_proj_dur_months": float(np.mean([
            ((datetime.fromisoformat(p.get("period", {}).get("end", now.strftime("%Y-%m-%d"))) -
              datetime.fromisoformat(p.get("period", {}).get("start", now.strftime("%Y-%m-%d")))).days / 30)
            for p in projects if p.get("period", {}).get("start")
        ])) if projects else 0.0,
        "has_lead_comp": int(any("lead" in (c.get("name","").lower()) or "leadership" in c.get("name","").lower() for c in competencies)),
        "num_trainings": len(profile.get("training_history", [])) if "training_history" in profile else 0,
    }
    return features

# -------------------------
# Build dataset and train
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
        row = feat.copy()
        row["employee_id"] = p.get("employee_id")
        row["label"] = label
        rows.append(row)
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
        acc = float(accuracy_score(y_test, y_pred))
        try:
            auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        except Exception:
            auc = None
    else:
        clf.fit(Xs, y)
        acc = None
        auc = None

    model_pack = {"model": clf, "scaler": scaler, "feature_cols": feature_cols}
    dataset_info = {"df": df, "accuracy": acc, "auc": auc, "n": len(df)}
    joblib.dump(model_pack, os.path.join(BASE_DIR, "psa_leadership_model.joblib"))
    return model_pack, dataset_info

# -------------------------
# Prediction & explainability
# -------------------------
def predict_for_profile(model_pack: dict, profile: dict) -> Dict:
    feat = profile_to_features(profile)
    X_row = np.array([feat[c] for c in model_pack["feature_cols"]], dtype=float).reshape(1, -1)
    Xs = model_pack["scaler"].transform(X_row)
    prob = float(model_pack["model"].predict_proba(Xs)[0, 1])
    pred = int(model_pack["model"].predict(Xs)[0])
    coefs = model_pack["model"].coef_[0]
    contributions = {model_pack["feature_cols"][i]: float(coefs[i] * Xs.flatten()[i]) for i in range(len(coefs))}
    contrib_sorted = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    return {"probability": prob, "prediction": pred, "features": feat, "contributions": contrib_sorted}

# -------------------------
# Simple recommender
# -------------------------
DEFAULT_TRAINING_DB = {
    "cloud architecture": ["Cloud Architecture Masterclass", "Designing Reliable Cloud Systems"],
    "cloud devops & automation": ["Infrastructure as Code (IaC) with Terraform", "CI/CD for Cloud"],
    "leadership development": ["Leading High Performance Teams", "Coaching Skills for Managers"]
}

def recommend_trainings_from_skills(profile: dict, functions_df: pd.DataFrame) -> Tuple[list, list]:
    skills = [(s.get("skill_name") or "").lower() for s in profile.get("skills", [])]
    trainings, matched = [], []
    for sk in skills:
        for key in DEFAULT_TRAINING_DB:
            if key in sk or sk in key:
                trainings += DEFAULT_TRAINING_DB[key]
                matched.append(sk)
                break
    return list(dict.fromkeys(trainings)), matched

# -------------------------
# Conversational assistant (OpenAI + fallback)
# -------------------------
def handle_conversation(profile: dict, message: str) -> Dict:
    """
    Uses OpenAI GPT model for contextual conversational support.
    Falls back to rule-based assistant if no API key is set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # fallback logic
        m = (message or "").lower()
        if any(w in m for w in ["stress", "anxious", "burnout"]):
            return {"reply": "I'm sorry you're feeling this way. Try short breaks and talk with your manager. I can also suggest EAP resources."}
        if any(w in m for w in ["career", "promotion", "move"]):
            return {"reply": "I can suggest growth opportunities and training plans. Try asking 'What skills should I build for leadership?'"}
        if any(w in m for w in ["training", "learn", "skill"]):
            trainings, matched = recommend_trainings_from_skills(profile, pd.DataFrame())
            if trainings:
                return {"reply": f"Recommended trainings based on your skills ({', '.join(matched)}): {', '.join(trainings)}"}
            return {"reply": "Tell me which skill you'd like to build, and I'll suggest relevant courses."}
        return {"reply": "I can help with career guidance, training suggestions, or wellbeing tips. Try asking about 'training' or 'career'."}

    try:
        name = profile.get("personal_info", {}).get("name", "the employee")
        role = profile.get("employment_info", {}).get("job_title", "employee")
        skills = [s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]

        system_prompt = f"""
        You are PSA's AI Career Assistant. 
        You help employees like {name} ({role}) with wellbeing, growth, and internal mobility.
        Be empathetic, professional, and concise (max 6 sentences).
        Known skills: {', '.join(skills[:10])}.
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
        ai_reply = response.choices[0].message.content.strip()
        return {"reply": ai_reply}

    except Exception as e:
        return {"reply": f"(AI assistant error: {e})"}

# -------------------------
# Streamlit layout
# -------------------------
st.title("PSA — Future-Ready Workforce (Streamlit ML + AI Assistant)")
st.markdown("""
Empower PSA employees with AI-driven insights on career growth, well-being, and leadership potential.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data & Model")
    employees = load_employee_json()
    st.write(f"Loaded {len(employees)} profiles.")
    functions_df = load_functions_skills()

    if st.button("Train / Re-train Leadership Model"):
        with st.spinner("Training model..."):
            model_pack, info = build_dataset_and_train()
        st.success("Training finished.")
        st.write(f"Accuracy: {info.get('accuracy')}, AUC: {info.get('auc')}")
    else:
        model_pack, info = build_dataset_and_train()

with col2:
    st.header("Employee Browser & Prediction")
    emp_map = {e.get("employee_id"): e for e in employees}
    emp_ids = list(emp_map.keys())
    emp_id = st.selectbox("Select employee", [""] + emp_ids)

    if emp_id:
        profile = emp_map[emp_id]
        st.subheader(f"{profile.get('personal_info',{}).get('name','')} — {profile.get('employment_info',{}).get('job_title','')}")
        res = predict_for_profile(model_pack, profile)
        st.write(f"Leadership probability: **{res['probability']:.2%}** (Prediction: {res['prediction']})")

        st.markdown("---")
        st.subheader("AI Assistant (Conversational)")
        question = st.text_input("Ask about career, wellbeing, or upskilling:")
        if st.button("Ask"):
            reply = handle_conversation(profile, question)
            st.write(reply["reply"])
