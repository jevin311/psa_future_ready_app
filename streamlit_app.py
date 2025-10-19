# psa_future_ready_app.py
# PSA Future-Ready Workforce — End-to-End AI + ML App (Robust Version)
# Place in same folder as:
#   - Employee_Profiles.json
#   - Functions & Skills.xlsx
# Run:
#   streamlit run psa_future_ready_app.py

import os
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Optional OpenAI (only used if a key is provided)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — AI Platform")
BASE_DIR = Path(__file__).resolve().parent
EMP_PATH = BASE_DIR / "Employee_Profiles.json"
FUNC_PATH = BASE_DIR / "Functions & Skills.xlsx"
MODEL_PATH = BASE_DIR / "psa_leadership_model.joblib"

PSA_VALUES = [
    "Collaboration", "Innovation", "Customer Focus",
    "Integrity", "Accountability", "Sustainability"
]

# ===============================
# OpenAI API Key (embedded)
# ===============================
OPENAI_API_KEY = "d18b91ff44e44f55bc3d48e6a085160c"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("### 🧭 Accessibility")
if st.sidebar.checkbox("High contrast mode"):
    st.markdown("<style>.stApp{background-color:black;color:white;}</style>", unsafe_allow_html=True)
if st.sidebar.checkbox("Large font size"):
    st.markdown("<style>.stApp *{font-size:18px !important;}</style>", unsafe_allow_html=True)

# ===============================
# HELPERS
# ===============================
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def contains_lead_word(title: str) -> bool:
    if not title:
        return False
    t = str(title).lower()
    for kw in ["manager", "lead", "head", "director", "chief", "vp", "principal"]:
        if kw in t:
            return True
    return False

def derive_is_leader(profile: Dict) -> int:
    # Current role
    if contains_lead_word(profile.get("employment_info", {}).get("job_title", "")):
        return 1
    # Past roles
    for pos in (profile.get("positions_history") or []):
        if contains_lead_word(pos.get("role_title", "")):
            return 1
    # Competencies
    for c in (profile.get("competencies") or []):
        if "leadership" in str(c.get("name", "")).lower() and (c.get("level") or 0) > 3:
            return 1
    return 0

def parse_date(s):
    if not s or pd.isna(s):
        return None
    if isinstance(s, datetime):
        return s
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            continue
    return None

def profile_to_features(profile: Dict) -> Dict:
    ei = profile.get("employment_info", {}) or {}
    now = datetime.utcnow()
    hire = parse_date(ei.get("hire_date"))
    in_role = parse_date(ei.get("in_role_since"))

    years_total = (now - hire).days / 365.25 if hire else 0.0
    years_in_role = (now - in_role).days / 365.25 if in_role else 0.0

    skills = profile.get("skills", []) or []
    comps = profile.get("competencies", []) or []
    projs = profile.get("projects", []) or []

    # Performance data
    perf_reviews = profile.get("performance_reviews", []) or []
    avg_performance_rating = float(np.mean([r.get("rating", 3) for r in perf_reviews])) if perf_reviews else 3.0

    # Engagement data
    eng_scores = profile.get("engagement_scores", []) or []
    latest_engagement_score = float((eng_scores[-1].get("score", 70)) if eng_scores else 70)

    # Behavioral/development data
    num_trainings = len(profile.get("training_history", []) or [])
    num_cross_functional_projects = len([p for p in projs if str(p.get("scope")).lower() == "cross-functional"])

    return {
        "years_total": years_total,
        "years_in_role": years_in_role,
        "num_skills": len(skills),
        "num_competencies": len(comps),
        "num_projects": len(projs),
        "avg_performance_rating": avg_performance_rating,
        "latest_engagement_score": latest_engagement_score,
        "num_trainings": num_trainings,
        "num_cross_functional_projects": num_cross_functional_projects,
    }

# ===============================
# DATA LOADING (DEFENSIVE)
# ===============================
@st.cache_data
def load_employee_json() -> List[Dict]:
    if not EMP_PATH.exists():
        st.error(f"Employee_Profiles.json not found at {EMP_PATH}")
        return []
    try:
        return json.loads(EMP_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Could not parse Employee_Profiles.json: {e}")
        return []

def _normalize_skill_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)
    candidates = {"skill", "skills", "skill name", "skill names",
                  "competency", "competencies", "key skill", "key skills"}
    skill_col = next((c for c in df.columns if c in candidates), None)
    if skill_col is None:
        text_cols = [c for c in df.columns if df[c].dtype == "object"]
        skill_col = text_cols[0] if text_cols else None
    if skill_col and skill_col != "Skill":
        df.rename(columns={skill_col: "Skill"}, inplace=True)
    return df

@st.cache_data
def load_functions_skills() -> pd.DataFrame:
    if not FUNC_PATH.exists():
        st.error(f"Functions & Skills.xlsx not found at {FUNC_PATH}")
        return pd.DataFrame()
    try:
        try:
            xl = pd.read_excel(FUNC_PATH, sheet_name=None, engine="openpyxl")
        except Exception:
            xl = pd.read_excel(FUNC_PATH, sheet_name=None)

        frames = []
        for name, df in xl.items():
            df = _normalize_skill_columns(df)
            if df.empty:
                continue
            df["__sheet__"] = name
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        if "Skill" not in out.columns:
            text_cols = [c for c in out.columns if c not in {"__sheet__"} and out[c].dtype == "object"]
            if text_cols:
                melted = out.melt(id_vars="__sheet__", value_vars=text_cols, var_name="Field", value_name="Skill")
                out = melted[["__sheet__", "Skill"]]
            else:
                out["Skill"] = None
        return out
    except Exception as e:
        st.warning(f"Error reading Functions & Skills.xlsx: {e}")
        return pd.DataFrame()

# ===============================
# MODEL PIPELINE
# ===============================
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
    df = df.dropna(subset=["label"])
    if df.empty:
        return None, {}

    X = df[[c for c in df.columns if c != "label"]].fillna(0).astype(float)
    y = df["label"].astype(int)

    class_weight = "balanced" if (len(y) > 0 and y.mean() < 0.4) else None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)

    acc, auc = None, None
    if len(df) > 10 and len(y.unique()) > 1:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                Xs, y, test_size=0.25, random_state=42, stratify=y
            )
            clf.fit(X_train, y_train)
            acc = float(accuracy_score(y_test, clf.predict(X_test)))
            auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        except ValueError:
            clf.fit(Xs, y)
    else:
        clf.fit(Xs, y)

    try:
        joblib.dump({"model": clf, "scaler": scaler, "feature_cols": list(X.columns)}, MODEL_PATH)
    except Exception:
        pass

    info = {"df": df, "accuracy": acc, "auc": auc}
    model_pack = {"model": clf, "scaler": scaler, "feature_cols": list(X.columns)}
    return model_pack, info

def predict_for_profile(model_pack: Dict, profile: Dict) -> float:
    feat = profile_to_features(profile)
    X_row = np.array([[feat.get(c, 0) for c in model_pack["feature_cols"]]], dtype=float)
    Xs = model_pack["scaler"].transform(X_row)
    prob = float(model_pack["model"].predict_proba(Xs)[0, 1])
    return prob

@st.cache_data
def get_all_predictions(_model_pack, employees):
    if not _model_pack:
        return {}
    return {e.get("employee_id"): predict_for_profile(_model_pack, e) for e in employees}

# ===============================
# AI HELPERS
# ===============================
def handle_conversation(profile: Dict, message: str) -> Dict:
    client = get_openai_client()
    name = profile.get("personal_info", {}).get("name", "the employee")
    role = profile.get("employment_info", {}).get("job_title", "employee")
    if not client:
        return {"reply": f"Hi {name}, I'm your career assistant. I can help with growth, wellbeing, and engagement. What would you like to talk about today?"}

    sys_prompt = f"""
    You are PSA's Career & Wellbeing Assistant. You are empathetic, supportive, and action-oriented.
    You are speaking to: {name}, who is a {role} at PSA.
    Your primary goal is to support them in:
    1) Continuous Development  2) Mental Well-being  3) Engagement
    Respond concisely (2–3 sentences) and end with a supportive tone or open question.
    """

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=250
        )
        return {"reply": r.choices[0].message.content.strip()}
    except Exception as e:
        st.error(f"AI error: {e}")
        return {"reply": "I seem to be having trouble connecting. Please check the API key."}

# ===============================
# (The rest of your code remains identical to your original UI & tabs)
# ===============================
