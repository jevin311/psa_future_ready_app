
# psa_future_ready_app.py
# Streamlit app: PSA Future-Ready Workforce (ML leadership predictor + recommender)
# Place this file in the same environment that has access to:
#   /mnt/data/Employee_Profiles.json
#   /mnt/data/Functions & Skills.xlsx
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

st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — ML Demo")

# -------------------------
# Robust file loaders
# -------------------------
# Base directory fallback
BASE_DIR = os.getcwd()

# Default file paths
EMPLOYEE_FILE = os.path.join(BASE_DIR, "Employee_Profiles.json")
FUNCTIONS_FILE = os.path.join(BASE_DIR, "Functions & Skills.xlsx")

# fallback to /mnt/data if files not found
if not os.path.isfile(EMPLOYEE_FILE):
    EMPLOYEE_FILE = "/mnt/data/Employee_Profiles.json"

if not os.path.isfile(FUNCTIONS_FILE):
    FUNCTIONS_FILE = "/mnt/data/Functions & Skills.xlsx"

# Optional debug info
st.write("BASE_DIR:", BASE_DIR)
st.write("EMPLOYEE_FILE exists:", os.path.isfile(EMPLOYEE_FILE))
st.write("FUNCTIONS_FILE exists:", os.path.isfile(FUNCTIONS_FILE))
@st.cache_data
def load_employee_json():
    if os.path.isfile(EMPLOYEE_FILE):
        with open(EMPLOYEE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    st.error(f"Cannot find Employee_Profiles.json at {EMPLOYEE_FILE}")
    return []

@st.cache_data
def load_functions_skills():
    if os.path.isfile(FUNCTIONS_FILE):
        xl = pd.read_excel(FUNCTIONS_FILE, sheet_name=None, engine="openpyxl")
        dfs = []
        for name, df in xl.items():
            df["__sheet__"] = name
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    st.warning(f"Cannot find Functions & Skills.xlsx at {FUNCTIONS_FILE}")
    return pd.DataFrame()


# -------------------------
# Heuristic label creation
# -------------------------
def derive_is_leader(profile: dict) -> int:
    """
    Heuristic to derive whether the employee is a leader (1) or not (0).
    Rules (ORed):
      - Current role_title contains Manager/Lead/Head/Director/Chief
      - Any past role_title contains Manager/Lead/Head/Director/Chief
      - Competencies include 'Leadership' or 'Stakeholder & Partnership Management' at Advanced
      - Projects where role contains 'Lead' or 'Program Lead'
    This is a heuristic ONLY to enable training when no ground-truth label exists.
    """
    def contains_lead_word(title):
        if not title:
            return False
        title_l = title.lower()
        for kw in ["manager", "lead", "head", "director", "chief"]:
            if kw in title_l:
                return True
        return False

    # check current position (positions_history entries with end==None considered current)
    ph = profile.get("positions_history", []) or []
    for pos in ph:
        end = None
        per = pos.get("period", {}) or {}
        end = per.get("end")
        # treat None or null-like as current
        if end in (None, "", "null"):
            if contains_lead_word(pos.get("role_title", "")):
                return 1

    # check any history
    for pos in ph:
        if contains_lead_word(pos.get("role_title", "")):
            return 1

    # competencies: look for leadership
    for comp in profile.get("competencies", []) or []:
        name = (comp.get("name", "") or "").lower()
        level = (comp.get("level", "") or "").lower()
        if "leadership" in name or "stakeholder" in name:
            # treat advanced stakeholder/leadership as signal
            if level in ("advanced", "expert"):
                return 1

    # projects role
    for proj in profile.get("projects", []) or []:
        role = proj.get("role", "") or ""
        if contains_lead_word(role):
            return 1

    # otherwise not leader
    return 0

# -------------------------
# Feature engineering
# -------------------------
def profile_to_features(profile: dict) -> Dict:
    # numeric features we will compute:
    # - years_total (hire_date to last_updated)
    # - years_in_role (in_role_since to last_updated)
    # - num_skills
    # - num_competencies
    # - num_projects
    # - num_positions
    # - num_languages
    # - avg_project_duration_months
    # - has_leadership_competency (binary)
    pi = profile.get("personal_info", {}) or {}
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

    num_skills = len(skills)
    num_comp = len(competencies)
    num_projects = len(projects)
    num_positions = len(positions)
    num_languages = len(languages)

    # avg project duration in months
    durations = []
    for p in projects:
        ps = (p.get("period", {}) or {}).get("start")
        pe = (p.get("period", {}) or {}).get("end")
        s = parse_date(ps)
        e = parse_date(pe) if pe else now
        if s and e:
            durations.append((e - s).days / 30.0)
    avg_proj_dur = float(np.mean(durations)) if durations else 0.0

    has_lead_comp = int(any("lead" in (c.get("name","").lower()) or "leadership" in c.get("name","").lower() for c in competencies))

    # training history length (approx)
    num_trainings = 0
    if "training_history" in profile:
        th = profile.get("training_history") or []
        if isinstance(th, list):
            num_trainings = len(th)
        else:
            num_trainings = 0

    features = {
        "years_total": float(years_total),
        "years_in_role": float(years_in_role),
        "num_skills": int(num_skills),
        "num_competencies": int(num_comp),
        "num_projects": int(num_projects),
        "num_positions": int(num_positions),
        "num_languages": int(num_languages),
        "avg_proj_dur_months": float(avg_proj_dur),
        "has_lead_comp": int(has_lead_comp),
        "num_trainings": int(num_trainings),
    }
    return features

# -------------------------
# Build dataset and train
# -------------------------
@st.cache_resource
def build_dataset_and_train(emp_json_path="Employee_Profiles.json", retrain=False) -> Tuple[dict, dict]:
    # returns: model_pack, dataset_info
    employees = load_employee_json(emp_json_path)
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

    # Prepare X,y
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    # scaler + trainer
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # simple logistic regression
    clf = LogisticRegression(max_iter=1000)
    # If dataset is very small, fit on full data but still compute CV metrics; here we do train/test split if possible
    if len(df) >= 6 and len(y.unique()) > 1:
        try:
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        try:
            auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
        except Exception:
            auc = None
    else:
        # fit on full data when too small
        clf.fit(Xs, y)
        acc = None
        auc = None

    model_pack = {"model": clf, "scaler": scaler, "feature_cols": feature_cols}
    dataset_info = {"df": df, "accuracy": acc, "auc": auc, "n": len(df)}
    # persist model to /mnt/data for reuse
    try:
        joblib.dump(model_pack, "/mnt/data/psa_leadership_model.joblib")
    except Exception:
        pass
    return model_pack, dataset_info

# -------------------------
# Prediction & explainability
# -------------------------
def predict_for_profile(model_pack: dict, profile: dict) -> Dict:
    feat = profile_to_features(profile)
    feature_cols = model_pack["feature_cols"]
    X_row = np.array([feat[c] for c in feature_cols], dtype=float).reshape(1, -1)
    Xs = model_pack["scaler"].transform(X_row)
    prob = float(model_pack["model"].predict_proba(Xs)[0,1])
    pred = int(model_pack["model"].predict(Xs)[0])
    # compute per-feature contribution using coef * (x - mean) approximation
    try:
        coefs = model_pack["model"].coef_[0]
        # standardised feature values:
        std_vals = Xs.flatten()
        contributions = {feature_cols[i]: float(coefs[i] * std_vals[i]) for i in range(len(feature_cols))}
        # sort top contributors
        contrib_sorted = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    except Exception:
        contrib_sorted = []
    return {
        "probability": prob,
        "prediction": pred,
        "features": feat,
        "contributions": contrib_sorted
    }

# -------------------------
# Simple recommender mapping skills -> trainings
# -------------------------
DEFAULT_TRAINING_DB = {
    # skill lower -> recommended courses
    "cloud architecture": ["Cloud Architecture Masterclass", "Designing Reliable Cloud Systems"],
    "cloud devops & automation": ["Infrastructure as Code (IaC) with Terraform", "CI/CD for Cloud"],
    "securing cloud infrastructure": ["Cloud Security Fundamentals", "Zero Trust Architecture"],
    "vulnerability management": ["Vulnerability Management Essentials", "Pentesting Foundations"],
    "network security management": ["Network Security Operations", "Firewall & IDS/IPS Fundamentals"],
    "financial planning and analysis": ["Advanced FP&A", "Driver-based Forecasting"],
    "financial modeling": ["Financial Modeling Bootcamp", "Excel for Finance"],
    "treasury": ["Treasury Management Fundamentals", "Cash & Liquidity Management"],
    "talent management": ["Talent Management Strategy", "Succession Planning"],
    "leadership development": ["Leading High Performance Teams", "Coaching Skills for Managers"]
}

def recommend_trainings_from_skills(profile: dict, functions_df: pd.DataFrame) -> Tuple[list, list]:
    skills = [ (s.get("skill_name") or "").lower() for s in profile.get("skills", []) ]
    trainings = []
    matched_skills = []
    for sk in skills:
        if not sk:
            continue
        if sk in DEFAULT_TRAINING_DB:
            trainings += DEFAULT_TRAINING_DB[sk]
            matched_skills.append(sk)
        else:
            # fuzzy match: check if any key is substring
            for key in DEFAULT_TRAINING_DB:
                if key in sk or sk in key:
                    trainings += DEFAULT_TRAINING_DB[key]
                    matched_skills.append(sk)
                    break
    # deduplicate
    trainings = list(dict.fromkeys(trainings))
    return trainings, matched_skills

# -------------------------
# Conversation handler (rule-based)
# -------------------------
def handle_conversation(profile: dict, message: str) -> Dict:
    m = (message or "").lower()
    if any(w in m for w in ["stress","stressed","anxious","anxiety","burnout"]):
        return {"reply": "I'm sorry you're feeling this way. Consider reaching out to your manager or EAP. Short breathing breaks (5-min) and stepping away help. Would you like resources on EAP or a manager check-in template?"}
    if any(w in m for w in ["career","promotion","role","change","move"]):
        return {"reply": "I can generate a personalised career pathway for you. Click 'Recommend Trainings' for suggested upskilling and internal mobility options."}
    if any(w in m for w in ["training","learn","skill"]):
        trainings, matched = recommend_trainings_from_skills(profile, pd.DataFrame())
        if trainings:
            return {"reply": f"Recommended trainings based on your skills ({', '.join(matched)}): {', '.join(trainings[:6])}"}
        else:
            return {"reply": "Tell me which skill you'd like to build and I'll suggest courses."}
    return {"reply": "I can help with career guidance, training suggestions, or wellbeing tips. Try asking about 'training', 'career', or 'stress'."}

# -------------------------
# Streamlit layout
# -------------------------
st.title("PSA — Future-Ready Workforce (Streamlit ML Demo)")
st.markdown(
    """
This demo trains a **logistic regression** model to predict leadership potential using heuristically
derived labels from sample employee profiles. Labels are *derived* (not HR ground-truth) by rules
(e.g., past/current manager roles, leadership competencies). Use as prototype only — replace labels
with HR-provided outcomes for production.
"""
)

# left column: data + training controls
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data & Model")
    employees = load_employee_json()
    st.write(f"Loaded {len(employees)} employee profiles.")
    functions_df = load_functions_skills()
    if not functions_df.empty:
        st.write("Functions & Skills taxonomy loaded.")
    # training / rebuild button
    if st.button("Train / Re-train Leadership Model"):
        with st.spinner("Training model..."):
            model_pack, info = build_dataset_and_train(retrain=True)
        st.success("Training finished.")
        if info.get("accuracy") is not None:
            st.write(f"Test accuracy: **{info['accuracy']:.3f}** | AUC: **{info['auc']}**")
        else:
            st.write("Small dataset: model trained on full data (no holdout). Provide more labelled profiles for robust evaluation.")
    else:
        model_pack, info = build_dataset_and_train()

    if model_pack is None:
        st.error("Model not available — check that Employee_Profiles.json loaded.")
    else:
        st.write(f"Model trained on n = {info.get('n', '?')} profiles.")
        if info.get("accuracy") is not None:
            st.metric("Test accuracy", value=f"{info['accuracy']:.3f}")

    st.markdown("**Notes on labels:** The app heuristically labels employees as 'leader' if their role/title/competencies indicate managerial experience. Replace with HR ground-truth for production.")

with col2:
    st.header("Employee Browser & Prediction")
    # sidebar-like selection
    emp_map = {e.get("employee_id"): e for e in employees}
    emp_options = list(emp_map.keys())
    selected_emp_id = st.selectbox("Select employee", [""] + emp_options)
    if selected_emp_id:
        profile = emp_map[selected_emp_id]
        # show profile summary
        st.subheader(f"{profile.get('personal_info',{}).get('name', '')} — {profile.get('employment_info',{}).get('job_title','')}")
        left, right = st.columns(2)
        with left:
            st.write("**Skills**")
            skills = [s.get("skill_name") for s in profile.get("skills",[])]
            st.write(", ".join([s for s in skills if s]) if skills else "—")
            st.write("**Competencies**")
            comps = [f"{c.get('name')} ({c.get('level')})" for c in profile.get("competencies",[])]
            st.write(", ".join([c for c in comps if c]) if comps else "—")
            st.write("**Languages**")
            langs = [f"{l.get('language')} ({l.get('proficiency')})" for l in profile.get("personal_info",{}).get("languages",[])]
            st.write(", ".join([la for la in langs if la]) if langs else "—")
        with right:
            st.write("**Positions History (most recent first)**")
            ph = profile.get("positions_history", []) or []
            for pos in ph:
                st.write(f"- {pos.get('role_title')} ({pos.get('period',{}).get('start')} → {pos.get('period',{}).get('end') or 'Present'})")
            st.write("**Projects**")
            for proj in profile.get("projects", []) or []:
                st.write(f"- {proj.get('project_name')} — {proj.get('role')}")
        st.markdown("---")
        # Predict leadership
        if model_pack:
            res = predict_for_profile(model_pack, profile)
            prob = res["probability"]
            st.subheader("Leadership Prediction")
            st.write(f"Predicted leadership probability: **{prob:.2%}**")
            st.write(f"Binary prediction (threshold 0.5): **{res['prediction']}** (1 = leader)")
            st.markdown("**Top feature contributions (standardised coef × feature)**")
            for feat, val in res["contributions"][:6]:
                st.write(f"- {feat}: {val:.3f}")

        # Recommender
        st.markdown("---")
        st.subheader("Recommended Trainings & Pathways")
        trainings, matched = recommend_trainings_from_skills(profile, functions_df)
        if trainings:
            st.write("Suggested trainings:")
            for t in trainings:
                st.write(f"- {t}")
        else:
            st.write("No direct mapping found in the small internal database. Consider cross-functional learning or leadership courses.")
        # Internal mobility hint (very simple)
        job_title = (profile.get("employment_info", {}).get("job_title", "") or "").lower()
        if "analyst" in job_title:
            st.info("Suggested career ladder: Senior Analyst → Principal Analyst → Team Lead / Manager (consider leadership tracks).")
        elif "engineer" in job_title or "architect" in job_title:
            st.info("Suggested career ladder: Senior Engineer → Tech Lead → Engineering Manager / Principal Engineer.")
        elif "manager" in job_title or "lead" in job_title:
            st.info("Already on leadership track — consider executive coaching or strategic leadership courses.")
        else:
            st.info("Consider cross-training and stretch assignments to broaden exposure.")

        # conversational panel
        st.markdown("---")
        st.subheader("Assistant (conversational)")
        question = st.text_input("Ask the assistant about career, training, or wellbeing")
        if st.button("Ask"):
            if not question.strip():
                st.warning("Type a question first.")
            else:
                reply = handle_conversation(profile, question)
                st.write(reply.get("reply"))

# bottom: dataset view & export
st.markdown("---")
st.header("Dataset (derived features & labels)")
if info.get("n", 0) > 0:
    df_view = info["df"].reset_index()
    st.dataframe(df_view)
    if st.button("Download derived dataset (CSV)"):
        csv = df_view.to_csv(index=False).encode("utf-8")
        st.download_button("Click to download CSV", data=csv, file_name="psa_derived_dataset.csv", mime="text/csv")

st.markdown(
    """
**Next steps & cautions**
- Replace heuristic labels with HR-provided leadership outcomes for higher-fidelity models.
- Add privacy / data governance before deploying (PII must be protected).
- Consider richer features (360 feedback, engagement surveys, assessment scores) and explainability (SHAP) for production.
"""
)

