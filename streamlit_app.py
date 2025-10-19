
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
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input(
    "🔑 OpenAI API Key",
    type="password",
    help="Optional. Paste your OpenAI key to enable AI features."
)
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

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
# Optional OpenAI (only used if a key is provided)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
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
    """Normalize headers, try to identify a 'Skill' column, and name it 'Skill'."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Strip + keep original names for display; also create a lower map for renaming
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # Identify probable skill column
    candidates = {
        "skill", "skills", "skill name", "skill names",
        "competency", "competencies", "key skill", "key skills"
    }
    skill_col = next((c for c in df.columns if c in candidates), None)

    if skill_col is None:
        # Fallback: pick a textual column that looks dense enough
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
        # Try openpyxl engine; fall back to default if not available
        try:
            xl = pd.read_excel(FUNC_PATH, sheet_name=None, engine="openpyxl")
        except Exception:
            xl = pd.read_excel(FUNC_PATH, sheet_name=None)

        frames = []
        for name, df in xl.items():
            df = _normalize_skill_columns(df)
            if df.empty:
                continue
            df["__sheet__"] = name  # keep function/department name
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)

        # Ensure we have a 'Skill' column; if not, attempt to melt textual columns
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

    # Persist
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
        return {
            "reply": f"Hi {name}, I'm your career assistant. I can help with growth, wellbeing, and engagement. "
                     f"What would you like to talk about today?"
        }

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

def _summarise_functions_for_prompt(all_functions_df: pd.DataFrame) -> str:
    """Return a compact bullet list: Function + up to 5 unique skills."""
    if all_functions_df is None or all_functions_df.empty or "__sheet__" not in all_functions_df.columns:
        return "Not available."
    df = all_functions_df.copy()
    if "Skill" not in df.columns:
        # try to melt textual columns into a 'Skill' column
        text_cols = [c for c in df.columns if c not in {"__sheet__"} and df[c].dtype == "object"]
        if text_cols:
            df = df.melt(id_vars="__sheet__", value_vars=text_cols, var_name="Field", value_name="Skill")[["__sheet__", "Skill"]]
        else:
            df["Skill"] = None

    if "Skill" not in df.columns:
        return "Not available."

    tmp = (
        df.dropna(subset=["Skill"])
          .groupby("__sheet__")["Skill"]
          .apply(lambda x: ", ".join(pd.Series(x).dropna().astype(str).unique()[:5]))
          .reset_index()
    )
    if tmp.empty:
        return "Not available."
    return "\n".join(
        f"- Function: {row['__sheet__']}, Key Skills: {row['Skill']}"
        for _, row in tmp.iterrows()
    )

def generate_career_pathway(profile: Dict, leadership_prob: float, all_functions_df: pd.DataFrame) -> Dict:
    client = get_openai_client()
    name = profile.get("personal_info", {}).get("name", "the employee")
    role = profile.get("employment_info", {}).get("job_title", "employee")
    skills = [s.get("skill_name") for s in (profile.get("skills") or []) if s.get("skill_name")]
    function_list = _summarise_functions_for_prompt(all_functions_df)

    if not client:
        return {
            "ai_reply": (
                f"As a {role}, consider next steps like Senior {role}, Team Lead, or Manager roles. "
                f"Strengthen leadership, stakeholder management, and innovation. Explore cross-functional projects "
                f"and internal postings aligned to your interests."
            )
        }

    sys_prompt = f"""
    You are PSA's AI Career Advisor.
    Generate a concise, actionable 3-step career plan.
    Employee: {name}
    Current Role: {role}
    Current Skills: {", ".join(skills[:15])}
    Predicted Leadership Potential: {leadership_prob:.0%}

    Available PSA Functions & Key Skills:
    {function_list}

    For each step, include:
    1) Future Role
    2) Skill Gap & Upskilling (2–3 items with concrete actions)
    3) Internal Mobility (team/project suggestions)
    Be optimistic, specific, and aligned with PSA values.
    """

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Generate a personalized career pathway for {name}."},
            ],
            temperature=0.8,
            max_tokens=500
        )
        return {"ai_reply": r.choices[0].message.content.strip()}
    except Exception as e:
        st.error(f"AI error: {e}")
        return {"ai_reply": f"(Error generating pathway: {e})"}

@st.cache_data
def find_mentors(current_profile: Dict, all_employees: List[Dict], all_predictions: Dict[str, float], min_prob: float = 0.6):
    mentors = []
    current_id = current_profile.get("employee_id")
    current_skills = set([s.get("skill_name") for s in (current_profile.get("skills") or []) if s.get("skill_name")])
    current_dept = (current_profile.get("employment_info") or {}).get("department")

    for emp in all_employees:
        emp_id = emp.get("employee_id")
        if emp_id == current_id:
            continue
        prob = all_predictions.get(emp_id, 0.0)
        if prob >= min_prob:
            emp_skills = set([s.get("skill_name") for s in (emp.get("skills") or []) if s.get("skill_name")])
            emp_dept = (emp.get("employment_info") or {}).get("department")
            skill_overlap = len(current_skills.intersection(emp_skills))
            dept_match = (emp_dept == current_dept)
            if dept_match or skill_overlap > 1:
                mentors.append({
                    "name": (emp.get("personal_info") or {}).get("name"),
                    "role": (emp.get("employment_info") or {}).get("job_title"),
                    "department": emp_dept,
                    "prob": prob,
                    "skill_overlap": skill_overlap,
                    "id": emp_id
                })
    return sorted(mentors, key=lambda x: (x["prob"], x["skill_overlap"]), reverse=True)

# ===============================
# UI
# ===============================
st.title("🚀 PSA Future-Ready Workforce — AI Platform")
st.write("AI-driven leadership prediction, personalised career pathways, conversational assistant, mentorship, and recognition system.")

employees = load_employee_json()
functions_df = load_functions_skills()
model_pack, info = build_dataset_and_train()
all_predictions = get_all_predictions(model_pack, employees)

if not employees or not model_pack:
    st.error("Failed to load employee data or train model. Please check file paths and data.")
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.header("📊 Model Overview")
    if info.get("accuracy") is not None:
        st.metric("Model Accuracy", f"{info['accuracy']:.3f}")
    if info.get("auc") is not None:
        st.metric("ROC AUC", f"{info['auc']:.3f}")

    if info.get("df") is not None:
        st.write("Features used in model:")
        safe_cols = [c for c in info["df"].columns if c != "label"]
        st.dataframe(pd.DataFrame({"feature": safe_cols}), height=220)
    st.caption("Model trained on employee profile data — performance, engagement, tenure, and project signals.")

with right:
    st.header("👥 Employee Explorer")

    emp_map = {e.get("employee_id"): e for e in employees}
    display_names = [
        f"{(e.get('personal_info') or {}).get('name')} ({e.get('employee_id')})" for e in employees
    ]
    id_lookup = dict(zip(display_names, [e.get("employee_id") for e in employees]))
    selected_name = st.selectbox("Select Employee", [""] + sorted(display_names))

    if selected_name:
        selected_id = id_lookup[selected_name]
        profile = emp_map[selected_id]

        # Header
        name = (profile.get("personal_info") or {}).get("name", "")
        role = (profile.get("employment_info") or {}).get("job_title", "")
        dept = (profile.get("employment_info") or {}).get("department", "")
        hire = (profile.get("employment_info") or {}).get("hire_date", "")
        st.subheader(f"{name} — {role}")
        st.markdown(f"**Department:** {dept} | **Hire Date:** {hire} | **ID:** {selected_id}")

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Overview & Leadership",
            "🚀 Career Pathway",
            "💬 Wellbeing Assistant",
            "🤝 Mentorship Hub",
            "🌟 Recognition"
        ])

        with tab1:
            st.markdown("### 🎯 Leadership Prediction")
            prob = all_predictions.get(selected_id, 0.0)
            st.metric("Predicted Leadership Potential", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))

            if prob > 0.6:
                st.success("High leadership potential — recommend strategic leadership training and mentorship.")
            elif prob > 0.3:
                st.info("Emerging leader — build via cross-functional projects and coaching.")
            else:
                st.warning("Focus on deepening technical expertise and broadening exposure.")

            st.markdown("### 🧩 Background & Skills")
            skills_list = [s.get("skill_name") for s in (profile.get("skills") or []) if s.get("skill_name")]
            st.write(f"**Skills ({len(skills_list)}):**", ", ".join(skills_list) if skills_list else "No skills listed.")
            comps_list = [c.get("name") for c in (profile.get("competencies") or []) if c.get("name")]
            st.write(f"**Competencies ({len(comps_list)}):**", ", ".join(comps_list) if comps_list else "No competencies listed.")

            with st.expander("View Raw Profile Data"):
                st.json(profile)

        with tab2:
            st.markdown("### 🚀 Personalised Career Pathway")
            st.write("Click to generate an AI-powered career plan (next roles, skill gaps, mobility options).")
            if st.button("Generate Career Pathway", key="pathway_btn"):
                with st.spinner("Generating AI pathway..."):
                    plan = generate_career_pathway(profile, prob, functions_df)
                st.markdown(plan["ai_reply"])

        with tab3:
            st.markdown("### 💬 Conversational Career Assistant")
            st.write("Ask about your career, skill development, or mental well-being.")

            if "messages" not in st.session_state:
                st.session_state.messages = {}
            if selected_id not in st.session_state.messages:
                st.session_state.messages[selected_id] = []

            # Show history
            for msg in st.session_state.messages[selected_id]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            user_msg = st.chat_input("Type your message...")
            if user_msg:
                st.session_state.messages[selected_id].append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.write(user_msg)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        res = handle_conversation(profile, user_msg)
                        reply = res["reply"]
                        st.write(reply)
                st.session_state.messages[selected_id].append({"role": "assistant", "content": reply})

        with tab4:
            st.markdown("### 🤝 Mentorship Hub")
            st.write("Find colleagues with high leadership potential to learn from.")
            if st.button("Find Potential Mentors", key="mentor_btn"):
                with st.spinner("Searching..."):
                    mentors = find_mentors(profile, employees, all_predictions)
                if not mentors:
                    st.info("No mentors found right now. Try adjusting criteria or check later.")
                else:
                    st.success(f"Found {len(mentors)} potential mentors!")
                    cols = st.columns(3)
                    for i, m in enumerate(mentors[:6]):
                        with cols[i % 3]:
                            with st.container(border=True):
                                st.subheader(m["name"])
                                st.write(f"**Role:** {m['role']}")
                                st.write(f"**Dept:** {m['department']}")
                                st.write(f"**Leadership Score:** {m['prob']:.1%}")
                                st.write(f"**Skill Overlap:** {m['skill_overlap']}")
                                st.button("Request Mentorship", key=f"req_{m['id']}")

        with tab5:
            st.markdown("### 🌟 Recognition & Feedback")
            st.write("Recognize a colleague for demonstrating PSA's values.")
            others = [n for n in sorted(display_names) if n != selected_name]
            recipient_name = st.selectbox("Who do you want to recognize?", others)
            values_selected = st.multiselect("Which PSA values did they demonstrate?", PSA_VALUES)
            feedback_msg = st.text_area("Your recognition message:", height=100)
            if st.button("Submit Recognition", key="recog_btn"):
                if recipient_name and values_selected and feedback_msg:
                    st.success(f"Thank you for recognizing {recipient_name}! Your feedback has been shared.")
                else:
                    st.warning("Please select a recipient, choose at least one value, and write a message.")

st.markdown("---")
st.caption("Built with ❤️ for PSA — empowering a future-ready workforce through AI and inclusivity.")
