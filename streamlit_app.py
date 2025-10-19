# psa_future_ready_app.py
# PSA Future-Ready Workforce — AI & ML Integrated Streamlit App (with Mentorship & Recognition)
# Run: streamlit run psa_future_ready_app.py
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
from typing import Tuple, Dict, List, Any
from openai import OpenAI
from pathlib import Path

# -------------------------
# Config & paths
# -------------------------
st.set_page_config(layout="wide", page_title="PSA Future-Ready Workforce — Full")
BASE_DIR = Path(__file__).resolve().parent
EMP_JSON = BASE_DIR / "Employee_Profiles.json"
FUNC_XLSX = BASE_DIR / "Functions & Skills.xlsx"
MODEL_PATH = BASE_DIR / "psa_model.joblib"
MENTOR_PATH = BASE_DIR / "mentorship_matches.json"
RECOG_PATH = BASE_DIR / "recognitions.json"

# -------------------------
# Sidebar: OpenAI key & accessibility toggles
# -------------------------
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("🔑 OpenAI API Key (optional)", type="password")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input
st.sidebar.markdown("**Accessibility**")
high_contrast = st.sidebar.checkbox("High contrast mode", value=False)
large_font = st.sidebar.checkbox("Large font (easier to read)", value=False)

# small CSS accessibility tweaks
css = ""
if high_contrast:
    css += """
    :root { color-scheme: dark; }
    .stApp { background-color: #000; color: #fff; }
    """
if large_font:
    css += """
    .stApp, .stApp * { font-size: 18px !important; }
    """
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# -------------------------
# OpenAI lazy client
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
# Basic file helpers (persist mentor & recognition data)
# -------------------------
def load_json_file(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def save_json_file(p: Path, data):
    try:
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False

# initialize storage files if missing
if not MENTOR_PATH.exists():
    save_json_file(MENTOR_PATH, {"matches": []})
if not RECOG_PATH.exists():
    save_json_file(RECOG_PATH, {"recognitions": []})

# -------------------------
# Data loading (profiles, taxonomy)
# -------------------------
@st.cache_data
def load_employee_json() -> List[dict]:
    if EMP_JSON.exists():
        try:
            return json.loads(EMP_JSON.read_text(encoding="utf-8"))
        except Exception as e:
            st.error(f"Could not parse {EMP_JSON}: {e}")
            return []
    else:
        st.error(f"Missing {EMP_JSON}. Place it in the app folder.")
        return []

@st.cache_data
def load_functions_skills() -> pd.DataFrame:
    if FUNC_XLSX.exists():
        try:
            xl = pd.read_excel(FUNC_XLSX, sheet_name=None, engine="openpyxl")
            frames = []
            for name, df in xl.items():
                df["__sheet__"] = name
                frames.append(df)
            return pd.concat(frames, ignore_index=True)
        except Exception as e:
            st.warning(f"Could not read {FUNC_XLSX}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# -------------------------
# Heuristic label creation + feature engineering
# includes reading optional performance_score and engagement_score if present
# -------------------------
def contains_lead_word(title: str) -> bool:
    if not title:
        return False
    t = title.lower()
    for kw in ["manager", "lead", "head", "director", "chief"]:
        if kw in t:
            return True
    return False

def derive_is_leader(profile: dict) -> int:
    # current/past roles
    for pos in profile.get("positions_history", []) or []:
        if contains_lead_word(pos.get("role_title", "")):
            return 1
    # leadership competency
    for c in profile.get("competencies", []) or []:
        if "leadership" in (c.get("name") or "").lower():
            if (c.get("level") or "").lower() in ("advanced", "expert"):
                return 1
    # project roles
    for p in profile.get("projects", []) or []:
        if contains_lead_word(p.get("role", "")):
            return 1
    return 0

def parse_date(s):
    if not s: return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

def profile_to_features(profile: dict) -> dict:
    ei = profile.get("employment_info", {}) or {}
    now = datetime.utcnow()
    hire = parse_date(ei.get("hire_date"))
    in_role = parse_date(ei.get("in_role_since"))
    years_total = (now - hire).days / 365.25 if hire else 0.0
    years_in_role = (now - in_role).days / 365.25 if in_role else 0.0
    skills = profile.get("skills", []) or []
    comps = profile.get("competencies", []) or []
    projs = profile.get("projects", []) or []
    positions = profile.get("positions_history", []) or []
    languages = profile.get("personal_info", {}).get("languages", []) or []
    # optional numeric fields if exist (behavioral/performance/engagement)
    perf = profile.get("performance_score") or profile.get("performance", None) or profile.get("employment_info", {}).get("performance_score", None)
    engage = profile.get("engagement_score") or profile.get("engagement", None) or profile.get("employment_info", {}).get("engagement_score", None)
    try:
        perf_val = float(perf) if perf is not None else 0.0
    except Exception:
        perf_val = 0.0
    try:
        engage_val = float(engage) if engage is not None else 0.0
    except Exception:
        engage_val = 0.0
    avg_proj_dur = 0.0
    durations = []
    for p in projs:
        s = parse_date((p.get("period") or {}).get("start"))
        e = parse_date((p.get("period") or {}).get("end")) or now
        if s and e:
            durations.append((e - s).days / 30.0)
    if durations:
        avg_proj_dur = float(np.mean(durations))
    return {
        "years_total": years_total,
        "years_in_role": years_in_role,
        "num_skills": len(skills),
        "num_competencies": len(comps),
        "num_projects": len(projs),
        "num_positions": len(positions),
        "num_languages": len(languages),
        "avg_proj_dur_months": avg_proj_dur,
        "has_lead_comp": int(any("lead" in (c.get("name","") or "").lower() for c in comps)),
        "num_trainings": len(profile.get("training_history", []) or []),
        "performance_score": perf_val,
        "engagement_score": engage_val
    }

# -------------------------
# Build dataset + train model (persist)
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
        feat["employee_id"] = p.get("employee_id") or p.get("id") or p.get("employeeId")
        rows.append(feat)
    df = pd.DataFrame(rows).set_index("employee_id")
    # handle empty df
    if df.shape[0] == 0:
        return None, {}
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].fillna(0).astype(float)
    y = df["label"].astype(int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000)
    acc = None; auc = None
    if len(df) >= 6 and len(y.unique()) > 1:
        try:
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42, stratify=y)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            try:
                auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
            except Exception:
                auc = None
        except Exception:
            clf.fit(Xs, y)
    else:
        clf.fit(Xs, y)
    model_pack = {"model": clf, "scaler": scaler, "feature_cols": feature_cols}
    try:
        joblib.dump(model_pack, MODEL_PATH)
    except Exception:
        pass
    return model_pack, {"df": df, "accuracy": acc, "auc": auc, "n": len(df)}

# -------------------------
# Prediction & explainability
# -------------------------
def predict_for_profile(model_pack: dict, profile: dict) -> dict:
    feat = profile_to_features(profile)
    cols = model_pack["feature_cols"]
    X_row = np.array([feat.get(c, 0.0) for c in cols], dtype=float).reshape(1, -1)
    Xs = model_pack["scaler"].transform(X_row)
    prob = float(model_pack["model"].predict_proba(Xs)[0,1])
    pred = int(model_pack["model"].predict(Xs)[0])
    try:
        coefs = model_pack["model"].coef_[0]
        contrib = {cols[i]: float(coefs[i] * Xs[0,i]) for i in range(len(cols))}
        contrib_sorted = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)
    except Exception:
        contrib_sorted = []
    return {"probability": prob, "prediction": pred, "features": feat, "contributions": contrib_sorted}

# -------------------------
# Recommender (skill->training) - unchanged
# -------------------------
DEFAULT_TRAINING_DB = {
    "cloud": ["Cloud Architecture Masterclass", "IaC (Terraform)"],
    "leadership": ["Leading High Performance Teams", "Coaching Skills for Managers"],
    "analytics": ["Data Analytics Foundations", "SQL for Analysts"],
    "finance": ["Financial Modeling Bootcamp", "Advanced FP&A"]
}

def recommend_trainings_from_skills(profile: dict) -> Tuple[List[str], List[str]]:
    skills = [(s.get("skill_name") or "").lower() for s in profile.get("skills", [])]
    trainings = []; matched = []
    for sk in skills:
        for key, courses in DEFAULT_TRAINING_DB.items():
            if key in sk or sk in key:
                trainings.extend(courses)
                matched.append(sk)
    return list(dict.fromkeys(trainings)), matched

# -------------------------
# Conversational assistant (OpenAI + fallback)
# -------------------------
def handle_conversation(profile: dict, message: str, leadership_prob: float = 0.0) -> dict:
    client = get_openai_client()
    if not client:
        # fallback
        m = (message or "").lower()
        if any(w in m for w in ["stress","burnout","anxious","anxiety","overwhelm"]):
            return {"reply": "I'm sorry you're feeling this way. Consider contacting EAP or your manager; take short breaks, and try breathing exercises."}
        if any(w in m for w in ["career","promotion","role","move"]):
            trainings, matched = recommend_trainings_from_skills(profile)
            return {"reply": f"Consider internal mobility and targeted upskilling. Suggested trainings: {', '.join(trainings[:5]) if trainings else 'Leadership basics.'}"}
        return {"reply": "I can help with career, training, or wellbeing. Try asking about 'career path' or 'training'."}
    # AI-driven
    name = profile.get("personal_info", {}).get("name", "employee")
    role = profile.get("employment_info", {}).get("job_title", "employee")
    skills = [s.get("skill_name") for s in profile.get("skills", []) if s.get("skill_name")]
    system_prompt = f"""
    You are PSA's AI Career & Wellbeing Assistant. PSA's priorities: inclusive growth, wellbeing, digital upskilling, and internal mobility.
    Employee context:
    - Name: {name}
    - Role: {role}
    - Leadership potential: {leadership_prob:.2%}
    - Key skills: {', '.join(skills[:10])}
    Provide a concise, empathetic, and actionable reply (max 6 sentences). If asked about training or career, list 2-3 concrete suggestions with short rationale.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            max_tokens=300,
            temperature=0.7
        )
        return {"reply": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"reply": f"(AI assistant error: {e})"}

# -------------------------
# Mentorship matching engine
# -------------------------
def suggest_mentors(profile: dict, employees: List[dict], model_pack: dict, top_k: int = 5) -> List[dict]:
    """
    Suggest internal mentors:
    - Candidate mentors: employees with higher leadership probability OR derived leader label
    - Score = skill_overlap_weight * overlap_count + leadership_bonus*mentor_prob + experience_bonus
    """
    # compute mentor probabilities or labels
    mentor_list = []
    for cand in employees:
        if cand.get("employee_id") == profile.get("employee_id"):
            continue
        cand_features = profile_to_features(cand)
        # predict if model exists
        try:
            pred = predict_for_profile(model_pack, cand) if model_pack else {"probability": derive_is_leader(cand)}
            mentor_prob = pred["probability"]
        except Exception:
            mentor_prob = 1.0 if derive_is_leader(cand) else 0.0
        # compute skill overlap: mentee needs skills they have; mentor should have them plus complementary
        mentee_skills = set((s.get("skill_name") or "").lower() for s in profile.get("skills", []) if s.get("skill_name"))
        cand_skills = set((s.get("skill_name") or "").lower() for s in cand.get("skills", []) if s.get("skill_name"))
        overlap = len(mentee_skills & cand_skills)
        # diversity factor: prefer mentors from different department to broaden exposure
        dept_same = (profile.get("employment_info", {}).get("department") == cand.get("employment_info", {}).get("department"))
        # experience difference (years_total)
        exp_diff = cand_features.get("years_total", 0.0) - profile_to_features(profile).get("years_total", 0.0)
        exp_diff = max(0.0, exp_diff)
        # score
        score = overlap * 2.0 + mentor_prob * 3.0 + (exp_diff * 0.2) + (0 if dept_same else 0.5)
        mentor_list.append({
            "employee_id": cand.get("employee_id"),
            "name": cand.get("personal_info", {}).get("name"),
            "role": cand.get("employment_info", {}).get("job_title"),
            "overlap": overlap,
            "mentor_prob": mentor_prob,
            "experience_years": cand_features.get("years_total", 0.0),
            "score": score
        })
    # sort and return top_k
    mentor_list_sorted = sorted(mentor_list, key=lambda x: x["score"], reverse=True)
    return mentor_list_sorted[:top_k]

def record_mentorship_match(mentee_id: str, mentor_id: str):
    data = load_json_file(MENTOR_PATH, {"matches": []})
    match = {
        "mentee_id": mentee_id,
        "mentor_id": mentor_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    data["matches"].append(match)
    save_json_file(MENTOR_PATH, data)
    return match

# -------------------------
# Feedback & recognition (simple)
# -------------------------
def record_recognition(target_emp_id: str, from_name: str, message: str, tags: List[str]):
    data = load_json_file(RECOG_PATH, {"recognitions": []})
    rec = {
        "target": target_emp_id,
        "from": from_name,
        "message": message,
        "tags": tags,
        "timestamp": datetime.utcnow().isoformat()
    }
    data["recognitions"].append(rec)
    save_json_file(RECOG_PATH, data)
    return rec

def get_recognitions_for(emp_id: str) -> List[dict]:
    data = load_json_file(RECOG_PATH, {"recognitions": []})
    return [r for r in data["recognitions"] if r["target"] == emp_id]

def recognition_leaderboard(top_n: int = 5):
    data = load_json_file(RECOG_PATH, {"recognitions": []})
    df = pd.DataFrame(data["recognitions"])
    if df.empty:
        return []
    counts = df.groupby("target").size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False).head(top_n)
    return counts.to_dict(orient="records")

# -------------------------
# UI - Main layout
# -------------------------
st.title("🚢 PSA Future-Ready Workforce — Full Solution")
st.markdown("AI-driven career pathways, mentoring, wellbeing support, leadership prediction, and recognition.")

# Load data & model
employees = load_employee_json()
functions_df = load_functions_skills()
model_pack, info = build_dataset_and_train()

if not employees:
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.header("Data & Model")
    st.write(f"Profiles loaded: **{len(employees)}**")
    if info.get("accuracy") is not None:
        st.metric("Test accuracy", f"{info['accuracy']:.3f}")
    st.button("Retrain Model", on_click=lambda: build_dataset_and_train())

    st.markdown("**Recognition Leaderboard**")
    lb = recognition_leaderboard(5)
    if lb:
        for row in lb:
            emp = row["target"]
            name = next((e.get("personal_info", {}).get("name") for e in employees if e.get("employee_id")==emp), emp)
            st.write(f"- {name}: {row['count']} recognitions")
    else:
        st.write("No recognitions yet.")

with right:
    st.header("Employee insights & actions")
    emp_map = {e.get("employee_id"): e for e in employees}
    emp_ids = list(emp_map.keys())
    selected = st.selectbox("Select employee", [""] + emp_ids)
    if selected:
        profile = emp_map[selected]
        name = profile.get("personal_info", {}).get("name", "")
        title = profile.get("employment_info", {}).get("job_title", "")
        dept = profile.get("employment_info", {}).get("department", "")
        st.subheader(f"{name} — {title}")
        st.caption(f"Department: {dept}")

        # Show skills, competencies
        st.markdown("**Skills**: " + ", ".join([s.get("skill_name") for s in profile.get("skills", []) or []]))
        st.markdown("**Competencies**: " + ", ".join([c.get("name") for c in profile.get("competencies", []) or []]))

        # Leadership prediction
        if model_pack:
            pred = predict_for_profile(model_pack, profile)
            prob = pred["probability"]
            st.metric("Leadership potential", f"{prob:.1%}")
            if prob > 0.6:
                st.success("High leadership potential — consider mentorship & stretch assignments.")
            elif prob > 0.3:
                st.info("Emerging leader — suggest leadership development.")
            else:
                st.write("Focus on upskilling and exposure to cross-functional projects.")
            st.markdown("Top feature contributions:")
            for f, v in pred["contributions"][:6]:
                st.write(f"- {f}: {v:.3f}")
        else:
            st.warning("Model not available.")

        # Training recommendations
        st.markdown("### Recommended trainings")
        trainings, matched = recommend_trainings_from_skills(profile)
        if trainings:
            for t in trainings:
                st.write(f"- {t}")
        else:
            st.write("No direct matches — consider leadership or cross-functional training.")

        # Career pathway generation (AI-powered)
        st.markdown("### Personalised Career Pathway & Upskilling Plan")
        if st.button("Generate Career Pathway & Plan"):
            prob_val = pred["probability"] if model_pack else derive_is_leader(profile)
            plan = generate_career_pathway(profile, prob_val)
            st.write(plan.get("ai_reply", plan))

        # Mentorship suggestions
        st.markdown("### Mentorship suggestions")
        mentors = suggest_mentors(profile, employees, model_pack, top_k=5)
        if mentors:
            for m in mentors:
                st.write(f"- {m['name']} ({m['role']}) — score {m['score']:.2f} — overlap {m['overlap']} skills — leadership_prob {m['mentor_prob']:.2f}")
            # choose first as an action
            chosen = st.selectbox("Pick mentor to match", [""] + [m["employee_id"] for m in mentors])
            mentor_names = {m["employee_id"]: m["name"] for m in mentors}
            if st.button("Record mentorship match"):
                if chosen:
                    match = record_mentorship_match(profile.get("employee_id"), chosen)
                    st.success(f"Recorded match with {mentor_names.get(chosen, chosen)}")
                else:
                    st.warning("Select a mentor first.")
        else:
            st.write("No mentor suggestions available.")

        # Recognition & feedback
        st.markdown("### Recognitions & Feedback")
        st.write("Send recognition to this employee (peer/manager).")
        from_name = st.text_input("Your name", key=f"from_{selected}")
        rec_msg = st.text_area("Recognition message", key=f"msg_{selected}")
        tags_txt = st.text_input("Tags (comma separated)", key=f"tags_{selected}")
        if st.button("Send recognition"):
            if not from_name or not rec_msg.strip():
                st.warning("Please provide your name and a message.")
            else:
                tags = [t.strip() for t in tags_txt.split(",") if t.strip()]
                rec = record_recognition(selected, from_name, rec_msg.strip(), tags)
                st.success("Recognition recorded. Thank you!")
        st.markdown("**Recognition history**")
        recs = get_recognitions_for(selected)
        if recs:
            for r in sorted(recs, key=lambda x: x["timestamp"], reverse=True):
                ts = r.get("timestamp")
                frm = r.get("from")
                msg = r.get("message")
                tags_display = ", ".join(r.get("tags", []))
                st.write(f"- {ts} — **{frm}**: {msg}  {'(' + tags_display + ')' if tags_display else ''}")
        else:
            st.write("No recognitions yet.")

# Bottom: dataset export & notes
st.markdown("---")
st.header("Export & Notes")
if info.get("df") is not None:
    df_out = info["df"].reset_index()
    st.download_button("Download derived dataset (CSV)", df_out.to_csv(index=False).encode("utf-8"), "psa_derived_dataset.csv", "text/csv")
st.markdown("""
### What this app delivers
- Predicts leadership potential (using role history, competencies, performance/engagement fields if present).
- Recommends trainings and uses AI to generate personalised career pathways & upskilling plans.
- Provides an AI conversational assistant (OpenAI) with a rule-based fallback.
- Suggests targeted mentors and records mentorship matches for program tracking.
- Tracks recognition & feedback, with a lightweight leaderboard.

### Next steps for production
- Replace heuristic labels with HR-provided leadership outcomes; add privacy, access controls, and audit logs.
- Expand training catalogue and integrate LMS / internal role catalog for precise internal mobility options.
- Add SHAP or similar explainability for model transparency.
- Integrate scheduled reviews / mentor workflows, and a feedback loop from outcomes into model training.
""")
