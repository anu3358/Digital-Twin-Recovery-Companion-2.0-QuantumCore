# app.py - Digital-Twin Recovery Companion - PATCHED & ENHANCED (FINAL) - SAFE SESSION HANDLING
import os
import io
import time
import math
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------
# Local project imports - rely on your files
# -----------------------------------------
from database import SessionLocal, engine, Base
# models must expose: User, PatientProfile, SensorStream, Prediction, TwinModel
from models import User, PatientProfile, SensorStream, Prediction, TwinModel
from audit import log_action
from report import generate_report

# util.auth may not exist or may differ across versions - safe fallback
try:
    from util.auth import authenticate, verify_password, hash_password, pwd_context
except Exception:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

    def verify_password(plain: str, hashed: str) -> bool:
        try:
            return pwd_context.verify(plain, hashed)
        except Exception:
            return False

    def authenticate(db, email: str, password: str):
        user = db.query(User).filter(User.email == email).first()
        if user and verify_password(password, user.hashed_password):
            return user
        return None

    def hash_password(pw: str) -> str:
        return pwd_context.hash(pw)

# Ensure DB tables exist (imports models to register mappings)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    # swallow for deploy environments where DB isn't ready; logs will show details
    print("Warning: Base.metadata.create_all failed:", e)

# -----------------------
# Streamlit page settings
# -----------------------
st.set_page_config(page_title="Digital-Twin Recovery Companion", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Inject dark theme CSS
# -----------------------
_dark_css = r"""
<style>
/* Page background gradient */
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #071022 0%, #04121b 40%, #021018 100%);
  color: #e6eef8;
  min-height: 100vh;
}
/* Sidebar background */
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#071421,#041022) !important; color: #d7eefb; }
/* Card backgrounds and inputs */
.css-1v3fvcr, .css-1d391kg, .stApp, .stMarkdown { background: rgba(10,16,24,0.45) !important; border-radius: 8px; }
.stButton>button, .stDownloadButton>button { border-radius: 10px; }
.stTextInput>div>div>input, textarea { background: rgba(255,255,255,0.02) !important; color: #e6eef8 !important; border-radius: 6px; }
/* Headings */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #cbe8ff !important; }
.streamlit-expanderHeader { color: #c8e7ff !important; }
/* Make small plot icons visible on dark bg */
.plotly-graph-div .modebar { background: rgba(0,0,0,0.35) !important; }
</style>
"""
st.markdown(_dark_css, unsafe_allow_html=True)

# -----------------------
# Safe rerun helper
# -----------------------
def safe_rerun():
    """
    Try to call Streamlit's experimental rerun. If unavailable or it errors,
    set a small session-state flip and stop the current run so the page reloads
    naturally on the next request.
    """
    try:
        st.experimental_rerun()
    except Exception:
        st.session_state["_safe_rerun_ts"] = time.time()
        try:
            st.stop()
        except Exception:
            return

# ---------- SAFE USER HELPERS ----------
def login_store_primitives(user):
    """Store only primitives in session state (never a SQLAlchemy ORM object)."""
    st.session_state["user_id"] = int(user.id)
    st.session_state["user_email"] = str(user.email)
    st.session_state["role"] = str(user.role)
    st.session_state["user_name"] = getattr(user, "full_name", None)

def logout_clear():
    """Clear login-related session keys."""
    for k in ["user_id", "user_email", "role", "user_name"]:
        if k in st.session_state:
            del st.session_state[k]

def get_current_user():
    """Return a fresh SQLAlchemy User object for current session user_id (or None)."""
    uid = st.session_state.get("user_id")
    if not uid:
        return None
    db = SessionLocal()
    try:
        # use session.get if available (SQLAlchemy 1.4+)
        try:
            u = db.get(User, uid)
        except Exception:
            u = db.query(User).filter(User.id == uid).first()
        return u
    finally:
        db.close()

# -----------------------
# Helper utilities
# -----------------------
def infer_sampling_hz(df: Optional[pd.DataFrame]) -> int:
    if df is None or "timestamp" not in df.columns:
        return 1
    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        if len(ts) < 2:
            return 1
        diffs = ts.sort_values().diff().dt.total_seconds().dropna()
        med = diffs.median() if len(diffs) else 1.0
        if med <= 0:
            return 1
        hz = int(round(1.0 / med))
        return max(1, hz)
    except Exception:
        return 1

def extract_features(df: pd.DataFrame) -> dict:
    feats = {}
    df2 = df.copy()
    # ensure numeric for known columns
    for c in ["accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0)
    if {"accel_x","accel_y","accel_z"}.issubset(df2.columns):
        df2["accel_mag"] = np.sqrt(df2["accel_x"]**2 + df2["accel_y"]**2 + df2["accel_z"]**2)
        feats["accel_mag_mean"] = float(df2["accel_mag"].mean())
        feats["accel_mag_std"] = float(df2["accel_mag"].std())
        feats["accel_mag_rms"] = float(np.sqrt((df2["accel_mag"]**2).mean()))
    if "emg" in df2.columns:
        feats["emg_mean"] = float(df2["emg"].mean())
        feats["emg_rms"] = float(np.sqrt((df2["emg"]**2).mean()))
    if "hr" in df2.columns:
        feats["hr_mean"] = float(df2["hr"].mean())
        feats["hr_std"] = float(df2["hr"].std())
    if "spo2" in df2.columns:
        feats["spo2_mean"] = float(df2["spo2"].mean())
    if "step_count" in df2.columns:
        steps_diff = df2["step_count"].diff().clip(lower=0).fillna(0)
        hz = infer_sampling_hz(df2)
        cadence = steps_diff.mean() * 60 * hz
        feats["cadence_est"] = float(cadence)
        feats["total_steps"] = int(df2["step_count"].max()) if len(df2) else 0
    return feats

def load_csv_from_path(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

# Path for auto-generated dataset (if present)
AUTOGEN_PATH = Path("data/generated_sample.csv")
if AUTOGEN_PATH.exists():
    st.sidebar.success("Found generated dataset (auto-load available)")

# -----------------------
# Seeding demo users (safe)
# -----------------------
def seed_demo_if_missing():
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == "patient@example.com").first():
            pwd = pwd_context
            for email, role, name in [
                ("admin@example.com", "admin", "Admin User"),
                ("clinician@example.com", "clinician", "Clinician One"),
                ("patient@example.com", "patient", "Patient One"),
            ]:
                if db.query(User).filter(User.email == email).first():
                    continue
                u = User(email=email, hashed_password=pwd.hash("changeme"), role=role, full_name=name)
                db.add(u); db.commit(); db.refresh(u)
                if role == "patient":
                    p = PatientProfile(user_id=u.id, demographics={"age":45}, medical_history="Demo")
                    db.add(p); db.commit()
            print("Seeded demo users.")
    except Exception as e:
        print("Seed error:", e)
    finally:
        db.close()

# Auto-seed on startup if env var set OR if Judge Mode requires it later
if os.getenv("SEED_ON_STARTUP", "0") == "1":
    seed_demo_if_missing()

# -----------------------
# Session state defaults (use primitives, not ORM)
# -----------------------
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "last_uploaded_df" not in st.session_state:
    st.session_state["last_uploaded_df"] = None
if "latest_feats" not in st.session_state:
    st.session_state["latest_feats"] = None
if "_judge_enabled" not in st.session_state:
    st.session_state["_judge_enabled"] = False

# -----------------------
# Sidebar: login & judge mode (with explicit enable)
# -----------------------
with st.sidebar:
    st.title("Digital-Twin")
    judge_opt = st.checkbox("🎯 Judge Mode (auto-demo)", value=True, help="Auto-login and preload demo/sample data for instant walkthrough.")
    st.caption("Auto-login and preload demo/generated data for instant walkthrough.")

    # require explicit permission button to enable judge mode (safer for demos)
    if judge_opt and not st.session_state["_judge_enabled"]:
        st.warning("Judge Mode will auto-login a demo user and auto-load sample data. Click to enable.")
        if st.button("Enable Judge Mode (confirm)"):
            st.session_state["_judge_enabled"] = True
            seed_demo_if_missing()
            safe_rerun()

    if st.session_state["_judge_enabled"] and not st.session_state.get("user_id"):
        # auto-login demo patient (store primitives, not ORM)
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == "patient@example.com").first()
            if user:
                login_store_primitives(user)
                st.success(f"Auto-logged in as {user.email}")
        finally:
            db.close()

    if st.session_state.get("user_id"):
        ue = st.session_state.get("user_email", "unknown")
        ur = st.session_state.get("role", "unknown")
        st.markdown(f"**Logged in:** {ue}  \n`({ur})`")
        if st.button("Logout", use_container_width=True):
            logout_clear()
            safe_rerun()
    else:
        st.subheader("Login")
        email = st.text_input("Email", value="patient@example.com")
        password = st.text_input("Password", value="changeme", type="password")
        role_pick = st.selectbox("Role", options=["patient","clinician","admin"])
        if st.button("Sign in", use_container_width=True):
            db = SessionLocal()
            try:
                user = authenticate(db, email, password)
                if user and (role_pick == user.role or role_pick == "admin"):
                    # store primitives
                    login_store_primitives(user)
                    try:
                        log_action(db, user.id, "login", {"role": user.role})
                    except Exception:
                        pass
                    safe_rerun()
                else:
                    st.error("Invalid credentials or role mismatch")
            finally:
                db.close()

# Main title
st.title("Digital-Twin Recovery Companion")

# If not logged in, stop (sidebar prompts)
if not st.session_state.get("user_id"):
    st.info("Please log in from the sidebar to continue.")
    st.stop()

# use primitive role
role = st.session_state.get("role")

# -----------------------
# Tabs
# -----------------------
tabs = ["🏠 Overview", "📥 Data Ingestion"]
if role in ["clinician","admin"]:
    tabs.append("👩‍⚕️ Clinician")
if role == "admin":
    tabs.append("🛠️ Admin")
tabs.append("🧬 Data Generator")
active = st.tabs(tabs)

# -----------------------
# Overview tab
# -----------------------
with active[0]:
    col_main, col_side = st.columns([2, 1], gap="large")
    with col_main:
        st.header("Recovery Progress")
        days = list(range(30))
        base_values = [0.35 + i*0.02 + math.sin(i/3)*0.01 for i in days]
        # if a previous training simulation improved accuracy, slightly boost curve
        boost_factor = st.session_state.get("training_boost", 0.0)
        values = [v * (1.0 + boost_factor) for v in base_values]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=values, mode="lines+markers", name="Recovery Index"))
        fig.update_layout(title="30-Day Recovery Trajectory", height=340, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Mood & Pain Heatmap")
        timeslots = ["08:00","12:00","16:00","20:00"]
        days_labels = [(datetime.today() - timedelta(days=i)).strftime("%a %d") for i in range(6, -1, -1)]
        pain = np.clip(np.random.normal(4, 1.5, size=(7,4)), 0, 10)
        heat = go.Figure(data=go.Heatmap(z=pain, x=timeslots, y=days_labels, zmin=0, zmax=10, colorbar=dict(title="Pain")))
        heat.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(heat, use_container_width=True)

        st.markdown("#### Digital Twin (Animated)")
        # Gait stick figure + 3D brain in two columns
        gait_col, brain_col = st.columns([2,1])
        # build interactive controls retained from previous version
        with st.expander("Digital Twin Controls", expanded=False):
            dt_mode = st.selectbox("Mode", ["Walk","Balance","Step-Up"], key="dt_mode")
            dt_speed = st.selectbox("Speed", ["Slow","Normal","Fast"], key="dt_speed")
            dt_muscle = st.selectbox("Muscle Highlight", ["None","EMG","Fatigue","Pain"], key="dt_muscle")
            st.markdown("Use Play to animate gait; use Simulate Training to smooth motion (demo).")

        # Build gait frames (same as before but speed-scaled)
        t = np.linspace(0, 2*np.pi, 30)
        base = np.array([[0,1.8,0],[0,1.4,0],[-0.3,1.1,0],[0,1.4,0],[0.3,1.1,0],
                         [0,1.4,0],[0,0.8,0],[-0.2,0.2,0],[0,0.8,0],[0.2,0.2,0]])
        frames = []
        speed_factor = 1.0 if dt_speed=="Normal" else (0.6 if dt_speed=="Slow" else 1.6)
        for i in range(len(t)):
            phase = 0.12 * np.sin(t[i] * speed_factor)
            xs = base[:,0] + np.array([0,0,phase,0,-phase,0,0,phase,0,-phase])
            ys = base[:,1]
            frames.append(go.Frame(data=[go.Scatter3d(x=xs, y=ys, z=[0]*len(xs), mode='lines+markers')]))

        gait_fig = go.Figure(data=[go.Scatter3d(x=base[:,0], y=base[:,1], z=[0]*10, mode='lines+markers')], frames=frames)
        gait_fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                               height=360, margin=dict(l=10,r=10,t=10,b=10))
        gait_fig.update_layout(updatemenus=[{"type":"buttons", "buttons":[
            {"label":"Play gait", "method":"animate", "args":[None, {"frame": {"duration":60, "redraw": True}, "fromcurrent": True}]},
            {"label":"Pause", "method":"animate", "args":[ [None], {"mode":"immediate", "frame": {"duration": 0}} ]}
        ]}])

        # Brain base (neurons)
        n_neurons = 80
        np.random.seed(42)
        bx = np.random.uniform(-1,1,n_neurons); by = np.random.uniform(-1,1,n_neurons); bz = np.random.uniform(-1,1,n_neurons)
        brain_fig = go.Figure(data=[go.Scatter3d(x=bx, y=by, z=bz, mode='markers',
                                                 marker=dict(size=5, color=[0.2]*n_neurons, colorscale='Viridis'))])
        brain_fig.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10),
                                scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))

        with gait_col:
            st.plotly_chart(gait_fig, use_container_width=True)
            st.caption("Digital Twin Gait (stick-figure) — Play/Pause controls available.")
        with brain_col:
            brain_plotter = st.empty()
            brain_plotter.plotly_chart(brain_fig, use_container_width=True)
            st.caption("Digital Twin Neural Activity (demo) — Simulate Training to animate")

    with col_side:
        st.subheader("What-if Simulation")
        extra = st.slider("Extra balance training (min/day)", 0, 60, 10)
        conf = st.select_slider("Confidence", options=["Low","Medium","High"], value="Medium")
        if st.button("Run Simulation"):
            model = TwinModel()
            user_id = st.session_state.get("user_id")
            pred = model.predict(patient_id=user_id, scenario={"extra_minutes_balance": extra})
            db = SessionLocal()
            try:
                try:
                    log_action(db, user_id, "prediction", {"extra_minutes": extra, "conf": conf})
                except Exception:
                    pass
            finally:
                db.close()
            st.metric("Predicted gait speed Δ", f"{pred.get('gait_speed_change_pct',0)} %")
            st.metric("Adherence score", f"{pred.get('adherence_score',0)}")

        st.markdown("---")
        st.markdown("#### Generate PDF Report")
        patient_name = st.text_input("Patient Name", value="Patient One")
        if st.button("Download Report"):
            metrics = {"Gait Speed Change %": 12.5, "Adherence Score": 85, "Next Step": "Add 5 min balance training"}
            pdf_bytes = generate_report(patient_name or "Unknown", metrics)
            st.download_button("Download PDF", data=pdf_bytes, file_name="recovery_report.pdf", mime="application/pdf")

# -----------------------
# Data Ingestion tab
# -----------------------
with active[1]:
    st.header("📥 Ingest Wearable CSV")
    st.write("Upload CSV with columns: `timestamp, patient_id (optional), accel_x, accel_y, accel_z, emg, spo2, hr, step_count`")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"], accept_multiple_files=False)
    use_autogen = False
    if AUTOGEN_PATH.exists():
        use_autogen = st.checkbox(f"Load generated dataset: {AUTOGEN_PATH.name}", value=False)

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("CSV uploaded")
        except Exception as e:
            st.error("Failed to read uploaded CSV: " + str(e))
            df = None
    elif use_autogen:
        df = load_csv_from_path(str(AUTOGEN_PATH))
        if df is None:
            st.error("Failed to load generated dataset.")

    if df is not None:
        df.columns = [c.strip() for c in df.columns]
        st.session_state["last_uploaded_df"] = df
        st.subheader("Preview")
        st.dataframe(df.head(200), use_container_width=True)

        patients = []
        if "patient_id" in df.columns:
            patients = sorted(pd.unique(df["patient_id"]).tolist())
            st.markdown(f"Detected patient IDs: {patients}")
            pid_sel = st.selectbox("Select patient (per-patient plots)", options=["All"] + [str(x) for x in patients], index=0)
        else:
            pid_sel = "All"

        plot_df = df if pid_sel == "All" else df[df["patient_id"].astype(str) == str(pid_sel)]
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(n=2000, random_state=42).sort_index()

        # Plot accelerometer
        if {"accel_x","accel_y","accel_z"}.issubset(df.columns):
            st.markdown("**Accelerometer**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["accel_x"], name="accel_x"))
            fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["accel_y"], name="accel_y"))
            fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["accel_z"], name="accel_z"))
            fig.update_layout(height=240, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # EMG
        if "emg" in df.columns:
            st.markdown("**EMG**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["emg"], name="emg"))
            fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # HR
        if "hr" in df.columns:
            st.markdown("**Heart Rate**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["hr"], name="hr"))
            fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        # features & session state
        feats = extract_features(plot_df)
        st.markdown("#### Extracted features (for model)")
        st.json(feats)
        st.session_state["latest_feats"] = feats

        # store small sample to DB
        if st.button("Store small sample to DB"):
            db = SessionLocal()
            try:
                user_id = st.session_state.get("user_id")
                patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == user_id).first()
                if not patient_profile:
                    patient_profile = PatientProfile(user_id=user_id, demographics={}, medical_history="")
                    db.add(patient_profile); db.commit(); db.refresh(patient_profile)
                sample = df.head(200)
                for _, row in sample.iterrows():
                    payload = {}
                    for col in ["accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]:
                        if col in row.index:
                            try:
                                payload[col] = float(row[col]) if not pd.isna(row[col]) else 0.0
                            except Exception:
                                payload[col] = 0.0
                    s = SensorStream(patient_id=patient_profile.id, sensor_type="wearable_csv", payload=payload)
                    db.add(s)
                db.commit()
                try:
                    log_action(db, user_id, "csv_upload", {"rows": len(df)})
                except Exception:
                    pass
                st.success("Stored sample rows to DB")
            except Exception as e:
                st.error("DB store failed: " + str(e))
            finally:
                db.close()

        # Predict from features
        st.markdown("---")
        st.markdown("#### Predict from extracted features")
        extra_minutes = st.slider("Extra balance training (min/day)", 0, 60, 15)
        if st.button("Run Prediction from Features"):
            feats = st.session_state.get("latest_feats")
            if not feats:
                st.warning("Please upload/select data first.")
            else:
                model = TwinModel()
                user_id = st.session_state.get("user_id")
                res = model.predict(patient_id=user_id, scenario={"extra_minutes_balance": extra_minutes}, feats=feats)
                db = SessionLocal()
                try:
                    try:
                        log_action(db, user_id, "prediction", {"extra_minutes": extra_minutes})
                    except Exception:
                        pass
                finally:
                    db.close()
                st.metric("Predicted gait speed Δ", f"{res.get('gait_speed_change_pct',0)} %")
                st.metric("Adherence score", f"{res.get('adherence_score',0)}")

# -----------------------
# Clinician tab
# -----------------------
if role in ["clinician","admin"]:
    if "👩‍⚕️ Clinician" in tabs:
        idx = tabs.index("👩‍⚕️ Clinician")
        with active[idx]:
            st.header("Clinician Dashboard")
            db = SessionLocal()
            try:
                patients = db.query(PatientProfile).all()
                rows = []
                for p in patients:
                    u = db.query(User).filter(User.id == p.user_id).first()
                    rows.append({"id": p.id, "name": (u.full_name or u.email) if u else "Unknown"})
                st.table(rows)
            finally:
                db.close()

# -----------------------
# Admin tab
# -----------------------
if role == "admin":
    if "🛠️ Admin" in tabs:
        idx_admin = tabs.index("🛠️ Admin")
        with active[idx_admin]:
            st.header("Admin Tools")
            db = SessionLocal()
            try:
                full_name = st.text_input("Full name")
                email_new = st.text_input("Email")
                pw_new = st.text_input("Password", type="password")
                role_new = st.selectbox("Role", ["patient","clinician"])
                if st.button("Create User"):
                    if not email_new or not pw_new:
                        st.error("Email and password required")
                    else:
                        if db.query(User).filter(User.email == email_new).first():
                            st.error("Email exists")
                        else:
                            u = User(email=email_new, hashed_password=pwd_context.hash(pw_new), role=role_new, full_name=full_name)
                            db.add(u); db.commit(); db.refresh(u)
                            if role_new == "patient":
                                db.add(PatientProfile(user_id=u.id, demographics={}, medical_history=""))
                                db.commit()
                            st.success("User created")
                st.markdown("---")
                st.code(f"DB = {os.getenv('DATABASE_URL','sqlite:///./data/app.db')}")
            finally:
                db.close()

# -----------------------
# Data Generator & Training Simulation
# -----------------------
if tabs[-1] == "🧬 Data Generator":
    with active[-1]:
        st.header("🧬 Synthetic Dataset Generator (based on sample)")

        # ===== SAFE sample_df retrieval (DO NOT use `or` with DataFrames) =====
        _last = st.session_state.get("last_uploaded_df")
        if _last is not None:
            sample_df = _last
        elif AUTOGEN_PATH.exists():
            sample_df = load_csv_from_path(str(AUTOGEN_PATH))
        else:
            sample_df = None
        # =====================================================================

        default_hz = infer_sampling_hz(sample_df)

        n_pat = st.slider("Patients", 1, 20, 5)
        hours = st.slider("Hours per patient", 1, 48, 2)
        hz = st.slider("Sampling freq (Hz)", 1, 10, default_hz if default_hz else 1)
        mode = st.selectbox("Activity mode", ["mixed", "low", "medium", "high"])

        if st.button("⚙️ Generate Dataset (based on sample)"):
            with st.spinner("Generating dataset..."):
                parts = []
                for pid in range(1, n_pat+1):
                    lvl = np.random.choice(["low","medium","high"]) if mode == "mixed" else mode
                    total = max(1, hours * 3600 * hz)
                    # limit to keep memory small on cloud; warn if huge
                    if total * n_pat > 5_000_000:
                        st.warning("Large generation requested — streamlit might run out of memory. Reduce hours or patients.")
                    ts = [datetime.now() - timedelta(seconds=j / hz) for j in range(total)]
                    ts.reverse()
                    stats = {}
                    if sample_df is not None and isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
                        for c in ["accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]:
                            if c in sample_df.columns:
                                stats[c] = {"mean": float(sample_df[c].mean()), "std": float(sample_df[c].std())}
                    tvals = np.linspace(0, 10*np.pi, total)
                    accel_x = (stats.get("accel_x",{}).get("mean",0) + (stats.get("accel_x",{}).get("std",0) * np.sin(tvals + pid))
                               + np.random.normal(0, max(0.15, stats.get("accel_x",{}).get("std",0.5)), total))
                    accel_y = (stats.get("accel_y",{}).get("mean",0) + (stats.get("accel_y",{}).get("std",0) * np.cos(tvals + pid))
                               + np.random.normal(0, max(0.15, stats.get("accel_y",{}).get("std",0.5)), total))
                    accel_z = (stats.get("accel_z",{}).get("mean",1) + np.random.normal(0, max(0.05, stats.get("accel_z",{}).get("std",0.05)), total))
                    emg = np.abs(np.random.normal(stats.get("emg",{}).get("mean",0.6), max(0.1, stats.get("emg",{}).get("std",0.3)), total))
                    spo2 = np.clip(np.random.normal(stats.get("spo2",{}).get("mean",97), 0.8, total), 85, 100)
                    hr = np.clip(np.random.normal(stats.get("hr",{}).get("mean",75), 6, total), 40, 200)
                    steps = np.cumsum(np.random.rand(total) < 0.08).astype(int)
                    parts.append(pd.DataFrame({
                        "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts],
                        "patient_id": pid,
                        "accel_x": accel_x,
                        "accel_y": accel_y,
                        "accel_z": accel_z,
                        "emg": emg,
                        "spo2": spo2,
                        "hr": hr,
                        "step_count": steps
                    }))
                df_gen = pd.concat(parts, ignore_index=True)
                os.makedirs("data", exist_ok=True)
                fname = f"data/generated_{n_pat}p_{hours}h_{hz}hz.csv"
                df_gen.to_csv(fname, index=False)
                # also write to AUTOGEN_PATH for quick loads in other tabs
                try:
                    df_gen.to_csv(AUTOGEN_PATH, index=False)
                except Exception:
                    pass
                st.success(f"Generated dataset: {fname}")
                st.metric("Rows", len(df_gen))
                st.download_button("Download CSV", data=df_gen.to_csv(index=False).encode("utf-8"), file_name="synthetic_dataset.csv", mime="text/csv")

        st.markdown("### 🧠 AI Training Simulation (visual)")
        if st.button("🚀 Simulate Training (visual)"):
            st.info("Running training simulation (demo only)...")
            epochs = 18
            accs = []
            prog = st.progress(0)
            chart = st.empty()
            brain = st.empty()
            n_neurons = 80
            xs = np.random.uniform(-1,1,n_neurons); ys = np.random.uniform(-1,1,n_neurons); zs = np.random.uniform(-1,1,n_neurons)

            for e in range(1, epochs+1):
                acc = 60 + 40*(1 - np.exp(-e/6)) + np.random.normal(0,1.2)
                accs.append(acc)
                prog.progress(e/epochs)

                # brain pulse update
                pulse = 0.25 + 0.75*np.sin(e/2 + np.linspace(0, 2*np.pi, n_neurons))
                colors = np.clip(0.2 + pulse/2, 0, 1)
                brain_fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                                                         marker=dict(size=6, color=colors, colorscale='Viridis'))])
                brain_fig.update_layout(height=320, margin=dict(l=0,r=0,t=30,b=0),
                                        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
                brain.plotly_chart(brain_fig, use_container_width=True)

                # accuracy chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(1,len(accs)+1)), y=accs, mode='lines+markers', name='Accuracy'))
                fig.update_layout(height=300, title=f"Epoch {e}/{epochs} - Acc {acc:.2f}%", yaxis=dict(range=[50,100]))
                chart.plotly_chart(fig, use_container_width=True)

                # small blocking sleep for animation effect (short)
                time.sleep(0.18)

            st.success("Simulation finished!")
            st.metric("Final Accuracy", f"{accs[-1]:.2f}%")
            # Use final accuracy to set a small boost to recovery curve (demo-only)
            boost = (accs[-1] - 60) / 300.0  # small fraction
            st.session_state["training_boost"] = boost

# End of app
