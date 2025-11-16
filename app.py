# app.py - Digital-Twin Recovery Companion (adapted to your dataset)
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# local imports (DB models/util)
from database import SessionLocal, engine, Base
from models import User, PatientProfile, SensorStream, Prediction, TwinModel

# Safe import of util.auth (fallback included)
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

from audit import log_action
from report import generate_report
from data_ingestion import parse_and_store  # optional; we use our own simple parser below

st.set_page_config(page_title="Digital-Twin Recovery Companion", layout="wide")
Base.metadata.create_all(bind=engine)

# ---------- Seed (guarded) ----------
def seed_demo():
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == "admin@example.com").first():
            pwd = pwd_context
            for email, role, name in [
                ("admin@example.com", "admin", "Admin User"),
                ("clinician@example.com", "clinician", "Clinician One"),
                ("patient@example.com", "patient", "Patient One"),
            ]:
                u = User(email=email, hashed_password=pwd.hash("changeme"), role=role, full_name=name)
                db.add(u); db.commit(); db.refresh(u)
                if role == "patient":
                    p = PatientProfile(user_id=u.id, demographics={"age":45}, medical_history="Demo")
                    db.add(p); db.commit()
    except Exception as e:
        print("Seed error:", e)
    finally:
        db.close()

if os.getenv("SEED_ON_STARTUP", "0") == "1":
    seed_demo()

# ---------- session ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ---------- Sidebar (login / judge) ----------
with st.sidebar:
    st.title("Digital-Twin")
    judge = st.checkbox("🎯 Judge Mode (auto-demo)", value=True)
    st.caption("Auto-login and preload demo/generated data for instant walkthrough.")
    if judge and not st.session_state.user:
        db = SessionLocal()
        user = db.query(User).filter(User.email == "patient@example.com").first()
        if user:
            st.session_state.user = user
            st.session_state.role = "patient"
            st.success("Auto-logged in as patient@example.com")
        db.close()

    if st.session_state.user:
        st.success(f"Logged in: {st.session_state.user.email} ({st.session_state.role})")
        if st.button("Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.role = None
            st.rerun()
    else:
        st.subheader("Login")
        email = st.text_input("Email", value="patient@example.com")
        password = st.text_input("Password", value="changeme", type="password")
        role_pick = st.selectbox("Role", options=["patient","clinician","admin"])
        if st.button("Sign in", use_container_width=True):
            db = SessionLocal()
            user = authenticate(db, email, password)
            if user and (role_pick == user.role or role_pick == "admin"):
                st.session_state.user = user
                st.session_state.role = user.role
                log_action(db, user.id, "login", {"role": user.role})
                st.rerun()
            else:
                st.error("Invalid credentials or role mismatch")
            db.close()

st.title("Digital-Twin Recovery Companion")

if not st.session_state.user:
    st.info("Please log in from the sidebar to continue.")
    st.stop()

role = st.session_state.role

# ---------- helper utilities ----------
def infer_sampling_hz(df):
    if "timestamp" not in df.columns:
        return 1
    try:
        ts = pd.to_datetime(df["timestamp"])
        diffs = ts.sort_values().diff().dt.total_seconds().dropna()
        med = diffs.median() if len(diffs) else 1.0
        return max(1, int(round(1.0 / med)))
    except Exception:
        return 1

def extract_features(df):
    # expects accel_x, accel_y, accel_z, emg, spo2, hr, step_count
    feats = {}
    df2 = df.copy()
    # ensure numeric
    for c in ["accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)
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
    # cadence: steps per minute (estimate)
    if "step_count" in df2.columns:
        # if step_count cumulative, compute diff rate
        steps_diff = df2["step_count"].diff().clip(lower=0).fillna(0)
        # compute samples_per_min from timestamps if possible
        hz = infer_sampling_hz(df2)
        cadence = steps_diff.mean() * 60 * hz
        feats["cadence_est"] = float(cadence)
        feats["total_steps"] = int(df2["step_count"].max())
    return feats

def load_csv_from_path(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

# auto-load a generated dataset if present (from previous generation)
AUTOGEN_PATH = Path("/mnt/data/generated_from_uploaded_5p_17h_1hz.csv")
if AUTOGEN_PATH.exists():
    st.sidebar.success("Found generated dataset (auto-load available)")

# ---------- tabs ----------
tabs = ["🏠 Overview", "📥 Data Ingestion"]
if role in ["clinician","admin"]:
    tabs.append("👩‍⚕️ Clinician")
if role == "admin":
    tabs.append("🛠️ Admin")
tabs.append("🧬 Data Generator")
active = st.tabs(tabs)

# ---------- Overview ----------
with active[0]:
    col_main, col_right = st.columns([2,1], gap="large")
    with col_main:
        st.subheader("Recovery Progress (Demo)")
        days = np.arange(30)
        values = [0.4 + i*0.02 + np.sin(i/3)*0.01 for i in days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=values, mode="lines+markers", name="Recovery"))
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Digital Twin — Animated Gait (demo)")
        # simple animated stick figure with option to choose patient from dataset if available
        available_patients = []
        # try to populate from session if any df was uploaded/loaded earlier
        if "last_uploaded_df" in st.session_state and st.session_state["last_uploaded_df"] is not None:
            df_check = st.session_state["last_uploaded_df"]
            if "patient_id" in df_check.columns:
                available_patients = sorted(pd.unique(df_check["patient_id"]).tolist())
        elif AUTOGEN_PATH.exists():
            df_aut = load_csv_from_path(AUTOGEN_PATH)
            if df_aut is not None and "patient_id" in df_aut.columns:
                available_patients = sorted(pd.unique(df_aut["patient_id"]).tolist())

        patient_choice = st.selectbox("Show patient", options=["Auto"] + [str(x) for x in available_patients], index=0)
        mode = st.selectbox("Mode", ["Walk","Balance","Step-Up"], key="dt_mode")
        speed = st.selectbox("Speed", ["Slow","Normal","Fast"], key="dt_speed")
        muscle = st.selectbox("Muscle Highlight", ["None","EMG","Fatigue","Pain"], key="dt_muscle")

        # stick-figure (same logic as before)
        t = np.linspace(0, 2*np.pi, 30)
        base = np.array([[0,1.8,0],[0,1.4,0],[-0.3,1.1,0],[0,1.4,0],[0.3,1.1,0],
                         [0,1.4,0],[0,0.8,0],[-0.2,0.2,0],[0,0.8,0],[0.2,0.2,0]])
        frames = []
        for i in range(len(t)):
            phase = 0.12 * np.sin(t[i] * (1.0 if speed=="Normal" else (0.6 if speed=="Slow" else 1.6)))
            xs = base[:,0] + np.array([0,0,phase,0,-phase,0,0,phase,0,-phase])
            ys = base[:,1]
            frames.append(go.Frame(data=[go.Scatter3d(x=xs, y=ys, z=[0]*len(xs), mode='lines+markers')]))
        twin_fig = go.Figure(data=[go.Scatter3d(x=base[:,0], y=base[:,1], z=[0]*10, mode='lines+markers')], frames=frames)
        twin_fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
                               height=360, margin=dict(l=10,r=10,t=10,b=10),
                               updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None]}]}])
        st.plotly_chart(twin_fig, use_container_width=True)

    with col_right:
        st.subheader("What-if Simulation")
        extra = st.slider("Extra balance training (min/day)", 0, 60, 10)
        if st.button("Run Simulation", key="overview_sim"):
            model = TwinModel()
            pred = model.predict(patient_id=st.session_state.user.id, scenario={"extra_minutes_balance": extra})
            log_action(SessionLocal(), st.session_state.user.id, "prediction", {"extra_minutes": extra})
            st.metric("Predicted gait speed Δ", f"{pred.get('gait_speed_change_pct',0)} %")
            st.metric("Adherence score", f"{pred.get('adherence_score',0)}")

# ---------- Data Ingestion ----------
with active[1]:
    st.subheader("📥 Ingest Wearable CSV (your schema)")
    st.write("Expect columns: `timestamp, patient_id (optional), accel_x, accel_y, accel_z, emg, spo2, hr, step_count`.")
    st.markdown("- You can upload your CSV, or load the auto-generated sample (if present).")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
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
        df = load_csv_from_path(AUTOGEN_PATH)
        if df is None:
            st.error("Failed to load auto-generated dataset.")

    if df is not None:
        # normalize column names (strip)
        df.columns = [c.strip() for c in df.columns]
        st.session_state["last_uploaded_df"] = df  # keep for overview use
        st.markdown("### Dataset preview")
        st.dataframe(df.head(200), use_container_width=True)

        # patient selection if present
        patients = []
        if "patient_id" in df.columns:
            patients = sorted(pd.unique(df["patient_id"]).tolist())
            st.markdown(f"Detected patient IDs: {patients}")
            pid_sel = st.selectbox("Select patient (for per-patient plots & features)", options=["All"] + [str(x) for x in patients], index=0)
        else:
            pid_sel = "All"

        # plotting (subsample timestamps if too many)
        plot_df = df if pid_sel == "All" else df[df["patient_id"].astype(str) == str(pid_sel)]
        # limit to 2000 points for plotting speed
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(n=2000, random_state=42).sort_index()

        # Accel plot
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

        # features (per selection)
        feat_df = df if pid_sel == "All" else df[df["patient_id"].astype(str) == str(pid_sel)]
        feats = extract_features(feat_df)
        st.markdown("#### Extracted features (for prediction)")
        st.json(feats)
        st.session_state["latest_feats"] = feats

        # store small sample to DB (first N rows) for auditing
        if st.button("Store small sample to DB"):
            db = SessionLocal()
            try:
                patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id == st.session_state.user.id).first()
                if not patient_profile:
                    patient_profile = PatientProfile(user_id=st.session_state.user.id, demographics={}, medical_history="")
                    db.add(patient_profile); db.commit(); db.refresh(patient_profile)
                sample = df.head(200)
                for _, row in sample.iterrows():
                    payload = {}
                    # store a compact payload
                    for col in ["accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]:
                        if col in row.index:
                            payload[col] = float(row[col]) if not pd.isna(row[col]) else 0.0
                    s = SensorStream(patient_id=patient_profile.id, sensor_type="wearable_csv", payload=payload)
                    db.add(s)
                db.commit()
                log_action(db, st.session_state.user.id, "csv_upload", {"rows": len(df)})
                st.success("Stored sample rows to DB")
            except Exception as e:
                st.error("DB store failed: " + str(e))
            finally:
                db.close()

        # Predict from features
        st.markdown("---")
        st.markdown("#### Predict from extracted features")
        extra_minutes = st.slider("Extra balance training (min/day)", 0, 60, 15, key="predict_extra")
        if st.button("Run Prediction from Features"):
            feats = st.session_state.get("latest_feats")
            if not feats:
                st.warning("Please upload/select data first.")
            else:
                model = TwinModel()
                res = model.predict(patient_id=st.session_state.user.id, scenario={"extra_minutes_balance": extra_minutes}, feats=feats)
                log_action(SessionLocal(), st.session_state.user.id, "prediction", {"extra_minutes": extra_minutes})
                st.metric("Predicted gait speed Δ", f"{res.get('gait_speed_change_pct',0)} %")
                st.metric("Adherence score", f"{res.get('adherence_score',0)}")

# ---------- Clinician ----------
if role in ["clinician","admin"]:
    idx = tabs.index("👩‍⚕️ Clinician") if "👩‍⚕️ Clinician" in tabs else None
    if idx is not None:
        with active[idx]:
            st.subheader("Clinician Dashboard")
            db = SessionLocal()
            patients = db.query(PatientProfile).all()
            rows = []
            for p in patients:
                u = db.query(User).filter(User.id == p.user_id).first()
                rows.append({"id": p.id, "name": u.full_name or u.email})
            st.table(rows)
            db.close()

# ---------- Admin ----------
if role == "admin":
    idx_admin = tabs.index("🛠️ Admin") if "🛠️ Admin" in tabs else None
    if idx_admin is not None:
        with active[idx_admin]:
            st.subheader("Admin Tools")
            db = SessionLocal()
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
            db.close()

# ---------- Data Generator & Training Simulation ----------
if tabs[-1] == "🧬 Data Generator":
    with active[-1]:
        st.subheader("🧬 Synthetic Dataset Generator (based on your sample)")
        # prefill parameters based on uploaded sample if available
        sample_df = None
        if "last_uploaded_df" in st.session_state and st.session_state["last_uploaded_df"] is not None:
            sample_df = st.session_state["last_uploaded_df"]
        elif AUTOGEN_PATH.exists():
            sample_df = load_csv_from_path(AUTOGEN_PATH)

        default_hz = infer_sampling_hz(sample_df) if sample_df is not None else 1
        n_pat = st.slider("Patients", 1, 20, 5)
        hours = st.slider("Hours per patient", 1, 48, 4)
        hz = st.slider("Sampling freq (Hz)", 1, 10, default_hz)
        mode = st.selectbox("Activity mode", ["mixed", "low", "medium", "high"])

        if st.button("⚙️ Generate Dataset (based on sample)"):
            with st.spinner("Generating..."):
                parts = []
                for pid in range(1, n_pat+1):
                    lvl = np.random.choice(["low","medium","high"]) if mode == "mixed" else mode
                    total = hours * 3600 * hz
                    ts = [datetime.now() - timedelta(seconds=j / hz) for j in range(total)]
                    ts.reverse()
                    # use sample stats if available
                    stats = {}
                    if sample_df is not None:
                        for c in ["accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]:
                            if c in sample_df.columns:
                                stats[c] = {"mean": float(sample_df[c].mean()), "std": float(sample_df[c].std())}
                    # generate signals tuned to stats
                    tvals = np.linspace(0, 10*np.pi, total)
                    accel_x = (stats.get("accel_x",{}).get("mean",0) +
                               stats.get("accel_x",{}).get("std",0)*np.sin(tvals + pid) +
                               np.random.normal(0, stats.get("accel_x",{}).get("std",0.5), total))
                    accel_y = (stats.get("accel_y",{}).get("mean",0) +
                               stats.get("accel_y",{}).get("std",0)*np.cos(tvals + pid) +
                               np.random.normal(0, stats.get("accel_y",{}).get("std",0.5), total))
                    accel_z = (stats.get("accel_z",{}).get("mean",1) +
                               np.random.normal(0, stats.get("accel_z",{}).get("std",0.05), total))
                    emg = np.abs(np.random.normal(stats.get("emg",{}).get("mean",0.6), stats.get("emg",{}).get("std",0.3), total))
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
                st.success("Generated dataset: " + fname)
                st.metric("Rows", len(df_gen))
                st.download_button("Download CSV", data=df_gen.to_csv(index=False).encode("utf-8"),
                                   file_name="synthetic_dataset.csv", mime="text/csv")

        st.markdown("### 🧠 AI Training Simulation (visual)")
        if st.button("🚀 Simulate Training (visual)"):
            st.info("Running training simulation...")
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
                pulse = 0.3 + 0.7*np.sin(e/2 + np.linspace(0,2*np.pi,n_neurons))
                colors = np.clip(0.2 + pulse/2, 0, 1)
                brain_fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                                                         marker=dict(size=6, color=colors, colorscale='Viridis'))])
                brain_fig.update_layout(height=320, margin=dict(l=0,r=0,t=30,b=0),
                                        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
                brain.plotly_chart(brain_fig, use_container_width=True)
                fig = go.Figure(); fig.add_trace(go.Scatter(x=list(range(1,len(accs)+1)), y=accs, mode='lines+markers'))
                fig.update_layout(height=300, title=f"Epoch {e}/{epochs} - Acc {acc:.2f}%", yaxis=dict(range=[50,100]))
                chart.plotly_chart(fig, use_container_width=True)
                st.sleep(0.18)
            st.success("Simulation finished!"); st.metric("Final Accuracy", f"{accs[-1]:.2f}%")

# ---------- end ----------
