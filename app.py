import os
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from database import SessionLocal, engine, Base
from models import User, PatientProfile, TwinModel
from util.auth import authenticate
from audit import log_action
from report import generate_report
from data_ingestion import parse_and_store
from passlib.context import CryptContext

st.set_page_config(page_title="Digital-Twin Recovery Companion", layout="wide")

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

# ---------- SAFE SEED ----------
def seed_demo():
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == "admin@example.com").first():
            pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
            for email, role, name in [
                ("admin@example.com", "admin", "Admin User"),
                ("clinician@example.com", "clinician", "Clinician One"),
                ("patient@example.com", "patient", "Patient One"),
            ]:
                u = User(email=email,
                         hashed_password=pwd.hash("changeme"),
                         role=role,
                         full_name=name)
                db.add(u)
                db.commit()
                db.refresh(u)
                if role == "patient":
                    p = PatientProfile(user_id=u.id,
                                       demographics={"age": 45},
                                       medical_history="post-op knee")
                    db.add(p)
                    db.commit()
            print("✅ Seeded demo users successfully.")
    except Exception as e:
        print(f"⚠️ Seed skipped/failed: {e}")
    finally:
        db.close()

if os.getenv("SEED_ON_STARTUP", "0") == "1":
    seed_demo()

# ---------- SESSION ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🏥 Digital-Twin")
    judge = st.toggle("🎯 Judge Mode (auto-demo)", value=True)
    st.caption("Auto-login demo user instantly.")

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
        role_pick = st.selectbox("Role", ["patient", "clinician", "admin"])
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

# ---------- TAB SETUP ----------
tabs = ["🏠 Overview", "📥 Data Ingestion"]
if role in ["clinician", "admin"]:
    tabs.append("👩‍⚕️ Clinician")
if role == "admin":
    tabs.append("🛠️ Admin")
tabs.append("🧬 Data Generator")

active_tab = st.tabs(tabs)

# ---------- OVERVIEW ----------
with active_tab[0]:
    col_info, col_sim = st.columns([2, 1], gap="large")
    with col_info:
        st.subheader("Recovery Progress")
        days = np.arange(30)
        values = [0.35 + i * 0.02 + np.sin(i / 3) * 0.01 for i in days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=values, mode="lines+markers", name="Recovery Index"))
        fig.update_layout(height=320, title="30-Day Recovery Trajectory",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Mood & Pain Heatmap")
        timeslots = ["08:00", "12:00", "16:00", "20:00"]
        days_labels = [(datetime.today() - timedelta(days=i)).strftime("%a %d")
                       for i in range(6, -1, -1)]
        pain = np.clip(np.random.normal(4, 1.5, (7, 4)), 0, 10)
        heat = go.Figure(data=go.Heatmap(z=pain, x=timeslots, y=days_labels,
                                         zmin=0, zmax=10, colorbar=dict(title="Pain")))
        heat.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(heat, use_container_width=True)

    with col_sim:
        st.subheader("What-if Simulation")
        extra = st.slider("Extra balance training (min/day)", 0, 60, 10)
        if st.button("Run Simulation"):
            model = TwinModel()
            pred = model.predict(patient_id=1, scenario={"extra_minutes_balance": extra})
            log_action(SessionLocal(), st.session_state.user.id,
                       "prediction", {"extra_minutes": extra})
            st.metric("Predicted gait speed Δ", f"{pred['gait_speed_change_pct']} %")
            st.metric("Adherence score", f"{pred['adherence_score']}")
        st.markdown("#### Generate PDF Report")
        if st.button("Download Report"):
            metrics = {"Gait Speed Change %": 12.5,
                       "Adherence Score": 85,
                       "Next Step": "Add 5 min balance training"}
            pdf_bytes = generate_report("Demo Patient", metrics)
            st.download_button("Download PDF", data=pdf_bytes,
                               file_name="recovery_report.pdf", mime="application/pdf")

# ---------- DATA INGESTION ----------
with active_tab[1]:
    st.subheader("📥 Ingest Wearable CSV")
    st.write("Upload CSV with columns: `timestamp, accel_x, accel_y, accel_z, emg, spo2, hr, step_count`.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    feats_key = "latest_feats"
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required = ["timestamp", "accel_x", "accel_y", "accel_z",
                        "emg", "spo2", "hr", "step_count"]
            if missing := [c for c in required if c not in df.columns]:
                st.error(f"Missing columns: {missing}")
                st.stop()
            st.success("✅ Uploaded!")
            st.dataframe(df.head(), use_container_width=True)
            feats = {f"{col}_mean": float(df[col].mean()) for col in df.columns if col != "timestamp"}
            st.session_state[feats_key] = feats
            st.json(feats)
        except Exception as e:
            st.error(f"Error: {e}")
    st.markdown("### 🤖 Predict from Uploaded Features")
    extra2 = st.slider("Extra balance training (min/day)", 0, 60, 15)
    if st.button("Run Prediction from CSV"):
        feats = st.session_state.get(feats_key)
        if not feats:
            st.warning("Upload CSV first.")
        else:
            model = TwinModel()
            res = model.predict(patient_id=1,
                                scenario={"extra_minutes_balance": extra2},
                                feats=feats)
            st.metric("Predicted gait speed Δ", f"{res['gait_speed_change_pct']} %")
            st.metric("Adherence score", f"{res['adherence_score']}")

# ---------- ADMIN ----------
if role == "admin":
    with active_tab[-2]:
        st.subheader("🛠️ Admin Tools")
        db = SessionLocal()
        full_name = st.text_input("Full name")
        email_new = st.text_input("Email")
        pw_new = st.text_input("Password", type="password")
        role_new = st.selectbox("Role", ["patient", "clinician"])
        if st.button("Create User"):
            if not email_new or not pw_new:
                st.error("Email and password required.")
            else:
                if db.query(User).filter(User.email == email_new).first():
                    st.error("Email already exists.")
                else:
                    pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
                    u = User(email=email_new,
                             hashed_password=pwd_context.hash(pw_new),
                             role=role_new,
                             full_name=full_name)
                    db.add(u)
                    db.commit()
                    db.refresh(u)
                    if role_new == "patient":
                        db.add(PatientProfile(user_id=u.id,
                                              demographics={}, medical_history=""))
                        db.commit()
                    st.success(f"Created {role_new}: {email_new}")
        st.markdown("---")
        st.code(f"DB = {os.getenv('DATABASE_URL', 'sqlite:///./data/app.db')}")
        db.close()

# ---------- DATA GENERATOR ----------
# ---------- DATA GENERATOR + AI TRAINING PREVIEW ----------
with active_tab[-1]:
    st.subheader("🧬 Synthetic Dataset Generator")
    st.write("""
    Generate realistic wearable datasets for testing, demo, and model training simulation.
    """)

    n_pat = st.slider("Number of patients", 1, 20, 3)
    hours = st.slider("Hours per patient", 1, 72, 12)
    hz = st.slider("Sampling frequency (Hz)", 1, 10, 1)
    mode = st.selectbox("Activity mode", ["mixed", "low", "medium", "high"], index=0)

    if st.button("⚙️ Generate Synthetic Dataset"):
        with st.spinner("Generating data... please wait ⏳"):
            progress = st.progress(0)
            from datetime import datetime, timedelta
            os.makedirs("data", exist_ok=True)

            def gen_one(pid, hours, hz, mode):
                total = hours * 3600 * hz
                ts = [datetime.now() - timedelta(seconds=i / hz) for i in range(total)]
                ts.reverse()
                if mode == "low":
                    accel_scale, emg_scale, step_prob = 0.3, 0.2, 0.005
                elif mode == "high":
                    accel_scale, emg_scale, step_prob = 2.0, 1.2, 0.2
                else:
                    accel_scale, emg_scale, step_prob = 1.0, 0.6, 0.1
                accel_x = np.sin(np.linspace(0, 20*np.pi, total))*accel_scale + np.random.normal(0,0.1,total)
                accel_y = np.cos(np.linspace(0, 20*np.pi, total))*accel_scale + np.random.normal(0,0.1,total)
                accel_z = np.ones(total) + np.random.normal(0,0.05,total)
                emg = np.abs(np.random.normal(emg_scale,0.25,total))
                spo2 = np.clip(np.random.normal(97,1,total),90,100)
                hr = np.clip(np.random.normal(75+10*accel_scale,8,total),55,180)
                steps = np.cumsum(np.random.rand(total)<step_prob).astype(int)
                return pd.DataFrame({
                    "timestamp":[t.strftime("%Y-%m-%d %H:%M:%S") for t in ts],
                    "patient_id":pid,"accel_x":accel_x,"accel_y":accel_y,
                    "accel_z":accel_z,"emg":emg,"spo2":spo2,"hr":hr,"step_count":steps})

            all_df=[]
            for i, pid in enumerate(range(1,n_pat+1)):
                lvl = np.random.choice(["low","medium","high"]) if mode=="mixed" else mode
                all_df.append(gen_one(pid,hours,hz,lvl))
                progress.progress((i+1)/n_pat)
            df = pd.concat(all_df, ignore_index=True)

            filename = f"data/generated_{n_pat}p_{hours}h_{hz}hz.csv"
            df.to_csv(filename, index=False)
            st.success(f"✅ Dataset ready: {filename}")
            st.metric("Rows generated", len(df))
            st.download_button("⬇️ Download CSV",
                               data=df.to_csv(index=False).encode("utf-8"),
                               file_name="synthetic_dataset.csv",
                               mime="text/csv")

    # ---------- AI TRAINING SIMULATION ----------
    # ---------- AI TRAINING SIMULATION + 3D DIGITAL TWIN BRAIN ----------
    st.markdown("### 🧠 AI Model Training Simulation")
    st.caption("Watch the Digital Twin 'learn' from synthetic data in real time.")

    if st.button("🚀 Simulate Training"):
        st.info("Initializing AI twin neural activity (simulation mode)...")

        epochs = 20
        accuracies, progress = [], st.progress(0)
        chart_area = st.empty()
        brain_area = st.empty()

        # --- setup the 3D brain base ---
        n_neurons = 80
        np.random.seed(42)
        xs = np.random.uniform(-1, 1, n_neurons)
        ys = np.random.uniform(-1, 1, n_neurons)
        zs = np.random.uniform(-1, 1, n_neurons)
        base_colors = np.full(n_neurons, 0.2)

        for epoch in range(1, epochs + 1):
            acc = 60 + 40 * (1 - np.exp(-epoch / 6)) + np.random.normal(0, 1.5)
            accuracies.append(acc)
            progress.progress(epoch / epochs)

            # ---- Update 3D brain pulse ----
            pulse = 0.2 + 0.8 * np.sin(epoch / 3 + np.linspace(0, 2*np.pi, n_neurons))
            colors = np.clip(base_colors + pulse / 2, 0, 1)
            brain_fig = go.Figure(data=[
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers',
                    marker=dict(size=5,
                                color=colors,
                                colorscale='Viridis',
                                opacity=0.8)
                )
            ])
            brain_fig.update_layout(
                title=f"Digital Twin Neural Activity — Epoch {epoch}",
                margin=dict(l=0, r=0, t=40, b=0),
                height=350,
                scene=dict(xaxis=dict(visible=False),
                           yaxis=dict(visible=False),
                           zaxis=dict(visible=False))
            )
            brain_area.plotly_chart(brain_fig, use_container_width=True)

            # ---- Training accuracy chart ----
            chart_fig = go.Figure()
            chart_fig.add_trace(go.Scatter(
                x=list(range(1, len(accuracies) + 1)),
                y=accuracies,
                mode="lines+markers",
                line=dict(color="#00C851", width=3),
                name="Training Accuracy (%)"
            ))
            chart_fig.update_layout(
                height=300,
                title=f"Epoch {epoch}/{epochs} — Accuracy {acc:.2f}%",
                xaxis_title="Epoch",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[50, 100])
            )
            chart_area.plotly_chart(chart_fig, use_container_width=True)
            st.sleep(0.25)

        st.success("🎉 Digital-Twin Brain training simulation complete!")
        st.metric("Final Accuracy", f"{accuracies[-1]:.2f} %")
        st.metric("Training Duration", "≈ 5 seconds (simulated)")
        st.markdown("""
        ✅ **Training Summary**
        - Neural pathways strengthened across 3D twin model  
        - Recovery predictions improved after simulated learning  
        - Demonstrates adaptive AI capability for personalized healthcare
        """)


        # Fake epoch training
        epochs = 20
        accuracies = []
        progress = st.progress(0)
        chart_area = st.empty()

        for epoch in range(1, epochs + 1):
            # Simulate accuracy curve with small random noise
            acc = 60 + 40 * (1 - np.exp(-epoch / 6)) + np.random.normal(0, 1.5)
            accuracies.append(acc)
            progress.progress(epoch / epochs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(accuracies) + 1)),
                y=accuracies,
                mode="lines+markers",
                name="Training Accuracy (%)",
                line=dict(color="#4CAF50", width=3)
            ))
            fig.update_layout(
                title=f"Epoch {epoch}/{epochs} — Model Accuracy: {acc:.2f}%",
                xaxis_title="Epoch",
                yaxis_title="Accuracy (%)",
                yaxis=dict(range=[50, 100]),
                height=350
            )
            chart_area.plotly_chart(fig, use_container_width=True)
            st.sleep(0.25)

        st.success("🎉 Model training simulation complete!")
        st.metric("Final Accuracy", f"{accuracies[-1]:.2f} %")
        st.metric("Training Duration", "≈ 5 seconds (simulated)")

        st.markdown("""
        ✅ **Simulation Summary:**
        - Model adapted using multi-patient dataset  
        - Personalized physiological signals learned  
        - Optimized recovery trajectory predicted  
        - Shows future capability for real-time AI twin learning
        """)
