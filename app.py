import os
import io
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from database import SessionLocal, engine, Base
from models import User, PatientProfile, SensorStream, Prediction, TwinModel

# Safe import of authentication helpers
try:
    # util/auth.py should define: authenticate(db, email, password), verify_password(plain, hashed), pwd_context
    from util.auth import authenticate, verify_password, pwd_context
except Exception:
    # Fallback basic implementation if util.auth is missing (uses pbkdf2_sha256)
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

from audit import log_action
from report import generate_report
from data_ingestion import parse_and_store

from passlib.context import CryptContext  # used in admin user creation / seed

st.set_page_config(page_title="Digital-Twin Recovery Companion", layout="wide")

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

# Seed (guarded)
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
                u = User(email=email, hashed_password=pwd.hash("changeme"), role=role, full_name=name)
                db.add(u); db.commit(); db.refresh(u)
                if role == "patient":
                    p = PatientProfile(user_id=u.id, demographics={"age":45}, medical_history="Demo")
                    db.add(p); db.commit()
    except Exception as e:
        print("Seed error:", e)
    finally:
        db.close()

if os.getenv("SEED_ON_STARTUP","0") == "1":
    seed_demo()

# session state
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# sidebar
with st.sidebar:
    st.title("Digital-Twin")
    judge = st.checkbox("🎯 Judge Mode (auto-demo)", value=True)
    st.caption("Auto-login and preload demo data for instant walkthrough.")

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

# tabs
tabs = ["🏠 Overview","📥 Data Ingestion"]
if role in ["clinician","admin"]:
    tabs.append("👩‍⚕️ Clinician")
if role == "admin":
    tabs.append("🛠️ Admin")
tabs.append("🧬 Data Generator")
active = st.tabs(tabs)

# Overview
with active[0]:
    col1,col2 = st.columns([2,1], gap="large")
    with col1:
        st.subheader("Recovery Progress")
        days = list(range(0,30))
        values = [0.4 + i*0.02 + np.sin(i/3)*0.01 for i in days]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days,y=values,mode="lines+markers",name="Recovery"))
        fig.update_layout(height=320,margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig,use_container_width=True)

        st.markdown("#### Digital Twin (Animated - Gait)")
        mode = st.selectbox("Mode", ["Walk","Balance","Step-Up"], key="dt_mode")
        speed = st.selectbox("Speed", ["Slow","Normal","Fast"], key="dt_speed")
        muscle = st.selectbox("Muscle Highlight", ["None","EMG","Fatigue","Pain"], key="dt_muscle")

        # simple animated stick figure (gait cycle)
        t = np.linspace(0,2*np.pi,30)
        base = np.array([[0,1.8,0],[0,1.4,0],[-0.3,1.1,0],[0,1.4,0],[0.3,1.1,0],
                         [0,1.4,0],[0,0.8,0],[-0.2,0.2,0],[0,0.8,0],[0.2,0.2,0]])
        frames = []
        for i in range(len(t)):
            phase = 0.15*np.sin(t[i])
            xs = base[:,0] + np.array([0,0,phase,0,-phase,0,0,phase,0,-phase])
            ys = base[:,1]
            frames.append(go.Frame(data=[go.Scatter3d(x=xs,y=ys,z=[0]*len(xs),mode='lines+markers')]))

        twin_fig = go.Figure(data=[go.Scatter3d(x=base[:,0],y=base[:,1],z=[0]*10,mode='lines+markers')],
                             frames=frames)
        twin_fig.update_layout(scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)),
                               height=360,margin=dict(l=10,r=10,t=10,b=10),
                               updatemenus=[{"type":"buttons","buttons":[{"label":"Play","method":"animate","args":[None]}]}])
        st.plotly_chart(twin_fig,use_container_width=True)

    with col2:
        st.subheader("What-if Simulation")
        extra = st.slider("Extra balance training (min/day)",0,60,10)
        if st.button("Run Simulation", key="sim1"):
            model = TwinModel()
            pred = model.predict(patient_id=st.session_state.user.id, scenario={"extra_minutes_balance":extra})
            log_action(SessionLocal(), st.session_state.user.id, "prediction", {"extra_minutes":extra})
            st.metric("Predicted gait speed Δ", f"{pred['gait_speed_change_pct']} %")
            st.metric("Adherence score", f"{pred['adherence_score']}")

# Data ingestion
with active[1]:
    st.subheader("📥 Ingest Wearable CSV")
    st.write("Upload CSV with columns: timestamp, accel_x, accel_y, accel_z, emg, spo2, hr, step_count")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    feats_key = "latest_feats"
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            req = ["timestamp","accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]
            miss = [c for c in req if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                st.success("CSV loaded")
                # show signal charts
                st.markdown("**Accelerometer (x,y,z)**")
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=df['timestamp'], y=df['accel_x'], name='acc_x'))
                fig_acc.add_trace(go.Scatter(x=df['timestamp'], y=df['accel_y'], name='acc_y'))
                fig_acc.add_trace(go.Scatter(x=df['timestamp'], y=df['accel_z'], name='acc_z'))
                fig_acc.update_layout(height=250, margin=dict(l=10,r=10,t=20,b=10))
                st.plotly_chart(fig_acc, use_container_width=True)
                st.markdown("**EMG**")
                fig_emg = go.Figure()
                fig_emg.add_trace(go.Scatter(x=df['timestamp'], y=df['emg'], name='emg'))
                fig_emg.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_emg, use_container_width=True)
                st.markdown("**Heart Rate**")
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Scatter(x=df['timestamp'], y=df['hr'], name='hr', line=dict(color='red')))
                fig_hr.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_hr, use_container_width=True)

                # features
                df['accel_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
                feats = {
                    'accel_mag_mean': float(df['accel_mag'].mean()),
                    'accel_mag_std': float(df['accel_mag'].std()),
                    'emg_rms': float(np.sqrt((df['emg']**2).mean())),
                    'hr_mean': float(df['hr'].mean()),
                    'spo2_mean': float(df['spo2'].mean()),
                    'cadence_est': float(df['step_count'].diff().clip(lower=0).fillna(0).mean()*60)
                }
                st.json(feats)
                st.session_state[feats_key] = feats

                # store a small sample to DB
                db = SessionLocal()
                patient_profile = db.query(PatientProfile).filter(PatientProfile.user_id==st.session_state.user.id).first()
                if not patient_profile:
                    patient_profile = PatientProfile(user_id=st.session_state.user.id, demographics={}, medical_history='')
                    db.add(patient_profile); db.commit(); db.refresh(patient_profile)
                try:
                    sample = df.head(200)
                    for _, row in sample.iterrows():
                        payload = {'accel':[row['accel_x'],row['accel_y'],row['accel_z']],'emg':float(row['emg']),
                                   'spo2':float(row['spo2']),'hr':float(row['hr']),'step_count':int(row['step_count'])}
                        s = SensorStream(patient_id=patient_profile.id, sensor_type='wearable_csv', payload=payload)
                        db.add(s)
                    db.commit()
                    log_action(db, st.session_state.user.id, 'csv_upload', {'rows': len(df)})
                    st.success('Stored sample rows to DB')
                except Exception as e:
                    st.error('DB store failed: '+str(e))
                finally:
                    db.close()
        except Exception as e:
            st.error('Failed to read CSV: '+str(e))

    st.markdown('#### Predict from Uploaded Features')
    extra2 = st.slider('Extra balance training (min/day)',0,60,15)
    if st.button('Run Prediction from Features'):
        feats = st.session_state.get(feats_key)
        if not feats:
            st.warning('Please upload CSV first.')
        else:
            model = TwinModel()
            res = model.predict(patient_id=st.session_state.user.id, scenario={'extra_minutes_balance':extra2}, feats=feats)
            log_action(SessionLocal(), st.session_state.user.id, 'prediction', {'extra_minutes':extra2})
            st.metric('Predicted gait speed Δ', f"{res['gait_speed_change_pct']} %")
            st.metric('Adherence score', f"{res['adherence_score']}")

# Clinician (simple)
if role in ['clinician','admin']:
    idx = tabs.index('👩‍⚕️ Clinician') if '👩‍⚕️ Clinician' in tabs else None
    if idx is not None:
        with active[idx]:
            st.subheader('Clinician Dashboard')
            db = SessionLocal()
            patients = db.query(PatientProfile).all()
            rows = []
            for p in patients:
                u = db.query(User).filter(User.id==p.user_id).first()
                rows.append({'id':p.id,'name':u.full_name or u.email})
            st.table(rows)
            db.close()

# Admin
if role == 'admin':
    idx_admin = tabs.index('🛠️ Admin') if '🛠️ Admin' in tabs else None
    if idx_admin is not None:
        with active[idx_admin]:
            st.subheader('Admin Tools')
            db = SessionLocal()
            full_name = st.text_input('Full name')
            email_new = st.text_input('Email')
            pw_new = st.text_input('Password', type='password')
            role_new = st.selectbox('Role', ['patient','clinician'])
            if st.button('Create User'):
                if not email_new or not pw_new:
                    st.error('Email and password required')
                else:
                    if db.query(User).filter(User.email==email_new).first():
                        st.error('Email exists')
                    else:
                        pwd = CryptContext(schemes=['pbkdf2_sha256'], deprecated='auto')
                        u = User(email=email_new, hashed_password=pwd.hash(pw_new), role=role_new, full_name=full_name)
                        db.add(u); db.commit(); db.refresh(u)
                        if role_new == 'patient':
                            db.add(PatientProfile(user_id=u.id, demographics={}, medical_history=''))
                            db.commit()
                        st.success('User created')
            st.markdown('---')
            st.code(f"DB = {os.getenv('DATABASE_URL','sqlite:///./data/app.db')}")
            db.close()

# Data Generator + training sim with brain animation (last tab)
if tabs[-1] == '🧬 Data Generator':
    with active[-1]:
        st.subheader('🧬 Synthetic Dataset Generator')
        st.write('Create wearable datasets for testing & demo.')
        n_pat = st.slider('Patients',1,20,3)
        hours = st.slider('Hours per patient',1,72,12)
        hz = st.slider('Sampling freq (Hz)',1,10,1)
        mode = st.selectbox('Activity mode',['mixed','low','medium','high'])

        if st.button('⚙️ Generate Dataset'):
            with st.spinner('Generating...'):
                progress = st.progress(0)
                parts = []
                for i in range(1,n_pat+1):
                    lvl = np.random.choice(['low','medium','high']) if mode=='mixed' else mode
                    # quick small generator to keep memory reasonable in cloud
                    total = hours*3600*hz
                    ts = [datetime.now() - timedelta(seconds=j/hz) for j in range(total)]
                    ts.reverse()
                    accel_x = np.random.normal(0,0.6,total)
                    accel_y = np.random.normal(0,0.6,total)
                    accel_z = np.random.normal(1,0.05,total)
                    emg = np.abs(np.random.normal(0.6,0.25,total))
                    spo2 = np.clip(np.random.normal(97,0.8,total),90,100)
                    hr = np.clip(np.random.normal(75,6,total),50,160)
                    steps = np.cumsum(np.random.rand(total) < 0.1).astype(int)
                    dfp = pd.DataFrame({'timestamp':[t.strftime('%Y-%m-%d %H:%M:%S') for t in ts],
                                        'patient_id':i,'accel_x':accel_x,'accel_y':accel_y,'accel_z':accel_z,
                                        'emg':emg,'spo2':spo2,'hr':hr,'step_count':steps})
                    parts.append(dfp)
                    progress.progress(i/n_pat)
                df_all = pd.concat(parts, ignore_index=True)
                os.makedirs("data", exist_ok=True)
                fname = f'data/generated_{n_pat}p_{hours}h_{hz}hz.csv'
                df_all.to_csv(fname, index=False)
                st.success('Dataset generated: '+fname)
                st.metric('Rows', len(df_all))
                st.download_button('Download CSV', data=df_all.to_csv(index=False).encode('utf-8'),
                                   file_name='synthetic_dataset.csv', mime='text/csv')

        st.markdown('### 🧠 AI Training Simulation (visual)')
        if st.button('🚀 Simulate Training'):
            st.info('Running training simulation...')
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
                # brain
                pulse = 0.3 + 0.7*np.sin(e/2 + np.linspace(0,2*np.pi,n_neurons))
                colors = np.clip(0.2 + pulse/2, 0,1)
                brain_fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers',
                                                         marker=dict(size=6, color=colors, colorscale='Viridis'))])
                brain_fig.update_layout(height=320, margin=dict(l=0,r=0,t=30,b=0),
                                        scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)))
                brain.plotly_chart(brain_fig, use_container_width=True)
                # chart
                fig = go.Figure(); fig.add_trace(go.Scatter(x=list(range(1,len(accs)+1)), y=accs, mode='lines+markers'))
                fig.update_layout(height=300, title=f'Epoch {e}/{epochs} - Acc {acc:.2f}%', yaxis=dict(range=[50,100]))
                chart.plotly_chart(fig, use_container_width=True)
                st.sleep(0.2)
            st.success('Simulation finished!'); st.metric('Final Accuracy', f'{accs[-1]:.2f}%')

# End
