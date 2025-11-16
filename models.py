# models.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Text
from database import Base

# ----- Mapped ORM classes -----
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(512), nullable=False)
    role = Column(String(32), nullable=False, default="patient")
    full_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PatientProfile(Base):
    __tablename__ = "patient_profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    demographics = Column(JSON, default={})
    medical_history = Column(Text, default="")

class SensorStream(Base):
    __tablename__ = "sensor_streams"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, index=True)
    sensor_type = Column(String(64))
    payload = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, index=True)
    model_name = Column(String(128))
    result = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# ----- Non-mapped helper class (not an ORM model) -----
class TwinModel:
    """
    Lightweight prediction stub — not mapped to DB.
    Replace with your real model class if needed.
    """
    def __init__(self):
        pass

    def predict(self, patient_id, scenario=None, feats=None):
        extra = (scenario or {}).get("extra_minutes_balance", 0)
        base_score = 10 + 0.5 * extra
        adherence = min(100, 70 + extra / 2)
        return {"gait_speed_change_pct": round(base_score, 2), "adherence_score": round(adherence, 1)}
