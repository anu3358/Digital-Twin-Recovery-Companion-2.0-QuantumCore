from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, JSON, Float, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(String, default='patient')
    created_at = Column(DateTime, default=datetime.utcnow)

class PatientProfile(Base):
    __tablename__ = 'patient_profiles'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    demographics = Column(JSON, nullable=True)
    medical_history = Column(Text, nullable=True)
    twin_state = Column(JSON, nullable=True)
    user = relationship('User')

class SensorStream(Base):
    __tablename__ = 'sensor_streams'
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey('patient_profiles.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    sensor_type = Column(String)
    payload = Column(JSON)

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey('patient_profiles.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String, nullable=True)
    result = Column(JSON)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    action = Column(String)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship('User')
