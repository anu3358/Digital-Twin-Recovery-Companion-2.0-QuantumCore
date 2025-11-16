# database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

# SQLite needs check_same_thread=False for multi-threaded servers like Streamlit
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# Use future=True for SQLAlchemy 1.4/2.0 compatibility
engine = create_engine(DATABASE_URL, future=True, connect_args=connect_args)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# IMPORTANT: only create declarative_base() here — import this Base from models.py
Base = declarative_base()
