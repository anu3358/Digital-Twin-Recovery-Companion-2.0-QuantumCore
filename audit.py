# audit.py
import json
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, JSON as SA_JSON, Text
from sqlalchemy.exc import SQLAlchemyError

# import DB Base & Session convenience
from database import Base, SessionLocal

# Try to import AuditLog from models if user defined it there
try:
    # models may or may not export AuditLog
    import models
    AuditLog = getattr(models, "AuditLog", None)
except Exception:
    AuditLog = None

# If AuditLog not present in models, create a lightweight mapped AuditLog here
if AuditLog is None:
    # define a mapped class using the shared Base so it's registered with SQLAlchemy
    class AuditLog(Base):
        __tablename__ = "audit_logs"
        id = Column(Integer, primary_key=True, index=True)
        user_id = Column(Integer, nullable=True, index=True)
        action = Column(String(128), nullable=False, index=True)
        payload = Column(SA_JSON, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)

    # expose it on module for other code that expects models.AuditLog
    # (so subsequent imports find it if they inspect audit.AuditLog)
    globals()["AuditLog"] = AuditLog

def log_action(db, user_id, action, payload=None):
    """
    Insert an audit log row using the provided SQLAlchemy session (db).
    If db is None, it will create a short-lived session to write the log.
    payload may be a dict or JSON-serializable object.
    """
    # Normalize payload to JSON-serializable form
    serializable = None
    try:
        if payload is None:
            serializable = None
        elif isinstance(payload, (str, int, float, list, dict)):
            serializable = payload
        else:
            # try to JSON-serialize arbitrary objects
            serializable = json.loads(json.dumps(payload, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        # Best-effort fallback
        try:
            serializable = str(payload)
        except Exception:
            serializable = {"unserializable_payload": True}

    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True

    try:
        row = AuditLog(user_id=user_id, action=action, payload=serializable)
        db.add(row)
        db.commit()
    except SQLAlchemyError as e:
        # rollback & fallback to printing to console (Streamlit logs)
        try:
            db.rollback()
        except Exception:
            pass
        print(f"[audit] DB write failed: {e}; action={action}; user_id={user_id}; payload={serializable}")
    except Exception as e:
        print(f"[audit] Unexpected error writing audit: {e}")
    finally:
        if close_session:
            try:
                db.close()
            except Exception:
                pass
