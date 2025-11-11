from database import SessionLocal
from models import AuditLog
from datetime import datetime

def log_action(db, user_id: int, action: str, meta: dict = None):
    try:
        entry = AuditLog(user_id=user_id, action=action, metadata=meta or {}, timestamp=datetime.utcnow())
        db.add(entry); db.commit()
    except Exception as e:
        print('Audit log failed:', e)
