import pandas as pd
import numpy as np
from models import SensorStream
from datetime import datetime

EXPECTED_COLS = ["timestamp","accel_x","accel_y","accel_z","emg","spo2","hr","step_count"]

def parse_and_store(csv_bytes: bytes, patient_id: int, db):
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected {EXPECTED_COLS}")
    # store a sample of rows as sensor_stream entries
    max_rows = min(2000, len(df))
    for i, row in df.head(max_rows).iterrows():
        payload = {
            "accel": [float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])],
            "emg": float(row["emg"]),
            "spo2": float(row["spo2"]),
            "hr": float(row["hr"]),
            "step_count": float(row["step_count"]),
        }
        s = SensorStream(patient_id=patient_id, timestamp=datetime.utcnow(), sensor_type="wearable_csv", payload=payload)
        db.add(s)
    db.commit()
    # feature engineering
    acc_mag = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
    feats = {
        "acc_mag_mean": float(acc_mag.mean()),
        "acc_mag_std": float(acc_mag.std()),
        "emg_rms": float(np.sqrt((df["emg"]**2).mean())),
        "hr_mean": float(df["hr"].mean()),
        "spo2_mean": float(df["spo2"].mean()),
        "cadence_est": float(df["step_count"].diff().clip(lower=0).fillna(0).mean() * 60)
    }
    return df.head(200), feats
