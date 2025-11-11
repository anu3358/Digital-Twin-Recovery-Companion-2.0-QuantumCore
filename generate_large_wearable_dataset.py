import pandas as pd, numpy as np, os
from datetime import datetime, timedelta
import argparse

def generate_patient_data(patient_id, hours=24, hz=1, activity='medium'):
    np.random.seed(patient_id*123)
    total = int(hours*3600*hz)
    timestamps = [datetime.now() - timedelta(seconds=i/hz) for i in range(total)]
    timestamps.reverse()
    if activity=='low':
        accel_scale, emg_scale, step_prob = 0.3, 0.2, 0.005
    elif activity=='high':
        accel_scale, emg_scale, step_prob = 2.0, 1.2, 0.2
    else:
        accel_scale, emg_scale, step_prob = 1.0, 0.6, 0.1
    accel_x = np.sin(np.linspace(0,20*np.pi,total))*accel_scale + np.random.normal(0,0.1,total)
    accel_y = np.cos(np.linspace(0,18*np.pi,total))*accel_scale + np.random.normal(0,0.1,total)
    accel_z = np.ones(total) + np.random.normal(0,0.05,total)
    emg = np.abs(np.random.normal(emg_scale,0.25,total))
    spo2 = np.clip(np.random.normal(97,1,total),90,100)
    hr = np.clip(np.random.normal(75+10*accel_scale,8,total),55,180)
    steps = np.cumsum(np.random.rand(total)<step_prob).astype(int)
    df = pd.DataFrame({
        'timestamp':[t.strftime('%Y-%m-%d %H:%M:%S') for t in timestamps],
        'patient_id':patient_id,'accel_x':accel_x,'accel_y':accel_y,'accel_z':accel_z,
        'emg':emg,'spo2':spo2,'hr':hr,'step_count':steps
    })
    return df

def main(patients=3, hours=24, hz=1):
    os.makedirs('data', exist_ok=True)
    parts = []
    for pid in range(1, patients+1):
        parts.append(generate_patient_data(pid, hours, hz, activity='medium'))
    df = pd.concat(parts, ignore_index=True)
    fname = f'data/sample_data_large_{patients}p_{hours}h_{hz}hz.csv'
    df.to_csv(fname, index=False)
    print('Saved', fname, 'rows=', len(df))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patients', type=int, default=3)
    parser.add_argument('--hours', type=int, default=24)
    parser.add_argument('--hz', type=int, default=1)
    args = parser.parse_args()
    main(args.patients, args.hours, args.hz)
