import pandas as pd

try:
    df = pd.read_csv('runs.csv')
    
    # filter for v26
    v26_runs = df[df['Name'].astype(str).str.contains('v26', case=False, na=False)].copy()
    
    # Sort by start_time if possible
    if 'Start Time' in df.columns:
        v26_runs = v26_runs.sort_values(by='Start Time', ascending=False)
    elif 'start_time' in df.columns:
        v26_runs = v26_runs.sort_values(by='start_time', ascending=False)
        
    cols = ['Name', 'metrics/mAP50-95B', 'metrics/mAP50B', 'metrics/precisionB', 'metrics/recallB']
    
    # Solo le v26
    print(v26_runs[cols].head(15).to_string(index=False))
except Exception as e:
    print(f"Error: {e}")
