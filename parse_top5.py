import pandas as pd

try:
    df = pd.read_csv('runs.csv')
    
    # filter for v26
    v26_runs = df[df['Name'].astype(str).str.contains('v26', case=False, na=False)].copy()
    
    # Sort by mAP50B as the primary metric, but we will print all so I can select the best 5 manually
    if 'metrics/mAP50B' in df.columns:
        v26_runs = v26_runs.sort_values(by='metrics/mAP50B', ascending=False)
        
    cols = ['Name', 'metrics/mAP50-95B', 'metrics/mAP50B', 'metrics/precisionB', 'metrics/recallB']
    
    print(v26_runs[cols].to_string(index=False))
except Exception as e:
    print(f"Error: {e}")
