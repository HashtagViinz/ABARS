import pandas as pd

try:
    df = pd.read_csv('runs.csv')
    
    # filter for v26
    v26_runs = df[df['Name'].astype(str).str.contains('v26', case=False, na=False)].copy()
    
    # drop where mAP is NaN
    v26_runs = v26_runs.dropna(subset=['metrics/mAP50B'])
    
    # calculate F1 score
    v26_runs['F1_Score'] = 2 * (v26_runs['metrics/precisionB'] * v26_runs['metrics/recallB']) / (v26_runs['metrics/precisionB'] + v26_runs['metrics/recallB'])
    
    # Sort by F1 Score
    v26_runs = v26_runs.sort_values(by='F1_Score', ascending=False)
    
    cols = ['Name', 'F1_Score', 'metrics/mAP50B', 'metrics/precisionB', 'metrics/recallB']
    
    print("=== TOP MODELS BY F1-SCORE ===")
    print(v26_runs[cols].head(15).to_string(index=False))
except Exception as e:
    print(f"Error: {e}")
