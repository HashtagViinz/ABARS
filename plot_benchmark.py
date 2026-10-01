import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for nicer graphs
sns.set_theme(style="whitegrid")

# Load data
df_latency = pd.read_csv('benchmark_metrics.csv')
df_metrics = pd.read_csv('metric_classes.csv')

# Clean model names in latency df to match metrics df (remove _ncnn_model)
df_latency['Model'] = df_latency['Model'].str.replace('_ncnn_model', '')

# Merge the dataframes on the Model column
# We take Precision from df_metrics, and FPS/Latency from df_latency
df = pd.merge(df_latency, df_metrics, on='Model', suffixes=('_old', ''))

# Drop the old precision column from latency df to avoid confusion
df = df.drop(columns=['Precision_old'])


# Clean model names for plotting
def clean_name(name):
    base = name.split('_')[0].upper() # V26N, V26M, V26S
    if 'dehaze' in name:
        return f"{base} (Dehaze+Bilat+Sharp)"
    elif 'bilateral_d11_sharp' in name:
        return f"{base} (Bilat D11+Sharp)"
    elif 'bilateral_sharp' in name:
        return f"{base} (Bilat+Sharp)"
    elif 'bilateral_d11' in name:
        return f"{base} (Bilat D11)"
    else:
        # Trova se ha un numero dopo tiled per indicare una versione (es. tiled-2)
        import re
        m = re.search(r'tiled-(\d+)', name)
        if m:
            return f"{base} (Natural v{m.group(1)})"
        return f"{base} (Natural)"

df['Short_Name'] = df['Model'].apply(clean_name)

# ---------------------------------------------------------
# Plot 1: Precision vs FPS (Scatter Plot)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

# Create scatter plot
ax = sns.scatterplot(
    data=df, 
    x='FPS_SAHI', 
    y='Precision', 
    hue='Short_Name', 
    style='Short_Name',
    s=250, 
    palette='tab10',
    markers=['o', 's', 'D', '^', 'v', '<', '>']
)

plt.title('Precision vs FPS', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Speed (FPS)', fontsize=14)
plt.ylabel('Precision', fontsize=14)

# Add value labels next to points
for i in range(df.shape[0]):
    plt.text(
        df['FPS_SAHI'][i] + 0.02, 
        df['Precision'][i] + 0.002, 
        df['Short_Name'][i], 
        fontsize=9,
        alpha=0.8
    )

# Adjust legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig('precision_vs_fps.png', dpi=300)
print("Salvato: precision_vs_fps.png")
plt.close()

# ---------------------------------------------------------
# Plot 2: Latency Breakdown Stacked Bar Chart
# ---------------------------------------------------------
# Sort by total latency for a better looking graph
df_sorted = df.sort_values('Total_Latency_ms', ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 8))
bar_width = 0.6
r = np.arange(len(df_sorted))

cv_times = df_sorted['CV_Preprocess_ms_SAHI']
ncnn_times = df_sorted['NCNN_Inference_ms_SAHI']
total_times = df_sorted['Total_Latency_ms']
# Overhead calculations
overhead_times = total_times - cv_times - ncnn_times
# Ensure no negative overhead due to rounding issues
overhead_times = np.maximum(overhead_times, 0)

# Create the stacked bars
plt.bar(r, cv_times, color='#3498db', edgecolor='white', width=bar_width, label='OpenCV (filters)')
plt.bar(r, ncnn_times, bottom=cv_times, color='#e74c3c', edgecolor='white', width=bar_width, label='NCNN (Inference x4)')
plt.bar(r, overhead_times, bottom=cv_times + ncnn_times, color='#2ecc71', edgecolor='white', width=bar_width, label='SAHI')

# Customization
plt.xticks(r, df_sorted['Short_Name'], rotation=45, ha='right', fontsize=10)
plt.ylabel('Processing time per single 4K photo (Milliseconds)', fontsize=12)
plt.title('SAHI Processing Time Decomposition (Raspberry Pi)', fontsize=14, fontweight='bold', pad=15)

# Add total time labels on top of bars
for i in range(len(r)):
    plt.text(
        x=r[i], 
        y=total_times[i] + (total_times.max() * 0.01), 
        s=f"{total_times[i]:.0f} ms", 
        ha='center', 
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('latency_breakdown.png', dpi=300)
print("Salvato: latency_breakdown.png")
plt.close()

