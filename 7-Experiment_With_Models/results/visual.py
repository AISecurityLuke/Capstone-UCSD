import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib import cm
from pandas.plotting import parallel_coordinates

# Set style for better readability
plt.style.use('default')
sns.set_palette("husl")

# Always find results.csv relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
results_csv = os.path.join(script_dir, 'results.csv')
df = pd.read_csv(results_csv)

# Ensure model names are strings
df['model'] = df['model'].astype(str)

# Only keep top 5 models by f1_macro for all plots
df_top5 = df.sort_values('f1_macro', ascending=False).head(5)

# Calculate dynamic y-axis limits based on actual data
min_score = df_top5[['precision_macro', 'recall_macro', 'f1_macro']].min().min()
max_score = df_top5[['precision_macro', 'recall_macro', 'f1_macro']].max().max()
y_padding = (max_score - min_score) * 0.1
y_min = max(0, min_score - y_padding)
y_max = min(1, max_score + y_padding)

# --------------------------
# Grouped Bar Plot
# --------------------------
metrics = ['precision_macro', 'recall_macro', 'f1_macro']
df_melted = df_top5.melt(id_vars='model', value_vars=metrics, var_name='metric', value_name='score')

plt.figure(figsize=(14, 8))
ax = sns.barplot(x='model', y='score', hue='metric', data=df_melted, palette='Set2')
plt.title('Model Comparison - Precision, Recall, F1 (Top 5 by F1)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.ylim(y_min, y_max)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=10, padding=3)

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.legend(title='Metric', title_fontsize=12, fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'grouped_bar_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# --------------------------
# Enhanced Radar Chart (Top 5 Models)
# --------------------------
def make_radar_chart(data, metrics, labels, filename):
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.get_cmap('Set2', len(data))

    for idx, (i, row) in enumerate(data.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['model'], color=colors(idx), linewidth=4, marker='o', markersize=10)
        ax.fill(angles, values, color=colors(idx), alpha=0.2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12, fontweight='bold')
    ax.set_ylim(y_min, y_max)
    ax.set_title('Radar Chart of Top 5 Models by F1', fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

make_radar_chart(df_top5, metrics, ['Precision', 'Recall', 'F1'], 'radar_chart.png')

# --------------------------
# Parallel Coordinates Plot (Top 5 Models)
# --------------------------
plt.figure(figsize=(14, 8))
parallel_coordinates(df_top5[['model', 'precision_macro', 'recall_macro', 'f1_macro']], 'model', colormap=plt.get_cmap("Set2"))
plt.title('Parallel Coordinates Plot of Model Metrics (Top 5 by F1)', fontsize=16, fontweight='bold', pad=20)
plt.ylim(y_min, y_max)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'parallel_coordinates.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualizations generated with y-axis range: {y_min:.3f} to {y_max:.3f}")
print(f"Data range: {min_score:.3f} to {max_score:.3f}")