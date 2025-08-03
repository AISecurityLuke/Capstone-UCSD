#!/usr/bin/env python3
"""
Visualization Module for ML Experiment Results
Generates comprehensive visualizations from experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from matplotlib import cm
from pandas.plotting import parallel_coordinates

# Set style for better readability
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load experiment results"""
    results_path = Path(__file__).parent / 'results.csv'
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} model results")
    return df

def create_model_comparison(df):
    """Create comprehensive model comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold')
    
    # Prepare data
    metrics = ['precision_macro', 'recall_macro', 'f1_macro']
    colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
    
    # 1. Overall performance comparison
    x = np.arange(len(df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = df[metric].values
        axes[0, 0].bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), 
                      alpha=0.8, color=colors)
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance Metrics by Model', fontweight='bold')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(df['model'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # 2. F1 Score ranking
    df_sorted = df.sort_values('f1_macro', ascending=True)
    bars = axes[0, 1].barh(range(len(df_sorted)), df_sorted['f1_macro'], color=colors)
    axes[0, 1].set_yticks(range(len(df_sorted)))
    axes[0, 1].set_yticklabels(df_sorted['model'])
    axes[0, 1].set_xlabel('F1 Score')
    axes[0, 1].set_title('F1 Score Ranking', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0, 1].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    # 3. Precision vs Recall scatter
    scatter = axes[1, 0].scatter(df['precision_macro'], df['recall_macro'], 
                                s=100, c=df['f1_macro'], cmap='viridis', alpha=0.7)
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Precision vs Recall (colored by F1)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add model labels
    for i, model in enumerate(df['model']):
        axes[1, 0].annotate(model, (df['precision_macro'].iloc[i], df['recall_macro'].iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('F1 Score')
    
    # 4. Model type analysis
    model_types = df['model'].str.extract(r'(\w+)')[0].value_counts()
    axes[1, 1].pie(model_types.values, labels=model_types.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors[:len(model_types)])
    axes[1, 1].set_title('Model Type Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    # Save to images subdirectory
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model comparison saved to {images_dir / 'model_comparison.png'}")

def create_radar_chart(df):
    """Create radar chart for top 5 models"""
    # Get top 5 models by F1 score
    top_models = df.nlargest(5, 'f1_macro')
    
    # Prepare data for radar chart
    categories = ['Precision', 'Recall', 'F1 Score']
    values = top_models[['precision_macro', 'recall_macro', 'f1_macro']].values
    
    # Number of variables
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, (model, values_row) in enumerate(zip(top_models['model'], values)):
        values_row = np.concatenate((values_row, [values_row[0]]))  # Complete the circle
        ax.plot(angles, values_row, 'o-', linewidth=3, label=model, color=colors[i], markersize=8)
        ax.fill(angles, values_row, alpha=0.1, color=colors[i])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Top 5 Models Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
    
    # Save to images subdirectory
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved to {images_dir / 'radar_chart.png'}")

def create_parallel_coordinates(df):
    """Create parallel coordinates plot"""
    # Prepare data
    plot_data = df[['model', 'precision_macro', 'recall_macro', 'f1_macro']].copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize the data for better visualization
    metrics = ['precision_macro', 'recall_macro', 'f1_macro']
    for metric in metrics:
        plot_data[metric] = (plot_data[metric] - plot_data[metric].min()) / (plot_data[metric].max() - plot_data[metric].min())
    
    # Create parallel coordinates plot
    pd.plotting.parallel_coordinates(plot_data, 'model', cols=metrics, ax=ax, 
                                   color=plt.cm.Set3(np.linspace(0, 1, len(df))))
    
    ax.set_title('Model Performance Comparison - Parallel Coordinates', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Save to images subdirectory
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True)
    plt.savefig(images_dir / 'parallel_coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Parallel coordinates plot saved to {images_dir / 'parallel_coordinates.png'}")

def generate_visualizations():
    """Generate all visualizations"""
    print("Loading results...")
    df = load_results()
    
    if df is None or len(df) == 0:
        print("No results to visualize")
        return
    
    print("Creating visualizations...")
    
    # Create images directory
    images_dir = Path(__file__).parent / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    create_model_comparison(df)
    create_radar_chart(df)
    create_parallel_coordinates(df)
    
    print(f"\nAll visualizations saved to {images_dir}/")
    print("Files created:")
    print("  - model_comparison.png")
    print("  - radar_chart.png")
    print("  - parallel_coordinates.png")

if __name__ == "__main__":
    generate_visualizations()