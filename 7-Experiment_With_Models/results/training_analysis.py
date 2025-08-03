#!/usr/bin/env python3
"""
Training Analysis and Diagnostic Visualizations
Generates comprehensive plots to help diagnose and improve deep learning model training
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from pathlib import Path
import glob

# Set style for better readability
plt.style.use('default')
sns.set_palette("husl")

class TrainingAnalyzer:
    """Analyzes training data and generates diagnostic visualizations"""
    
    def __init__(self, results_dir="results", logs_dir="logs"):
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        self.script_dir = Path(__file__).parent
        
    def load_training_data(self):
        """Load training data from various sources"""
        data = {
            'results': None,
            'training_logs': [],
            'model_metrics': {}
        }
        
        # Load main results
        results_file = self.results_dir / 'results.csv'
        if results_file.exists():
            data['results'] = pd.read_csv(results_file)
        
        # Load training logs
        log_files = list(self.logs_dir.glob('*.log'))
        for log_file in log_files:
            data['training_logs'].append(self.parse_training_log(log_file))
        
        return data
    
    def parse_training_log(self, log_file):
        """Parse training log to extract epoch data"""
        epochs = []
        current_epoch = {}
        
        with open(log_file, 'r') as f:
            for line in f:
                if 'Epoch' in line and '/3' in line:
                    # New epoch
                    if current_epoch:
                        epochs.append(current_epoch)
                    current_epoch = {'epoch': len(epochs) + 1}
                    
                elif 'accuracy:' in line and 'loss:' in line:
                    # Training metrics
                    parts = line.split(' - ')
                    for part in parts:
                        if 'accuracy:' in part:
                            current_epoch['train_accuracy'] = float(part.split('accuracy: ')[1])
                        elif 'loss:' in part and 'val_loss' not in part:
                            current_epoch['train_loss'] = float(part.split('loss: ')[1])
                        elif 'val_accuracy:' in part:
                            current_epoch['val_accuracy'] = float(part.split('val_accuracy: ')[1])
                        elif 'val_loss:' in part:
                            current_epoch['val_loss'] = float(part.split('val_loss: ')[1])
        
        if current_epoch:
            epochs.append(current_epoch)
        
        return {
            'file': log_file.name,
            'epochs': epochs
        }
    
    def create_training_curves(self, training_data):
        """Create learning curves for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Curves Analysis', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(training_data['training_logs'])))
        
        for i, log_data in enumerate(training_data['training_logs']):
            if not log_data['epochs']:
                continue
                
            epochs = [e['epoch'] for e in log_data['epochs']]
            train_loss = [e.get('train_loss', 0) for e in log_data['epochs']]
            val_loss = [e.get('val_loss', 0) for e in log_data['epochs']]
            train_acc = [e.get('train_accuracy', 0) for e in log_data['epochs']]
            val_acc = [e.get('val_accuracy', 0) for e in log_data['epochs']]
            
            model_name = log_data['file'].replace('.log', '').replace('experiment_', '')
            
            # Loss curves
            axes[0, 0].plot(epochs, train_loss, 'o-', label=f'{model_name} (train)', color=colors[i], alpha=0.7)
            axes[0, 0].plot(epochs, val_loss, 's--', label=f'{model_name} (val)', color=colors[i], alpha=0.7)
            
            # Accuracy curves
            axes[0, 1].plot(epochs, train_acc, 'o-', label=f'{model_name} (train)', color=colors[i], alpha=0.7)
            axes[0, 1].plot(epochs, val_acc, 's--', label=f'{model_name} (val)', color=colors[i], alpha=0.7)
            
            # Overfitting analysis
            if len(epochs) > 1:
                train_val_gap = [abs(t - v) for t, v in zip(train_acc, val_acc)]
                axes[1, 0].plot(epochs, train_val_gap, 'o-', label=model_name, color=colors[i], alpha=0.7)
            
            # Loss ratio (train/val)
            if len(epochs) > 1:
                loss_ratio = [t/v if v > 0 else 1 for t, v in zip(train_loss, val_loss)]
                axes[1, 1].plot(epochs, loss_ratio, 'o-', label=model_name, color=colors[i], alpha=0.7)
        
        # Labels and formatting
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Training vs Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Overfitting Analysis (Train-Val Gap)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('|Train Acc - Val Acc|')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Loss Ratio (Train/Val)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train Loss / Val Loss')
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Loss')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.script_dir / 'images' / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_convergence_analysis(self, training_data):
        """Analyze convergence speed and patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
        
        convergence_data = []
        
        for log_data in training_data['training_logs']:
            if not log_data['epochs']:
                continue
                
            epochs = log_data['epochs']
            model_name = log_data['file'].replace('.log', '').replace('experiment_', '')
            
            # Find convergence points
            val_losses = [e.get('val_loss', float('inf')) for e in epochs]
            val_accs = [e.get('val_accuracy', 0) for e in epochs]
            
            # Best validation loss epoch
            best_loss_epoch = np.argmin(val_losses) + 1
            best_loss = min(val_losses)
            
            # Best validation accuracy epoch
            best_acc_epoch = np.argmax(val_accs) + 1
            best_acc = max(val_accs)
            
            # Convergence to 90% of best performance
            target_loss = best_loss * 1.1
            target_acc = best_acc * 0.9
            
            loss_convergence = None
            acc_convergence = None
            
            for i, (loss, acc) in enumerate(zip(val_losses, val_accs)):
                if loss_convergence is None and loss <= target_loss:
                    loss_convergence = i + 1
                if acc_convergence is None and acc >= target_acc:
                    acc_convergence = i + 1
            
            convergence_data.append({
                'model': model_name,
                'best_loss_epoch': best_loss_epoch,
                'best_loss': best_loss,
                'best_acc_epoch': best_acc_epoch,
                'best_acc': best_acc,
                'loss_convergence': loss_convergence,
                'acc_convergence': acc_convergence,
                'total_epochs': len(epochs)
            })
        
        if convergence_data:
            df = pd.DataFrame(convergence_data)
            
            # Convergence speed comparison
            axes[0, 0].bar(df['model'], df['loss_convergence'], alpha=0.7)
            axes[0, 0].set_title('Loss Convergence Speed (Epochs)')
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('Epochs to Converge')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].bar(df['model'], df['acc_convergence'], alpha=0.7)
            axes[0, 1].set_title('Accuracy Convergence Speed (Epochs)')
            axes[0, 1].set_xlabel('Model')
            axes[0, 1].set_ylabel('Epochs to Converge')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Best performance comparison
            axes[1, 0].bar(df['model'], df['best_loss'], alpha=0.7)
            axes[1, 0].set_title('Best Validation Loss')
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].bar(df['model'], df['best_acc'], alpha=0.7)
            axes[1, 1].set_title('Best Validation Accuracy')
            axes[1, 1].set_xlabel('Model')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.script_dir / 'images' / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_training_insights(self, training_data):
        """Generate insights and recommendations"""
        insights = []
        
        for log_data in training_data['training_logs']:
            if not log_data['epochs']:
                continue
                
            model_name = log_data['file'].replace('.log', '').replace('experiment_', '')
            epochs = log_data['epochs']
            
            # Analyze training patterns
            val_losses = [e.get('val_loss', float('inf')) for e in epochs]
            val_accs = [e.get('val_accuracy', 0) for e in epochs]
            train_losses = [e.get('train_loss', float('inf')) for e in epochs]
            train_accs = [e.get('train_accuracy', 0) for e in epochs]
            
            # Detect overfitting
            if len(epochs) > 2:
                final_train_val_gap = abs(train_accs[-1] - val_accs[-1])
                if final_train_val_gap > 0.1:
                    insights.append(f"⚠️  {model_name}: Potential overfitting (gap: {final_train_val_gap:.3f})")
            
            # Detect underfitting
            if val_accs[-1] < 0.7:
                insights.append(f"🔧 {model_name}: Low validation accuracy ({val_accs[-1]:.3f}) - consider more training or model changes")
            
            # Detect instability
            if len(val_losses) > 2:
                loss_variance = np.var(val_losses[-3:])
                if loss_variance > 0.1:
                    insights.append(f"📈 {model_name}: High loss variance ({loss_variance:.3f}) - consider reducing learning rate")
            
            # Convergence analysis
            best_epoch = np.argmax(val_accs) + 1
            if best_epoch < len(epochs):
                insights.append(f"⏱️  {model_name}: Best performance at epoch {best_epoch}/{len(epochs)} - consider early stopping")
        
        # Save insights
        with open(self.script_dir / 'images' / 'training_insights.txt', 'w') as f:
            f.write("Training Insights and Recommendations\n")
            f.write("=" * 40 + "\n\n")
            for insight in insights:
                f.write(f"{insight}\n")
        
        return insights
    
    def generate_all_analyses(self):
        """Generate all training analyses"""
        print("Loading training data...")
        training_data = self.load_training_data()
        
        if not training_data['training_logs']:
            print("No training logs found. Run the experiment first.")
            return
        
        print("Creating training curves...")
        self.create_training_curves(training_data)
        
        print("Creating convergence analysis...")
        self.create_convergence_analysis(training_data)
        
        print("Generating insights...")
        insights = self.create_training_insights(training_data)
        
        print(f"\nGenerated {len(insights)} insights:")
        for insight in insights:
            print(f"  {insight}")
        
        print(f"\nTraining analysis saved to {self.script_dir / 'images'}:")
        print("  - training_analysis.png")
        print("  - training_insights.txt")

if __name__ == "__main__":
    analyzer = TrainingAnalyzer()
    analyzer.generate_all_analyses() 