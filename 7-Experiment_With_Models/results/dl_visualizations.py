#!/usr/bin/env python3
"""
Deep Learning Specific Visualizations
Specialized plots and analysis for deep learning model training and architecture comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from pathlib import Path
import glob
import re

# Set style for better readability
plt.style.use('default')
sns.set_palette("husl")

class DLVisualizer:
    """Specialized visualizer for deep learning model analysis"""
    
    def __init__(self, results_dir="results", logs_dir="logs", models_dir="models"):
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        self.models_dir = Path(models_dir)
        self.script_dir = Path(__file__).parent
        
    def load_dl_data(self):
        """Load deep learning specific data"""
        data = {
            'results': None,
            'training_logs': [],
            'model_architectures': {},
            'gradient_data': []
        }
        
        # Load main results
        results_file = self.results_dir / 'results.csv'
        if results_file.exists():
            df = pd.read_csv(results_file)
            # Filter for deep learning models
            dl_models = df[df['model'].str.contains('|'.join(['cnn', 'lstm', 'bilstm', 'transformer', 'hybrid', 'bert', 'distilbert', 'roberta', 'albert', 'distilroberta']), case=False)]
            data['results'] = dl_models
        
        # Load training logs
        log_files = list(self.logs_dir.glob('*.log'))
        for log_file in log_files:
            data['training_logs'].append(self.parse_dl_training_log(log_file))
        
        return data
    
    def parse_dl_training_log(self, log_file):
        """Parse DL training log with detailed epoch information"""
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
                        elif 'learning_rate:' in part:
                            current_epoch['learning_rate'] = float(part.split('learning_rate: ')[1])
        
        if current_epoch:
            epochs.append(current_epoch)
        
        return {
            'file': log_file.name,
            'epochs': epochs
        }
    
    def create_dl_training_curves(self, dl_data):
        """Create detailed DL training curves"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Deep Learning Training Analysis', fontsize=18, fontweight='bold')
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(dl_data['training_logs'])))
        
        for i, log_data in enumerate(dl_data['training_logs']):
            if not log_data['epochs']:
                continue
                
            epochs = [e['epoch'] for e in log_data['epochs']]
            train_loss = [e.get('train_loss', 0) for e in log_data['epochs']]
            val_loss = [e.get('val_loss', 0) for e in log_data['epochs']]
            train_acc = [e.get('train_accuracy', 0) for e in log_data['epochs']]
            val_acc = [e.get('val_accuracy', 0) for e in log_data['epochs']]
            lr = [e.get('learning_rate', 0.001) for e in log_data['epochs']]
            
            model_name = log_data['file'].replace('.log', '').replace('experiment_', '')
            
            # Loss curves with gradient analysis
            axes[0, 0].plot(epochs, train_loss, 'o-', label=f'{model_name} (train)', color=colors[i], alpha=0.7, linewidth=2)
            axes[0, 0].plot(epochs, val_loss, 's--', label=f'{model_name} (val)', color=colors[i], alpha=0.7, linewidth=2)
            
            # Accuracy curves
            axes[0, 1].plot(epochs, train_acc, 'o-', label=f'{model_name} (train)', color=colors[i], alpha=0.7, linewidth=2)
            axes[0, 1].plot(epochs, val_acc, 's--', label=f'{model_name} (val)', color=colors[i], alpha=0.7, linewidth=2)
            
            # Learning rate progression
            axes[1, 0].plot(epochs, lr, 'o-', label=model_name, color=colors[i], alpha=0.7, linewidth=2)
            
            # Gradient analysis (loss change rate)
            if len(train_loss) > 1:
                loss_gradients = np.diff(train_loss)
                axes[1, 1].plot(epochs[1:], loss_gradients, 'o-', label=model_name, color=colors[i], alpha=0.7, linewidth=2)
            
            # Overfitting analysis
            if len(epochs) > 1:
                train_val_gap = [abs(t - v) for t, v in zip(train_acc, val_acc)]
                axes[2, 0].plot(epochs, train_val_gap, 'o-', label=model_name, color=colors[i], alpha=0.7, linewidth=2)
            
            # Loss stability analysis
            if len(train_loss) > 2:
                loss_rolling_std = [np.std(train_loss[max(0, i-2):i+1]) for i in range(len(train_loss))]
                axes[2, 1].plot(epochs, loss_rolling_std, 'o-', label=model_name, color=colors[i], alpha=0.7, linewidth=2)
        
        # Labels and formatting
        axes[0, 0].set_title('Training vs Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Training vs Validation Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Learning Rate Progression', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        axes[1, 1].set_title('Loss Gradient Analysis', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Change Rate')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        axes[2, 0].set_title('Overfitting Analysis (Train-Val Gap)', fontweight='bold')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('|Train Acc - Val Acc|')
        axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
        
        axes[2, 1].set_title('Loss Stability Analysis', fontweight='bold')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Rolling Loss Std Dev')
        axes[2, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.script_dir / 'images' / 'dl_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_architecture_comparison(self, dl_data):
        """Compare different DL architectures"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deep Learning Architecture Comparison', fontsize=18, fontweight='bold')
        
        architecture_data = []
        
        for log_data in dl_data['training_logs']:
            if not log_data['epochs']:
                continue
                
            epochs = log_data['epochs']
            model_name = log_data['file'].replace('.log', '').replace('experiment_', '')
            
            # Extract architecture type
            arch_type = 'unknown'
            for arch in ['cnn', 'lstm', 'bilstm', 'transformer', 'hybrid', 'bert', 'distilbert', 'roberta', 'albert', 'distilroberta']:
                if arch in model_name.lower():
                    arch_type = arch
                    break
            
            # Calculate metrics
            val_losses = [e.get('val_loss', float('inf')) for e in epochs]
            val_accs = [e.get('val_accuracy', 0) for e in epochs]
            train_losses = [e.get('train_loss', float('inf')) for e in epochs]
            
            best_loss = min(val_losses)
            best_acc = max(val_accs)
            final_loss = val_losses[-1]
            final_acc = val_accs[-1]
            
            # Convergence analysis
            target_acc = best_acc * 0.9
            convergence_epoch = None
            for i, acc in enumerate(val_accs):
                if acc >= target_acc:
                    convergence_epoch = i + 1
                    break
            
            # Stability analysis
            loss_variance = np.var(val_losses) if len(val_losses) > 1 else 0
            train_val_gap = abs(train_losses[-1] - val_losses[-1]) if len(train_losses) > 0 else 0
            
            architecture_data.append({
                'model': model_name,
                'architecture': arch_type,
                'best_loss': best_loss,
                'best_acc': best_acc,
                'final_loss': final_loss,
                'final_acc': final_acc,
                'convergence_epoch': convergence_epoch,
                'total_epochs': len(epochs),
                'loss_variance': loss_variance,
                'train_val_gap': train_val_gap
            })
        
        if architecture_data:
            df = pd.DataFrame(architecture_data)
            
            # Best performance by architecture
            arch_performance = df.groupby('architecture').agg({
                'best_acc': 'mean',
                'best_loss': 'mean',
                'convergence_epoch': 'mean'
            }).reset_index()
            
            # Performance comparison
            axes[0, 0].bar(arch_performance['architecture'], arch_performance['best_acc'], alpha=0.7)
            axes[0, 0].set_title('Best Validation Accuracy by Architecture', fontweight='bold')
            axes[0, 0].set_xlabel('Architecture')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Convergence speed
            axes[0, 1].bar(arch_performance['architecture'], arch_performance['convergence_epoch'], alpha=0.7)
            axes[0, 1].set_title('Convergence Speed by Architecture', fontweight='bold')
            axes[0, 1].set_xlabel('Architecture')
            axes[0, 1].set_ylabel('Epochs to 90% of Best')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Stability comparison
            axes[1, 0].scatter(df['loss_variance'], df['best_acc'], alpha=0.7, s=100)
            axes[1, 0].set_title('Stability vs Performance', fontweight='bold')
            axes[1, 0].set_xlabel('Loss Variance')
            axes[1, 0].set_ylabel('Best Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add model labels
            for _, row in df.iterrows():
                axes[1, 0].annotate(row['model'], (row['loss_variance'], row['best_acc']), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Overfitting analysis
            axes[1, 1].scatter(df['train_val_gap'], df['best_acc'], alpha=0.7, s=100)
            axes[1, 1].set_title('Overfitting vs Performance', fontweight='bold')
            axes[1, 1].set_xlabel('Train-Val Loss Gap')
            axes[1, 1].set_ylabel('Best Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
            
            # Add model labels
            for _, row in df.iterrows():
                axes[1, 1].annotate(row['model'], (row['train_val_gap'], row['best_acc']), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.script_dir / 'images' / 'dl_architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_dl_insights(self, dl_data):
        """Generate DL-specific insights and recommendations"""
        insights = []
        
        for log_data in dl_data['training_logs']:
            if not log_data['epochs']:
                continue
                
            model_name = log_data['file'].replace('.log', '').replace('experiment_', '')
            epochs = log_data['epochs']
            
            # Extract architecture
            arch_type = 'unknown'
            for arch in ['cnn', 'lstm', 'bilstm', 'transformer', 'hybrid', 'bert', 'distilbert', 'roberta', 'albert', 'distilroberta']:
                if arch in model_name.lower():
                    arch_type = arch
                    break
            
            # Analyze training patterns
            val_losses = [e.get('val_loss', float('inf')) for e in epochs]
            val_accs = [e.get('val_accuracy', 0) for e in epochs]
            train_losses = [e.get('train_loss', float('inf')) for e in epochs]
            train_accs = [e.get('train_accuracy', 0) for e in epochs]
            lrs = [e.get('learning_rate', 0.001) for e in epochs]
            
            # Architecture-specific insights
            if arch_type == 'cnn':
                if val_accs[-1] < 0.8:
                    insights.append(f"🔧 {model_name} (CNN): Consider increasing filters or adding more layers")
                if len(epochs) < 5:
                    insights.append(f"⏱️ {model_name} (CNN): CNNs often benefit from more training epochs")
                    
            elif arch_type == 'lstm':
                if val_accs[-1] < 0.75:
                    insights.append(f"🔧 {model_name} (LSTM): Try bidirectional LSTM or increase units")
                if lrs[-1] > 0.001:
                    insights.append(f"📉 {model_name} (LSTM): Consider lower learning rate for LSTM stability")
                    
            elif arch_type == 'transformer':
                if val_accs[-1] < 0.85:
                    insights.append(f"🔧 {model_name} (Transformer): Try increasing attention heads or layers")
                if len(epochs) < 10:
                    insights.append(f"⏱️ {model_name} (Transformer): Transformers often need more epochs to converge")
            
            elif arch_type == 'roberta':
                if val_accs[-1] < 0.88:
                    insights.append(f"🔧 {model_name} (RoBERTa): Consider larger model or better fine-tuning")
                if lrs[-1] > 1e-5:
                    insights.append(f"📉 {model_name} (RoBERTa): Use lower learning rate for RoBERTa fine-tuning")
                    
            elif arch_type == 'albert':
                if val_accs[-1] < 0.82:
                    insights.append(f"🔧 {model_name} (ALBERT): Try larger ALBERT variant or longer training")
                if len(epochs) < 15:
                    insights.append(f"⏱️ {model_name} (ALBERT): ALBERT benefits from longer training")
                    
            elif arch_type == 'distilroberta':
                if val_accs[-1] < 0.86:
                    insights.append(f"🔧 {model_name} (DistilRoBERTa): Consider full RoBERTa for better performance")
                if len(epochs) < 8:
                    insights.append(f"⏱️ {model_name} (DistilRoBERTa): DistilRoBERTa needs adequate training time")
            
            # General DL insights
            if len(epochs) > 2:
                final_train_val_gap = abs(train_accs[-1] - val_accs[-1])
                if final_train_val_gap > 0.15:
                    insights.append(f"⚠️ {model_name}: Severe overfitting (gap: {final_train_val_gap:.3f}) - add dropout/regularization")
                elif final_train_val_gap > 0.08:
                    insights.append(f"⚠️ {model_name}: Moderate overfitting (gap: {final_train_val_gap:.3f}) - consider early stopping")
            
            if val_accs[-1] < 0.7:
                insights.append(f"🔧 {model_name}: Low validation accuracy ({val_accs[-1]:.3f}) - consider architecture changes or more data")
            
            if len(val_losses) > 2:
                loss_variance = np.var(val_losses[-3:])
                if loss_variance > 0.05:
                    insights.append(f"📈 {model_name}: High loss variance ({loss_variance:.3f}) - reduce learning rate")
            
            # Convergence analysis
            best_epoch = np.argmax(val_accs) + 1
            if best_epoch < len(epochs) - 1:
                insights.append(f"⏱️ {model_name}: Best at epoch {best_epoch}/{len(epochs)} - implement early stopping")
            
            # Learning rate analysis
            if len(lrs) > 1:
                lr_change = abs(lrs[-1] - lrs[0]) / lrs[0]
                if lr_change < 0.1:
                    insights.append(f"📊 {model_name}: Learning rate barely changed - consider more aggressive scheduling")
        
        # Save insights
        with open(self.script_dir / 'images' / 'dl_insights.txt', 'w') as f:
            f.write("Deep Learning Training Insights and Recommendations\n")
            f.write("=" * 50 + "\n\n")
            f.write("Architecture-Specific Recommendations:\n")
            f.write("-" * 35 + "\n")
            for insight in insights:
                f.write(f"{insight}\n")
            
            f.write("\nGeneral Deep Learning Best Practices:\n")
            f.write("-" * 35 + "\n")
            f.write("• Use learning rate scheduling for better convergence\n")
            f.write("• Implement early stopping to prevent overfitting\n")
            f.write("• Monitor gradient norms for stability\n")
            f.write("• Consider model ensemble for better performance\n")
            f.write("• Use data augmentation for small datasets\n")
        
        return insights
    
    def generate_all_dl_visualizations(self):
        """Generate all deep learning visualizations"""
        print("Loading deep learning data...")
        dl_data = self.load_dl_data()
        
        if not dl_data['training_logs']:
            print("No deep learning training logs found. Run the experiment with DL models first.")
            return
        
        print("Creating DL training curves...")
        self.create_dl_training_curves(dl_data)
        
        print("Creating architecture comparison...")
        self.create_architecture_comparison(dl_data)
        
        print("Generating DL insights...")
        insights = self.create_dl_insights(dl_data)
        
        print(f"\nGenerated {len(insights)} deep learning insights:")
        for insight in insights:
            print(f"  {insight}")
        
        print(f"\nDeep Learning visualizations saved to {self.script_dir / 'images'}:")
        print("  - dl_training_curves.png")
        print("  - dl_architecture_comparison.png")
        print("  - dl_insights.txt")

if __name__ == "__main__":
    visualizer = DLVisualizer()
    visualizer.generate_all_dl_visualizations() 