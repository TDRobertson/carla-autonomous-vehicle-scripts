"""
Results Visualization for GPS Spoofing Detection Models

Generate plots to visualize model performance, ROC curves, confusion matrices, etc.
"""

import numpy as np
from typing import Dict, List, Optional
import os

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib seaborn")


class ResultsVisualizer:
    """
    Create visualizations for model evaluation results.
    """
    
    def __init__(self, results: Dict, output_dir: str = "results"):
        """
        Initialize visualizer.
        
        Args:
            results: Dictionary of model results from MetricsCalculator
            output_dir: Directory to save plots
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualizations")
            
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
    def plot_model_comparison(self, filename: str = "model_comparison.png"):
        """
        Plot comparison of all models across key metrics.
        
        Args:
            filename: Output filename
        """
        models = list(self.results.keys())
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'auc']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [self.results[m].get(metric, 0) for m in models]
            
            ax = axes[idx]
            bars = ax.bar(range(len(models)), values, alpha=0.7)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Color bars by performance
            for i, (bar, val) in enumerate(zip(bars, values)):
                if val > 0.8:
                    bar.set_color('green')
                elif val > 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
                    
                # Add value labels
                ax.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
        
    def plot_confusion_matrices(
        self,
        models: Dict,
        X_val: np.ndarray,
        y_val: np.ndarray,
        filename: str = "confusion_matrices.png"
    ):
        """
        Plot confusion matrices for all models.
        
        Args:
            models: Dictionary of trained models
            X_val: Validation features
            y_val: Validation labels
            filename: Output filename
        """
        from sklearn.metrics import confusion_matrix
        
        n_models = len(models)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X_val)
            # Convert predictions
            y_pred_binary = np.where(y_pred == -1, 1, 0)
            
            cm = confusion_matrix(y_val, y_pred_binary)
            
            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'{name}\nConfusion Matrix', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Clean', 'Attack'])
            ax.set_yticklabels(['Clean', 'Attack'])
            
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle('Confusion Matrices for All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
        
    def plot_roc_curves(self, filename: str = "roc_curves.png"):
        """
        Plot ROC curves for all models.
        
        Args:
            filename: Output filename
        """
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (name, metrics), color in zip(self.results.items(), colors):
            if 'fpr' in metrics and 'tpr' in metrics:
                fpr = metrics['fpr']
                tpr = metrics['tpr']
                auc = metrics.get('auc', 0)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', 
                        color=color, linewidth=2)
                
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - GPS Spoofing Detection', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
        
    def plot_precision_recall_curves(self, filename: str = "precision_recall_curves.png"):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            filename: Output filename
        """
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (name, metrics), color in zip(self.results.items(), colors):
            if 'pr_precision' in metrics and 'pr_recall' in metrics:
                precision = metrics['pr_precision']
                recall = metrics['pr_recall']
                
                plt.plot(recall, precision, label=name, color=color, linewidth=2)
                
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - GPS Spoofing Detection', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(alpha=0.3)
        plt.ylim([0, 1.05])
        plt.xlim([0, 1.05])
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
        
    def plot_feature_importance(
        self,
        model,
        feature_names: List[str],
        model_name: str,
        filename: Optional[str] = None
    ):
        """
        Plot feature importance (for tree-based models).
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            model_name: Name of the model
            filename: Output filename (auto-generated if None)
        """
        if not hasattr(model.model, 'feature_importances_'):
            print(f"Model {model_name} does not support feature importance")
            return
            
        importances = model.model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices], alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        if filename is None:
            filename = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    print("Results Visualizer for GPS Spoofing Detection")
    print("="*60)
    print("\nGenerates:")
    print("  - Model comparison charts")
    print("  - Confusion matrices")
    print("  - ROC curves")
    print("  - Precision-Recall curves")
    print("  - Feature importance (for tree models)")
    print("\nUse via the training script: train_models.py")

