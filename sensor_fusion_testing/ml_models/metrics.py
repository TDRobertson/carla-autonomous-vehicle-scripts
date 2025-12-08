"""
Evaluation Metrics for GPS Spoofing Detection Models

Calculate performance metrics for one-class anomaly detection models.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from typing import Dict, Tuple, Optional
import pandas as pd


class MetricsCalculator:
    """
    Calculate and store evaluation metrics for anomaly detection models.
    
    For one-class classifiers:
    - Normal (clean) = label 0, predicted as 1
    - Anomaly (attack) = label 1, predicted as -1
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.results = {}
        
    @staticmethod
    def convert_predictions(y_pred: np.ndarray) -> np.ndarray:
        """
        Convert one-class predictions to binary labels.
        
        Args:
            y_pred: Predictions from one-class model (-1 for anomaly, 1 for normal)
            
        Returns:
            Binary predictions (0 for normal, 1 for anomaly)
        """
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        return np.where(y_pred == -1, 1, 0)
        
    def calculate_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels (0=clean, 1=attack)
            y_pred: Predicted labels from one-class model (-1=anomaly, 1=normal)
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary with metrics
        """
        # Convert predictions to binary
        y_pred_binary = self.convert_predictions(y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Calculate rates
        detection_rate = recall  # Same as recall for attack detection
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        metrics = {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'accuracy': accuracy,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_true),
            'attack_samples': int(np.sum(y_true)),
            'clean_samples': int(len(y_true) - np.sum(y_true))
        }
        
        self.results[model_name] = metrics
        return metrics
        
    def calculate_roc_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Calculate ROC curve and AUC score.
        
        Args:
            y_true: True labels (0=clean, 1=attack)
            y_scores: Anomaly scores from model (lower = more anomalous)
            model_name: Name of the model
            
        Returns:
            Dictionary with ROC metrics
        """
        # For ROC curve, we need scores where higher = more anomalous
        # Most models return lower scores for anomalies, so we negate
        y_scores_inverted = -y_scores
        
        try:
            auc = roc_auc_score(y_true, y_scores_inverted)
            fpr, tpr, thresholds = roc_curve(y_true, y_scores_inverted)
            
            roc_metrics = {
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            }
            
            if model_name in self.results:
                self.results[model_name].update(roc_metrics)
            else:
                self.results[model_name] = roc_metrics
                
            return roc_metrics
        except Exception as e:
            print(f"Warning: Could not calculate ROC metrics for {model_name}: {e}")
            return {'auc': 0.0, 'fpr': np.array([]), 'tpr': np.array([]), 'thresholds': np.array([])}
            
    def calculate_pr_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Calculate Precision-Recall curve.
        
        Args:
            y_true: True labels (0=clean, 1=attack)
            y_scores: Anomaly scores from model
            model_name: Name of the model
            
        Returns:
            Dictionary with PR curve metrics
        """
        # Invert scores for PR curve
        y_scores_inverted = -y_scores
        
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores_inverted)
            
            pr_metrics = {
                'pr_precision': precision,
                'pr_recall': recall,
                'pr_thresholds': thresholds
            }
            
            if model_name in self.results:
                self.results[model_name].update(pr_metrics)
            else:
                self.results[model_name] = pr_metrics
                
            return pr_metrics
        except Exception as e:
            print(f"Warning: Could not calculate PR metrics for {model_name}: {e}")
            return {'pr_precision': np.array([]), 'pr_recall': np.array([]), 'pr_thresholds': np.array([])}
            
    def calculate_time_to_detection(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        sampling_rate: float = 10.0
    ) -> Dict:
        """
        Calculate time to detection after attack starts.
        
        Args:
            y_true: True labels (0=clean, 1=attack)
            y_pred: Predictions from model
            timestamps: Optional array of timestamps
            sampling_rate: Sampling rate in Hz (default: 10 Hz GPS rate)
            
        Returns:
            Dictionary with time-to-detection metrics
        """
        y_pred_binary = self.convert_predictions(y_pred)
        
        # Find where attack starts
        attack_indices = np.where(y_true == 1)[0]
        if len(attack_indices) == 0:
            return {'time_to_detection': None, 'detection_delay_samples': None}
            
        attack_start = attack_indices[0]
        
        # Find first correct detection after attack starts
        detections_after_start = np.where((y_pred_binary[attack_start:] == 1))[0]
        
        if len(detections_after_start) == 0:
            return {
                'time_to_detection': None,
                'detection_delay_samples': None,
                'detection_status': 'Not detected'
            }
            
        first_detection = detections_after_start[0]
        
        if timestamps is not None:
            time_to_detection = timestamps[attack_start + first_detection] - timestamps[attack_start]
        else:
            time_to_detection = first_detection / sampling_rate
            
        return {
            'time_to_detection': time_to_detection,
            'detection_delay_samples': first_detection,
            'detection_status': 'Detected'
        }
        
    def evaluate_model(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Comprehensive evaluation of a model.
        
        Args:
            model: Trained one-class model
            X_val: Validation features
            y_val: Validation labels
            model_name: Name of the model
            timestamps: Optional timestamps for time-to-detection
            
        Returns:
            Dictionary with all metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions and scores
        y_pred = model.predict(X_val)
        y_scores = model.score_samples(X_val)
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics(y_val, y_pred, model_name)
        roc_metrics = self.calculate_roc_metrics(y_val, y_scores, model_name)
        pr_metrics = self.calculate_pr_metrics(y_val, y_scores, model_name)
        ttd_metrics = self.calculate_time_to_detection(y_val, y_pred, timestamps)
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **ttd_metrics}
        all_metrics['auc'] = roc_metrics.get('auc', 0.0)
        
        return all_metrics
        
    def print_summary(self, model_name: Optional[str] = None):
        """
        Print a summary of evaluation results.
        
        Args:
            model_name: If specified, print only this model. Otherwise print all.
        """
        if model_name:
            models_to_print = {model_name: self.results[model_name]}
        else:
            models_to_print = self.results
            
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        for name, metrics in models_to_print.items():
            print(f"{name}:")
            print(f"  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  Recall (Detection Rate): {metrics.get('recall', 0):.3f}")
            print(f"  F1-Score: {metrics.get('f1_score', 0):.3f}")
            print(f"  AUC: {metrics.get('auc', 0):.3f}")
            print(f"  False Positive Rate: {metrics.get('false_positive_rate', 0):.3f}")
            print(f"  Accuracy: {metrics.get('accuracy', 0):.3f}")
            
            if metrics.get('time_to_detection') is not None:
                print(f"  Time to Detection: {metrics['time_to_detection']:.2f}s")
                
            print(f"  True Positives: {metrics.get('true_positives', 0)}")
            print(f"  False Positives: {metrics.get('false_positives', 0)}")
            print(f"  True Negatives: {metrics.get('true_negatives', 0)}")
            print(f"  False Negatives: {metrics.get('false_negatives', 0)}")
            print()
            
        print(f"{'='*80}\n")
        
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame comparing all models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for name, metrics in self.results.items():
            row = {
                'Model': name,
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'AUC': metrics.get('auc', 0),
                'FPR': metrics.get('false_positive_rate', 0),
                'Accuracy': metrics.get('accuracy', 0),
                'Time-to-Detection': metrics.get('time_to_detection', None)
            }
            comparison_data.append(row)
            
        return pd.DataFrame(comparison_data)
        
    def save_results(self, filepath: str):
        """
        Save results to JSON file.
        
        Args:
            filepath: Path to save file
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            serializable_results[model_name] = serializable_metrics
            
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    print("Metrics Calculator for GPS Spoofing Detection")
    print("="*60)
    print("\nAvailable metrics:")
    print("  - Precision, Recall, F1-Score")
    print("  - ROC-AUC")
    print("  - Confusion Matrix")
    print("  - Detection Rate vs False Positive Rate")
    print("  - Time-to-Detection")
    print("\nUse via the training script: train_models.py")

