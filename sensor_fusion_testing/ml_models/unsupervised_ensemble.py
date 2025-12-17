"""
Unsupervised Ensemble for GPS Spoofing Detection

Combines multiple unsupervised anomaly detection models with:
    - Threshold-calibrated voting (thresholds from clean data percentiles)
    - Majority voting (2-of-3 models agree = anomaly)
    - Optional temporal smoothing (require N detections in M seconds)
    
No attack labels are used for training or calibration.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
import joblib

from .unsupervised_models import (
    UnsupervisedModelBase,
    IsolationForestUnsupervised,
    PCAReconstructionUnsupervised,
    LOFUnsupervised
)


class UnsupervisedEnsemble:
    """
    Ensemble of unsupervised anomaly detection models.
    
    Uses threshold-calibrated voting where each model's threshold is
    derived from clean data percentiles (no attack labels).
    
    Voting strategy:
        1. Each model produces a score for the sample
        2. Score is compared against model's threshold (calibrated from clean data)
        3. If score < threshold, model votes "anomaly"
        4. If majority (2-of-3) vote anomaly, ensemble predicts anomaly
        
    Optional temporal smoothing:
        - Requires N detections in last M samples to trigger alert
        - Reduces false positives from transient noise
    """
    
    def __init__(
        self,
        models: Dict[str, UnsupervisedModelBase],
        voting_threshold: float = 0.5,
        smoothing_window: int = 3,
        smoothing_required: int = 2
    ):
        """
        Initialize ensemble.
        
        Args:
            models: Dictionary of trained unsupervised models with calibrated thresholds
            voting_threshold: Fraction of models that must vote anomaly (0.5 = majority)
            smoothing_window: Number of recent samples to consider for smoothing
            smoothing_required: Minimum detections in window to trigger alert
        """
        self.models = models
        self.voting_threshold = voting_threshold
        self.smoothing_window = smoothing_window
        self.smoothing_required = smoothing_required
        
        # Detection history for temporal smoothing
        self.detection_history: deque = deque(maxlen=smoothing_window)
        
        # Validate all models have thresholds
        for name, model in models.items():
            if model.threshold is None:
                raise ValueError(f"Model '{name}' has no calibrated threshold. "
                               "Call calibrate_threshold() first.")
                               
    def reset_smoothing(self):
        """Reset the temporal smoothing buffer."""
        self.detection_history.clear()
        
    def get_individual_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get raw anomaly scores from each model.
        
        Args:
            X: Input samples (normalized)
            
        Returns:
            Dictionary mapping model names to score arrays
        """
        scores = {}
        for name, model in self.models.items():
            scores[name] = model.score_samples(X)
        return scores
        
    def get_individual_votes(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get binary votes from each model (before ensemble voting).
        
        Args:
            X: Input samples (normalized)
            
        Returns:
            Dictionary mapping model names to boolean arrays (True = anomaly)
        """
        votes = {}
        for name, model in self.models.items():
            votes[name] = model.predict_anomaly(X)
        return votes
        
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using majority voting (no temporal smoothing).
        
        Args:
            X: Input samples (normalized)
            
        Returns:
            Boolean array (True = anomaly detected)
        """
        # Get votes from all models
        votes = self.get_individual_votes(X)
        
        # Stack votes: shape (n_models, n_samples)
        vote_matrix = np.array(list(votes.values()))
        
        # Calculate fraction of models voting anomaly for each sample
        anomaly_fraction = np.mean(vote_matrix.astype(float), axis=0)
        
        # Majority vote
        is_anomaly = anomaly_fraction >= self.voting_threshold
        
        return is_anomaly
        
    def predict_single(self, x: np.ndarray, use_smoothing: bool = True) -> Tuple[bool, Dict]:
        """
        Predict for a single sample with optional temporal smoothing.
        
        Use this for real-time detection where you want to smooth out noise.
        
        Args:
            x: Single sample as 1D array (will be reshaped to 2D)
            use_smoothing: Whether to apply temporal smoothing
            
        Returns:
            Tuple of:
                - is_anomaly: Final detection decision
                - info: Dictionary with per-model votes, scores, and smoothing state
        """
        # Reshape to 2D if needed
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # Get individual model results
        scores = self.get_individual_scores(x)
        votes = self.get_individual_votes(x)
        
        # Raw ensemble vote (before smoothing)
        raw_detection = self.predict_raw(x)[0]
        
        # Build info dict
        info = {
            'scores': {name: float(s[0]) for name, s in scores.items()},
            'votes': {name: bool(v[0]) for name, v in votes.items()},
            'thresholds': {name: model.threshold for name, model in self.models.items()},
            'raw_detection': raw_detection,
            'smoothing_enabled': use_smoothing
        }
        
        if use_smoothing:
            # Add to history and apply smoothing
            self.detection_history.append(raw_detection)
            
            # Count recent detections
            n_recent_detections = sum(self.detection_history)
            is_anomaly = n_recent_detections >= self.smoothing_required
            
            info['smoothing_window'] = self.smoothing_window
            info['smoothing_required'] = self.smoothing_required
            info['recent_detections'] = n_recent_detections
            info['history_len'] = len(self.detection_history)
        else:
            is_anomaly = raw_detection
            
        info['final_detection'] = is_anomaly
        
        return is_anomaly, info
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies for batch of samples (no temporal smoothing).
        
        For real-time detection with smoothing, use predict_single().
        
        Args:
            X: Input samples (normalized)
            
        Returns:
            Boolean array (True = anomaly detected)
        """
        return self.predict_raw(X)
        
    def predict_sklearn_format(self, X: np.ndarray) -> np.ndarray:
        """
        Predict in sklearn format (-1 = anomaly, 1 = normal).
        
        Args:
            X: Input samples
            
        Returns:
            Array of predictions (-1 = anomaly, 1 = normal)
        """
        is_anomaly = self.predict(X)
        return np.where(is_anomaly, -1, 1)
        
    def get_summary(self) -> Dict:
        """
        Get summary of ensemble configuration.
        
        Returns:
            Dictionary with configuration details
        """
        return {
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'voting_threshold': self.voting_threshold,
            'smoothing_window': self.smoothing_window,
            'smoothing_required': self.smoothing_required,
            'thresholds': {name: model.threshold for name, model in self.models.items()}
        }
        
    def save(self, filepath: str):
        """
        Save ensemble to disk.
        
        Args:
            filepath: Path to save file (.pkl)
        """
        data = {
            'models': self.models,
            'voting_threshold': self.voting_threshold,
            'smoothing_window': self.smoothing_window,
            'smoothing_required': self.smoothing_required
        }
        joblib.dump(data, filepath)
        print(f"[UnsupervisedEnsemble] Saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'UnsupervisedEnsemble':
        """
        Load ensemble from disk.
        
        Args:
            filepath: Path to saved file (.pkl)
            
        Returns:
            Loaded UnsupervisedEnsemble instance
        """
        data = joblib.load(filepath)
        ensemble = cls(
            models=data['models'],
            voting_threshold=data['voting_threshold'],
            smoothing_window=data['smoothing_window'],
            smoothing_required=data['smoothing_required']
        )
        print(f"[UnsupervisedEnsemble] Loaded from {filepath}")
        return ensemble


def evaluate_unsupervised_ensemble(
    ensemble: UnsupervisedEnsemble,
    X_clean: np.ndarray,
    X_attack: np.ndarray
) -> Dict:
    """
    Evaluate ensemble performance on clean and attack data.
    
    Note: Attack labels are used ONLY for evaluation, not training or calibration.
    
    Args:
        ensemble: Trained and calibrated ensemble
        X_clean: Clean validation samples
        X_attack: Attack validation samples
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING UNSUPERVISED ENSEMBLE")
    print("="*60)
    
    # Predict on clean data
    clean_preds = ensemble.predict(X_clean)
    false_positives = np.sum(clean_preds)  # Anomalies in clean data = FP
    fpr = false_positives / len(X_clean)
    
    # Predict on attack data
    attack_preds = ensemble.predict(X_attack)
    true_positives = np.sum(attack_preds)  # Anomalies in attack data = TP
    tpr = true_positives / len(X_attack)  # Detection rate
    
    # Combined metrics
    n_total = len(X_clean) + len(X_attack)
    accuracy = (len(X_clean) - false_positives + true_positives) / n_total
    
    # Precision, Recall, F1
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0
        
    recall = tpr  # Same as detection rate
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    # Print results
    print(f"\nClean Data ({len(X_clean)} samples):")
    print(f"  False Positives: {false_positives} ({fpr*100:.2f}%)")
    
    print(f"\nAttack Data ({len(X_attack)} samples):")
    print(f"  True Positives: {true_positives} ({tpr*100:.2f}%)")
    print(f"  Missed: {len(X_attack) - true_positives}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall (Detection Rate): {recall*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    
    # Per-model breakdown
    print(f"\nPer-Model Results:")
    for name, model in ensemble.models.items():
        clean_votes = model.predict_anomaly(X_clean)
        attack_votes = model.predict_anomaly(X_attack)
        model_fpr = np.mean(clean_votes)
        model_tpr = np.mean(attack_votes)
        print(f"  {name:15s}: FPR={model_fpr*100:.2f}%, TPR={model_tpr*100:.2f}%")
    
    print("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': fpr,
        'detection_rate': tpr,
        'n_clean': len(X_clean),
        'n_attack': len(X_attack),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': len(X_clean) - int(false_positives),
        'false_negatives': len(X_attack) - int(true_positives)
    }


def create_and_calibrate_ensemble(
    X_train_clean: np.ndarray,
    X_calibration_clean: np.ndarray,
    target_fpr: float = 2.0,
    model_config: Optional[Dict] = None,
    voting_threshold: float = 0.5,
    smoothing_window: int = 3,
    smoothing_required: int = 2
) -> UnsupervisedEnsemble:
    """
    Create, train, and calibrate a complete unsupervised ensemble.
    
    Args:
        X_train_clean: Clean data for training models
        X_calibration_clean: Clean data for calibrating thresholds (should be held-out)
        target_fpr: Target false positive rate as percentile (2.0 = 2% FPR)
        model_config: Optional model hyperparameters
        voting_threshold: Fraction of models required for anomaly vote
        smoothing_window: Window size for temporal smoothing
        smoothing_required: Required detections in window
        
    Returns:
        Trained and calibrated UnsupervisedEnsemble
    """
    from .unsupervised_models import create_unsupervised_models, train_unsupervised_models
    
    # Create and train models
    models = create_unsupervised_models(model_config)
    models = train_unsupervised_models(models, X_train_clean)
    
    # Calibrate thresholds on held-out clean data
    print("\n" + "="*60)
    print("CALIBRATING THRESHOLDS")
    print("="*60)
    print(f"Using {len(X_calibration_clean)} clean samples for calibration")
    print(f"Target FPR: {target_fpr}%\n")
    
    for name, model in models.items():
        model.calibrate_threshold(X_calibration_clean, percentile=target_fpr)
        
    # Create ensemble
    ensemble = UnsupervisedEnsemble(
        models=models,
        voting_threshold=voting_threshold,
        smoothing_window=smoothing_window,
        smoothing_required=smoothing_required
    )
    
    print("\n[UnsupervisedEnsemble] Created successfully")
    print(f"  Models: {list(models.keys())}")
    print(f"  Voting threshold: {voting_threshold}")
    print(f"  Smoothing: {smoothing_required}/{smoothing_window}")
    
    return ensemble


if __name__ == "__main__":
    print("Unsupervised Ensemble for GPS Spoofing Detection")
    print("="*60)
    print("\nKey features:")
    print("  - Threshold-calibrated voting (clean data percentiles)")
    print("  - Majority voting (2-of-3 = anomaly)")
    print("  - Optional temporal smoothing")
    print("  - NO attack labels used for training or calibration")
    print("\nUse via: python train_unsupervised_ensemble.py")

