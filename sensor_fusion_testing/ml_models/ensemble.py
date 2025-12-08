"""
Ensemble Voting System for GPS Spoofing Detection

Combines predictions from multiple one-class classifiers to improve
detection accuracy and reduce false positives.
"""

import numpy as np
from typing import Dict, List, Optional
import joblib
from .one_class_trainer import OneClassTrainer


class EnsembleVoting:
    """
    Ensemble voting system that combines multiple one-class classifiers.
    
    Supports three voting strategies:
    1. Majority voting - Flag as anomaly if majority of models agree
    2. Weighted voting - Weight each model by validation performance
    3. Confidence voting - Use anomaly scores with thresholds
    """
    
    def __init__(
        self,
        models: Dict[str, OneClassTrainer],
        voting_strategy: str = 'majority',
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5
    ):
        """
        Initialize ensemble voting system.
        
        Args:
            models: Dictionary of trained one-class models
            voting_strategy: 'majority', 'weighted', or 'confidence'
            weights: Optional weights for each model (for weighted voting)
            threshold: Threshold for confidence voting (0-1)
        """
        self.models = models
        self.voting_strategy = voting_strategy
        self.weights = weights
        self.threshold = threshold
        
        if voting_strategy == 'weighted' and weights is None:
            # Default to equal weights
            self.weights = {name: 1.0 for name in models.keys()}
            
    def predict_majority(self, X: np.ndarray) -> np.ndarray:
        """
        Majority voting: Flag as anomaly if majority of models agree.
        
        Args:
            X: Input samples
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal)
        """
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            # Convert to binary: -1 (anomaly) -> 1, 1 (normal) -> 0
            pred_binary = np.where(pred == -1, 1, 0)
            predictions.append(pred_binary)
            
        # Stack predictions and take majority vote
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        majority_vote = np.mean(predictions, axis=0)  # Average across models
        
        # If majority (>0.5) vote anomaly, predict -1, else 1
        ensemble_pred = np.where(majority_vote > 0.5, -1, 1)
        
        return ensemble_pred
        
    def predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted voting: Weight each model's vote by its performance.
        
        Args:
            X: Input samples
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal)
        """
        weighted_votes = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            pred = model.predict(X)
            # Convert to binary
            pred_binary = np.where(pred == -1, 1, 0)
            # Apply weight
            weight = self.weights.get(name, 1.0)
            weighted_votes.append(pred_binary * weight)
            
        # Combine weighted votes
        weighted_sum = np.sum(weighted_votes, axis=0)
        weighted_avg = weighted_sum / total_weight
        
        # Threshold at 0.5
        ensemble_pred = np.where(weighted_avg > 0.5, -1, 1)
        
        return ensemble_pred
        
    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Confidence voting: Use anomaly scores with threshold.
        
        Args:
            X: Input samples
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal)
        """
        scores = []
        for model in self.models.values():
            try:
                score = model.score_samples(X)
                # Normalize scores to [0, 1] range (approximately)
                # Lower scores = more anomalous
                # We invert so higher = more anomalous
                score_normalized = -score
                scores.append(score_normalized)
            except Exception as e:
                print(f"Warning: Could not get scores from model: {e}")
                # Fallback to prediction
                pred = model.predict(X)
                pred_binary = np.where(pred == -1, 1, 0)
                scores.append(pred_binary)
                
        # Average scores
        avg_scores = np.mean(scores, axis=0)
        
        # Apply threshold (using median as threshold point)
        threshold_value = np.median(avg_scores)
        ensemble_pred = np.where(avg_scores > threshold_value, -1, 1)
        
        return ensemble_pred
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the selected voting strategy.
        
        Args:
            X: Input samples
            
        Returns:
            Predictions (-1 for anomaly, 1 for normal)
        """
        if self.voting_strategy == 'majority':
            return self.predict_majority(X)
        elif self.voting_strategy == 'weighted':
            return self.predict_weighted(X)
        elif self.voting_strategy == 'confidence':
            return self.predict_confidence(X)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
            
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.
        
        Args:
            X: Input samples
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions
        
    def get_agreement_score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate agreement score for each sample across all models.
        
        Args:
            X: Input samples
            
        Returns:
            Agreement scores (0-1) for each sample
        """
        predictions = []
        for model in self.models.values():
            pred = model.predict(X)
            pred_binary = np.where(pred == -1, 1, 0)
            predictions.append(pred_binary)
            
        predictions = np.array(predictions)
        
        # Calculate agreement as std dev (lower = more agreement)
        # Or as fraction of models that agree with majority
        majority = np.median(predictions, axis=0)
        agreement = np.mean(predictions == majority, axis=0)
        
        return agreement
        
    def save(self, filepath: str):
        """Save the ensemble system to disk."""
        ensemble_data = {
            'models': self.models,
            'voting_strategy': self.voting_strategy,
            'weights': self.weights,
            'threshold': self.threshold
        }
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str):
        """Load an ensemble system from disk."""
        ensemble_data = joblib.load(filepath)
        return cls(
            models=ensemble_data['models'],
            voting_strategy=ensemble_data['voting_strategy'],
            weights=ensemble_data['weights'],
            threshold=ensemble_data['threshold']
        )


def create_weighted_ensemble(
    models: Dict[str, OneClassTrainer],
    validation_metrics: Dict[str, Dict]
) -> EnsembleVoting:
    """
    Create a weighted ensemble based on validation performance.
    
    Args:
        models: Dictionary of trained models
        validation_metrics: Dictionary of validation metrics for each model
        
    Returns:
        EnsembleVoting instance with weights based on F1-scores
    """
    weights = {}
    
    for name in models.keys():
        if name in validation_metrics:
            # Use F1-score as weight (could also use AUC or other metric)
            f1_score = validation_metrics[name].get('f1_score', 0.5)
            weights[name] = max(f1_score, 0.1)  # Minimum weight of 0.1
        else:
            weights[name] = 0.5  # Default weight
            
    print(f"\nEnsemble weights based on validation performance:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.3f}")
        
    return EnsembleVoting(
        models=models,
        voting_strategy='weighted',
        weights=weights
    )


def evaluate_ensemble_strategies(
    models: Dict[str, OneClassTrainer],
    X_val: np.ndarray,
    y_val: np.ndarray,
    validation_metrics: Optional[Dict[str, Dict]] = None
) -> Dict[str, np.ndarray]:
    """
    Evaluate all ensemble strategies and return predictions.
    
    Args:
        models: Dictionary of trained models
        X_val: Validation features
        y_val: Validation labels
        validation_metrics: Optional validation metrics for weighted ensemble
        
    Returns:
        Dictionary mapping strategy names to predictions
    """
    strategies = ['majority', 'weighted', 'confidence']
    predictions = {}
    
    print(f"\n{'='*60}")
    print("EVALUATING ENSEMBLE STRATEGIES")
    print(f"{'='*60}\n")
    
    for strategy in strategies:
        print(f"Testing {strategy} voting...")
        
        if strategy == 'weighted' and validation_metrics is not None:
            ensemble = create_weighted_ensemble(models, validation_metrics)
        else:
            ensemble = EnsembleVoting(models, voting_strategy=strategy)
            
        pred = ensemble.predict(X_val)
        predictions[f'ensemble_{strategy}'] = pred
        
        # Quick accuracy calculation
        pred_binary = np.where(pred == -1, 1, 0)
        accuracy = np.mean(pred_binary == y_val)
        print(f"  Accuracy: {accuracy:.3f}\n")
        
    print(f"{'='*60}\n")
    
    return predictions


if __name__ == "__main__":
    print("Ensemble Voting System for GPS Spoofing Detection")
    print("="*60)
    print("\nVoting strategies:")
    print("  1. Majority - Simple majority vote")
    print("  2. Weighted - Weight by validation performance")
    print("  3. Confidence - Use anomaly scores with thresholds")
    print("\nUse via the training script: train_models.py")

