"""
One-Class Classifiers for GPS Spoofing Detection

Implements anomaly detection models that train only on clean (non-spoofed) data
and detect attacks as anomalies.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from typing import Dict, Optional
import joblib


class OneClassTrainer:
    """
    Base class for one-class anomaly detection models.
    
    One-class classifiers learn what "normal" looks like from clean data,
    then flag anything unusual as an anomaly (potential attack).
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name of the model for identification
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    def fit(self, X_clean: np.ndarray):
        """
        Train the model on clean data only.
        
        Args:
            X_clean: Clean (non-spoofed) training samples
        """
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Samples to predict
            
        Returns:
            Array of predictions: -1 for anomaly (attack), 1 for normal (clean)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        
        Args:
            X: Samples to score
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
            
        if hasattr(self.model, 'score_samples'):
            return self.model.score_samples(X)
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support anomaly scoring")
            
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, filepath)
        print(f"{self.model_name} saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"{self.model_name} loaded from {filepath}")


class IsolationForestModel(OneClassTrainer):
    """
    Isolation Forest for anomaly detection.
    
    Works by isolating anomalies through random partitioning.
    Anomalies require fewer partitions to isolate, making them easy to detect.
    
    Best for: High-dimensional data, fast training
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        max_features: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest.
        
        Args:
            n_estimators: Number of trees
            contamination: Expected proportion of anomalies in test data
            max_features: Features to consider for each split
            random_state: Random seed
        """
        super().__init__("Isolation Forest")
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
    def fit(self, X_clean: np.ndarray):
        """Train Isolation Forest on clean data."""
        print(f"Training {self.model_name} on {len(X_clean)} clean samples...")
        self.model.fit(X_clean)
        self.is_trained = True
        print(f"{self.model_name} training complete")


class OneClassSVMModel(OneClassTrainer):
    """
    One-Class Support Vector Machine for anomaly detection.
    
    Learns a boundary around normal data in feature space.
    Points outside the boundary are classified as anomalies.
    
    Best for: Complex decision boundaries, kernel tricks
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        nu: float = 0.1
    ):
        """
        Initialize One-Class SVM.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            gamma: Kernel coefficient
            nu: Upper bound on fraction of training errors
        """
        super().__init__("One-Class SVM")
        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu
        )
        
    def fit(self, X_clean: np.ndarray):
        """Train One-Class SVM on clean data."""
        print(f"Training {self.model_name} on {len(X_clean)} clean samples...")
        self.model.fit(X_clean)
        self.is_trained = True
        print(f"{self.model_name} training complete")


class LOFModel(OneClassTrainer):
    """
    Local Outlier Factor for anomaly detection.
    
    Compares local density of a point to the local densities of its neighbors.
    Points in regions of lower density are classified as anomalies.
    
    Best for: Detecting local anomalies, density-based detection
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        novelty: bool = True
    ):
        """
        Initialize Local Outlier Factor.
        
        Args:
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of anomalies
            novelty: Whether to use novelty detection mode (required for predict())
        """
        super().__init__("Local Outlier Factor")
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty  # Must be True to use predict()
        )
        
    def fit(self, X_clean: np.ndarray):
        """Train LOF on clean data."""
        print(f"Training {self.model_name} on {len(X_clean)} clean samples...")
        self.model.fit(X_clean)
        self.is_trained = True
        print(f"{self.model_name} training complete")


class EllipticEnvelopeModel(OneClassTrainer):
    """
    Elliptic Envelope for anomaly detection.
    
    Assumes data follows a Gaussian distribution and fits an ellipse around it.
    Points outside the ellipse are classified as anomalies.
    
    Best for: Gaussian-distributed features, fast inference
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        support_fraction: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize Elliptic Envelope.
        
        Args:
            contamination: Expected proportion of anomalies
            support_fraction: Proportion of points to include in support (None = auto)
            random_state: Random seed
        """
        super().__init__("Elliptic Envelope")
        self.model = EllipticEnvelope(
            contamination=contamination,
            support_fraction=support_fraction,
            random_state=random_state
        )
        
    def fit(self, X_clean: np.ndarray):
        """Train Elliptic Envelope on clean data."""
        print(f"Training {self.model_name} on {len(X_clean)} clean samples...")
        self.model.fit(X_clean)
        self.is_trained = True
        print(f"{self.model_name} training complete")


def create_all_models(config: Optional[Dict] = None) -> Dict[str, OneClassTrainer]:
    """
    Create instances of all one-class models with default or custom configurations.
    
    Args:
        config: Optional dictionary with model configurations
        
    Returns:
        Dictionary mapping model names to model instances
    """
    if config is None:
        config = {}
        
    models = {
        'isolation_forest': IsolationForestModel(
            n_estimators=config.get('if_n_estimators', 100),
            contamination=config.get('contamination', 0.1),
            max_features=config.get('if_max_features', 1.0),
            random_state=config.get('random_state', 42)
        ),
        'one_class_svm': OneClassSVMModel(
            kernel=config.get('svm_kernel', 'rbf'),
            gamma=config.get('svm_gamma', 'scale'),
            nu=config.get('svm_nu', 0.1)
        ),
        'lof': LOFModel(
            n_neighbors=config.get('lof_n_neighbors', 20),
            contamination=config.get('contamination', 0.1),
            novelty=True
        ),
        'elliptic_envelope': EllipticEnvelopeModel(
            contamination=config.get('contamination', 0.1),
            support_fraction=config.get('ee_support_fraction', None),
            random_state=config.get('random_state', 42)
        )
    }
    
    return models


def train_all_models(
    models: Dict[str, OneClassTrainer],
    X_train_clean: np.ndarray
) -> Dict[str, OneClassTrainer]:
    """
    Train all models on clean training data.
    
    Args:
        models: Dictionary of model instances
        X_train_clean: Clean training samples
        
    Returns:
        Dictionary of trained models
    """
    print(f"\n{'='*60}")
    print("TRAINING ALL MODELS")
    print(f"{'='*60}")
    print(f"Training data shape: {X_train_clean.shape}")
    print(f"Number of models: {len(models)}\n")
    
    for name, model in models.items():
        try:
            model.fit(X_train_clean)
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}\n")
    
    return models


if __name__ == "__main__":
    # Example usage
    print("One-Class Classifier Models for GPS Spoofing Detection")
    print("="*60)
    print("\nAvailable models:")
    print("  1. Isolation Forest - Tree-based isolation")
    print("  2. One-Class SVM - Kernel-based boundary")
    print("  3. Local Outlier Factor - Density-based detection")
    print("  4. Elliptic Envelope - Gaussian assumption")
    print("\nUse these models via the training script: train_models.py")

