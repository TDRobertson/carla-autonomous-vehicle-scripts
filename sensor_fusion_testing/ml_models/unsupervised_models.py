"""
Unsupervised Anomaly Detection Models for GPS Spoofing Detection

Implements pure unsupervised models that train ONLY on clean data and use
score-based thresholding calibrated from clean data percentiles (no attack labels).

Models:
    - IsolationForestUnsupervised: Tree-based isolation scoring
    - PCAReconstructionUnsupervised: Reconstruction error-based scoring
    - LOFUnsupervised: Density-based local outlier scoring
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from typing import Optional, Tuple
import joblib


class UnsupervisedModelBase:
    """
    Base class for unsupervised anomaly detection models.
    
    Key principle: Train on clean data only, calibrate thresholds using
    clean score percentiles (no attack labels used for training or calibration).
    """
    
    def __init__(self, name: str):
        """
        Initialize base model.
        
        Args:
            name: Model identifier
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.threshold = None  # Anomaly threshold (calibrated from clean scores)
        
    def fit(self, X_clean: np.ndarray):
        """
        Fit model on clean (non-spoofed) data only.
        
        Args:
            X_clean: Clean training samples (normalized)
        """
        raise NotImplementedError("Subclasses must implement fit()")
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.
        
        Convention: LOWER scores = MORE anomalous (matches sklearn convention)
        
        Args:
            X: Input samples
            
        Returns:
            Anomaly scores (lower = more anomalous)
        """
        raise NotImplementedError("Subclasses must implement score_samples()")
        
    def calibrate_threshold(self, X_clean: np.ndarray, percentile: float = 1.0) -> float:
        """
        Calibrate anomaly threshold from clean data scores.
        
        Uses percentile of clean scores to set threshold. This means
        approximately `percentile`% of clean data will be flagged as anomalies
        (target false positive rate).
        
        Args:
            X_clean: Clean validation samples (must be from held-out data)
            percentile: Target FPR as percentile (1.0 = 1% FPR, 5.0 = 5% FPR)
            
        Returns:
            Calibrated threshold value
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calibrating threshold")
            
        clean_scores = self.score_samples(X_clean)
        
        # Lower scores = more anomalous
        # We want to flag samples with scores below the threshold
        # So threshold = percentile of clean scores (e.g., 1st percentile)
        self.threshold = np.percentile(clean_scores, percentile)
        
        print(f"[{self.name}] Calibrated threshold at {percentile}th percentile: {self.threshold:.4f}")
        
        return self.threshold
        
    def predict_anomaly(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict anomaly labels using threshold.
        
        Args:
            X: Input samples
            threshold: Custom threshold (uses self.threshold if None)
            
        Returns:
            Boolean array (True = anomaly, False = normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        thresh = threshold if threshold is not None else self.threshold
        if thresh is None:
            raise ValueError("Threshold must be set (call calibrate_threshold() first)")
            
        scores = self.score_samples(X)
        
        # Lower scores = more anomalous = True (anomaly detected)
        return scores < thresh
        
    def save(self, filepath: str):
        """Save model and threshold to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        data = {
            'model': self.model,
            'name': self.name,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        joblib.dump(data, filepath)
        print(f"[{self.name}] Saved to {filepath}")
        
    def load(self, filepath: str):
        """Load model and threshold from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.name = data['name']
        self.threshold = data['threshold']
        self.is_trained = data['is_trained']
        print(f"[{self.name}] Loaded from {filepath}")
        return self


class IsolationForestUnsupervised(UnsupervisedModelBase):
    """
    Isolation Forest for unsupervised anomaly detection.
    
    Works by isolating observations through random recursive partitioning.
    Anomalies require fewer partitions to isolate.
    
    Scoring: Uses decision_function which returns the anomaly score.
    Lower scores = more anomalous.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest.
        
        Args:
            n_estimators: Number of isolation trees
            max_samples: Samples to draw for each tree ('auto' = min(256, n_samples))
            max_features: Features to consider for each split
            random_state: Random seed for reproducibility
        """
        super().__init__("IsolationForest")
        
        # Note: We do NOT use contamination parameter here since we're
        # calibrating threshold from clean data separately
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            contamination='auto',  # We ignore this, use our own threshold
            random_state=random_state,
            n_jobs=-1
        )
        
    def fit(self, X_clean: np.ndarray):
        """Train on clean data only."""
        print(f"[{self.name}] Training on {len(X_clean)} clean samples...")
        self.model.fit(X_clean)
        self.is_trained = True
        print(f"[{self.name}] Training complete")
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.
        
        Returns decision_function: lower = more anomalous.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        return self.model.decision_function(X)


class PCAReconstructionUnsupervised(UnsupervisedModelBase):
    """
    PCA Reconstruction Error for anomaly detection.
    
    Projects data to lower-dimensional space and reconstructs. Anomalies
    have high reconstruction error since they don't follow normal patterns.
    
    Scoring: Negative reconstruction error (to match sklearn convention).
    Lower scores = more anomalous (higher reconstruction error).
    """
    
    def __init__(
        self,
        n_components: Optional[float] = 0.95,
        random_state: int = 42
    ):
        """
        Initialize PCA model.
        
        Args:
            n_components: Number of components or variance ratio to preserve
                         (0.95 = keep 95% of variance)
            random_state: Random seed for reproducibility
        """
        super().__init__("PCAReconstruction")
        
        self.n_components = n_components
        self.model = PCA(
            n_components=n_components,
            random_state=random_state
        )
        
    def fit(self, X_clean: np.ndarray):
        """Train PCA on clean data only."""
        print(f"[{self.name}] Training on {len(X_clean)} clean samples...")
        self.model.fit(X_clean)
        self.is_trained = True
        
        # Report variance explained
        n_components_used = self.model.n_components_
        variance_explained = np.sum(self.model.explained_variance_ratio_) * 100
        print(f"[{self.name}] Using {n_components_used} components ({variance_explained:.1f}% variance)")
        print(f"[{self.name}] Training complete")
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores based on reconstruction error.
        
        Returns negative reconstruction error (lower = more anomalous).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
            
        # Project to lower dimensions and reconstruct
        X_projected = self.model.transform(X)
        X_reconstructed = self.model.inverse_transform(X_projected)
        
        # Calculate reconstruction error (MSE per sample)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)
        
        # Return negative error so lower = more anomalous (matches sklearn convention)
        return -reconstruction_error


class LOFUnsupervised(UnsupervisedModelBase):
    """
    Local Outlier Factor for unsupervised anomaly detection.
    
    Compares local density of a point to its neighbors. Points in
    lower-density regions are considered anomalies.
    
    Scoring: Uses decision_function (negative_outlier_factor).
    Lower scores = more anomalous.
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = 'minkowski',
        p: int = 2
    ):
        """
        Initialize LOF.
        
        Args:
            n_neighbors: Number of neighbors for density estimation
            metric: Distance metric
            p: Power parameter for Minkowski metric (2 = Euclidean)
        """
        super().__init__("LOF")
        
        # Note: novelty=True is required for predict() on new data
        # We don't use contamination since we calibrate threshold separately
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,  # Required for scoring new samples
            contamination='auto',  # Ignored, we use own threshold
            metric=metric,
            p=p,
            n_jobs=-1
        )
        
    def fit(self, X_clean: np.ndarray):
        """Train LOF on clean data only."""
        print(f"[{self.name}] Training on {len(X_clean)} clean samples...")
        self.model.fit(X_clean)
        self.is_trained = True
        print(f"[{self.name}] Training complete")
        
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.
        
        Returns decision_function: lower = more anomalous.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        return self.model.decision_function(X)


def create_unsupervised_models(config: Optional[dict] = None) -> dict:
    """
    Create instances of all unsupervised models.
    
    Args:
        config: Optional configuration dictionary with hyperparameters
        
    Returns:
        Dictionary mapping model names to model instances
    """
    if config is None:
        config = {}
        
    models = {
        'iforest': IsolationForestUnsupervised(
            n_estimators=config.get('if_n_estimators', 200),
            max_features=config.get('if_max_features', 1.0),
            random_state=config.get('random_state', 42)
        ),
        'pca': PCAReconstructionUnsupervised(
            n_components=config.get('pca_n_components', 0.95),
            random_state=config.get('random_state', 42)
        ),
        'lof': LOFUnsupervised(
            n_neighbors=config.get('lof_n_neighbors', 20)
        )
    }
    
    return models


def train_unsupervised_models(
    models: dict,
    X_train_clean: np.ndarray
) -> dict:
    """
    Train all unsupervised models on clean data.
    
    Args:
        models: Dictionary of model instances
        X_train_clean: Clean training data (normalized)
        
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*60)
    print("TRAINING UNSUPERVISED MODELS")
    print("="*60)
    print(f"Training data shape: {X_train_clean.shape}")
    print(f"Number of models: {len(models)}\n")
    
    for name, model in models.items():
        try:
            model.fit(X_train_clean)
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60 + "\n")
    
    return models


def calibrate_all_thresholds(
    models: dict,
    X_clean: np.ndarray,
    percentile: float = 2.0
) -> dict:
    """
    Calibrate thresholds for all models using clean data.
    
    Args:
        models: Dictionary of trained model instances
        X_clean: Clean data for calibration (should be held-out from training)
        percentile: Target false positive rate as percentile (e.g., 2.0 = 2% FPR)
        
    Returns:
        Dictionary mapping model names to thresholds
    """
    print("\n" + "="*60)
    print("CALIBRATING THRESHOLDS FROM CLEAN DATA")
    print("="*60)
    print(f"Calibration data shape: {X_clean.shape}")
    print(f"Target FPR: {percentile}% (using {percentile}th percentile)\n")
    
    thresholds = {}
    
    for name, model in models.items():
        try:
            thresh = model.calibrate_threshold(X_clean, percentile)
            thresholds[name] = thresh
        except Exception as e:
            print(f"ERROR calibrating {name}: {e}")
            
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60 + "\n")
    
    return thresholds


if __name__ == "__main__":
    print("Unsupervised Anomaly Detection Models")
    print("="*60)
    print("\nAvailable models:")
    print("  1. IsolationForest - Tree-based isolation scoring")
    print("  2. PCAReconstruction - Reconstruction error scoring")
    print("  3. LOF - Local density-based scoring")
    print("\nKey features:")
    print("  - Train on clean data ONLY (no attack labels)")
    print("  - Calibrate thresholds from clean score percentiles")
    print("  - Pure unsupervised approach for detecting unknown attacks")
    print("\nUse via: python train_unsupervised_ensemble.py")

