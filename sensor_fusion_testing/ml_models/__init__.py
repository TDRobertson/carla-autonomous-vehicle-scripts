"""
ML Models for GPS Spoofing Detection

This package contains machine learning models and utilities for detecting
GPS spoofing attacks using one-class anomaly detection techniques.

Includes both:
    - Supervised one-class classifiers (original approach)
    - Pure unsupervised ensemble (new, no attack labels)
"""

from .data_loader import DataLoader
from .one_class_trainer import (
    OneClassTrainer,
    IsolationForestModel,
    OneClassSVMModel,
    LOFModel,
    EllipticEnvelopeModel
)
from .metrics import MetricsCalculator
from .ensemble import EnsembleVoting
from .visualizer import ResultsVisualizer
from .unsupervised_models import (
    UnsupervisedModelBase,
    IsolationForestUnsupervised,
    PCAReconstructionUnsupervised,
    LOFUnsupervised,
    create_unsupervised_models,
    train_unsupervised_models,
    calibrate_all_thresholds
)
from .unsupervised_ensemble import (
    UnsupervisedEnsemble,
    evaluate_unsupervised_ensemble,
    create_and_calibrate_ensemble
)

__all__ = [
    # Data loading
    'DataLoader',
    # Original one-class models
    'OneClassTrainer',
    'IsolationForestModel',
    'OneClassSVMModel',
    'LOFModel',
    'EllipticEnvelopeModel',
    # Metrics and visualization
    'MetricsCalculator',
    'EnsembleVoting',
    'ResultsVisualizer',
    # Unsupervised models (new)
    'UnsupervisedModelBase',
    'IsolationForestUnsupervised',
    'PCAReconstructionUnsupervised',
    'LOFUnsupervised',
    'create_unsupervised_models',
    'train_unsupervised_models',
    'calibrate_all_thresholds',
    # Unsupervised ensemble (new)
    'UnsupervisedEnsemble',
    'evaluate_unsupervised_ensemble',
    'create_and_calibrate_ensemble',
]

