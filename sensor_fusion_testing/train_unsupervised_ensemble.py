#!/usr/bin/env python3
"""
Train Unsupervised Ensemble for GPS Spoofing Detection

Trains a pure unsupervised anomaly detection ensemble using ONLY clean data.
No attack labels are used for training or threshold calibration.

Models:
    - IsolationForest: Tree-based isolation scoring
    - PCA Reconstruction: Reconstruction error scoring  
    - LOF: Local density-based scoring

Calibration:
    - Thresholds derived from clean data percentiles
    - Target FPR controls false positive rate
    
Ensemble:
    - Majority voting (2-of-3 models agree)
    - Optional temporal smoothing

Usage:
    python train_unsupervised_ensemble.py --train-dir data/training --target-fpr 2.0
    
Output:
    trained_models_unsupervised/
        - iforest.pkl
        - pca.pkl
        - lof.pkl
        - unsup_ensemble.pkl
        - scaler.pkl
        - config.json
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(__file__))

from ml_models.data_loader import DataLoader
from ml_models.unsupervised_models import (
    create_unsupervised_models,
    train_unsupervised_models,
    calibrate_all_thresholds
)
from ml_models.unsupervised_ensemble import (
    UnsupervisedEnsemble,
    evaluate_unsupervised_ensemble
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train unsupervised ensemble for GPS spoofing detection"
    )
    
    # Data directories
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/training',
        help='Directory containing training CSV files (default: data/training)'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default='data/validation',
        help='Directory containing validation CSV files for evaluation (default: data/validation)'
    )
    
    # Model configuration
    parser.add_argument(
        '--target-fpr',
        type=float,
        default=2.0,
        help='Target false positive rate as percentile (default: 2.0 = 2%% FPR)'
    )
    parser.add_argument(
        '--calibration-split',
        type=float,
        default=0.2,
        help='Fraction of training clean data to hold out for calibration (default: 0.2)'
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--if-n-estimators',
        type=int,
        default=200,
        help='Number of trees for IsolationForest (default: 200)'
    )
    parser.add_argument(
        '--pca-variance',
        type=float,
        default=0.95,
        help='Variance ratio to preserve for PCA (default: 0.95)'
    )
    parser.add_argument(
        '--lof-neighbors',
        type=int,
        default=20,
        help='Number of neighbors for LOF (default: 20)'
    )
    
    # Ensemble configuration
    parser.add_argument(
        '--voting-threshold',
        type=float,
        default=0.5,
        help='Fraction of models required for anomaly vote (default: 0.5 = majority)'
    )
    parser.add_argument(
        '--smoothing-window',
        type=int,
        default=3,
        help='Window size for temporal smoothing (default: 3)'
    )
    parser.add_argument(
        '--smoothing-required',
        type=int,
        default=2,
        help='Required detections in window (default: 2)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='trained_models_unsupervised',
        help='Directory to save trained models (default: trained_models_unsupervised)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def load_data(args) -> dict:
    """
    Load and prepare data for training.
    
    Returns dictionary with:
        - X_train_clean: Clean data for training
        - X_calibration_clean: Clean data for threshold calibration
        - X_val_clean: Clean validation data (for evaluation)
        - X_val_attack: Attack validation data (for evaluation only)
        - feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    loader = DataLoader()
    
    # Load training data
    print(f"\nLoading training data from: {args.train_dir}")
    train_df = loader.load_dataset(args.train_dir)
    
    # Get ONLY clean samples for training
    train_clean_df = train_df[train_df['is_attack_active'] == 0].copy()
    print(f"Clean training samples: {len(train_clean_df)}")
    
    # Get feature matrix and fit scaler
    X_train_all, feature_names = loader.get_feature_matrix(train_clean_df, fit_scaler=True)
    
    # Split clean data: training vs calibration
    X_train_clean, X_calibration_clean = train_test_split(
        X_train_all,
        test_size=args.calibration_split,
        random_state=args.random_state
    )
    print(f"Training samples: {len(X_train_clean)}")
    print(f"Calibration samples: {len(X_calibration_clean)}")
    
    # Load validation data for evaluation
    X_val_clean = np.array([])
    X_val_attack = np.array([])
    
    if args.val_dir and os.path.exists(args.val_dir):
        print(f"\nLoading validation data from: {args.val_dir}")
        val_df = loader.load_dataset(args.val_dir)
        
        val_clean_df = val_df[val_df['is_attack_active'] == 0]
        val_attack_df = val_df[val_df['is_attack_active'] == 1]
        
        if len(val_clean_df) > 0:
            X_val_clean, _ = loader.get_feature_matrix(val_clean_df, fit_scaler=False)
        if len(val_attack_df) > 0:
            X_val_attack, _ = loader.get_feature_matrix(val_attack_df, fit_scaler=False)
            
        print(f"Validation clean samples: {len(X_val_clean)}")
        print(f"Validation attack samples: {len(X_val_attack)}")
    else:
        print("\nNo validation directory specified or found. Evaluation will be skipped.")
    
    print("="*60 + "\n")
    
    return {
        'X_train_clean': X_train_clean,
        'X_calibration_clean': X_calibration_clean,
        'X_val_clean': X_val_clean,
        'X_val_attack': X_val_attack,
        'feature_names': feature_names,
        'scaler': loader.scaler
    }


def create_model_config(args) -> dict:
    """Create model configuration from arguments."""
    return {
        'if_n_estimators': args.if_n_estimators,
        'pca_n_components': args.pca_variance,
        'lof_n_neighbors': args.lof_neighbors,
        'random_state': args.random_state
    }


def save_models(
    models: dict,
    ensemble: UnsupervisedEnsemble,
    scaler,
    args,
    evaluation_results: dict = None
):
    """Save all trained models and configuration."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Save individual models
    for name, model in models.items():
        filepath = os.path.join(output_dir, f"{name}.pkl")
        model.save(filepath)
        
    # Save ensemble
    ensemble_path = os.path.join(output_dir, "unsup_ensemble.pkl")
    ensemble.save(ensemble_path)
    
    # Save scaler
    import joblib
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[Scaler] Saved to {scaler_path}")
    
    # Save configuration
    config = {
        'training_timestamp': datetime.now().isoformat(),
        'target_fpr': args.target_fpr,
        'calibration_split': args.calibration_split,
        'model_config': create_model_config(args),
        'ensemble_config': {
            'voting_threshold': args.voting_threshold,
            'smoothing_window': args.smoothing_window,
            'smoothing_required': args.smoothing_required
        },
        'thresholds': {name: float(model.threshold) for name, model in models.items()},
        'evaluation': evaluation_results if evaluation_results else {}
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Saved to {config_path}")
    
    print("="*60 + "\n")
    print(f"All models saved to: {output_dir}/")


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("UNSUPERVISED ENSEMBLE TRAINING")
    print("="*70)
    print(f"Training directory: {args.train_dir}")
    print(f"Validation directory: {args.val_dir}")
    print(f"Target FPR: {args.target_fpr}%")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    # Load data
    data = load_data(args)
    
    # Create model configuration
    model_config = create_model_config(args)
    
    # Create and train models
    models = create_unsupervised_models(model_config)
    models = train_unsupervised_models(models, data['X_train_clean'])
    
    # Calibrate thresholds from clean data
    thresholds = calibrate_all_thresholds(
        models, 
        data['X_calibration_clean'],
        percentile=args.target_fpr
    )
    
    # Create ensemble
    ensemble = UnsupervisedEnsemble(
        models=models,
        voting_threshold=args.voting_threshold,
        smoothing_window=args.smoothing_window,
        smoothing_required=args.smoothing_required
    )
    
    print("\n" + "="*60)
    print("ENSEMBLE CREATED")
    print("="*60)
    summary = ensemble.get_summary()
    print(f"Models: {summary['model_names']}")
    print(f"Voting threshold: {summary['voting_threshold']}")
    print(f"Smoothing: {summary['smoothing_required']}/{summary['smoothing_window']}")
    print(f"Thresholds: {summary['thresholds']}")
    print("="*60)
    
    # Evaluate on validation data (labels used ONLY for evaluation)
    evaluation_results = None
    if len(data['X_val_clean']) > 0 and len(data['X_val_attack']) > 0:
        evaluation_results = evaluate_unsupervised_ensemble(
            ensemble,
            data['X_val_clean'],
            data['X_val_attack']
        )
    else:
        print("\n[WARNING] Skipping evaluation - no validation data available")
    
    # Save everything
    save_models(models, ensemble, data['scaler'], args, evaluation_results)
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Models saved to: {args.output_dir}/")
    print(f"\nTo use the trained ensemble:")
    print(f"  python detect_spoofing_live.py --mode unsupervised --model-dir {args.output_dir}")
    print("="*70 + "\n")
    
    return ensemble, evaluation_results


if __name__ == "__main__":
    main()

