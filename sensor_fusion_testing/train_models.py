#!/usr/bin/env python3
"""
Main Training Script for GPS Spoofing Detection Models

This script:
1. Loads training and validation data
2. Trains all four one-class classifiers
3. Evaluates each model individually
4. Creates and evaluates ensemble voting systems
5. Saves trained models
6. Generates performance visualizations

Usage:
    python train_models.py --train-dir data/training --val-dir data/validation
"""

import sys
import os
import argparse
from datetime import datetime

# Add ml_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml_models'))

from ml_models.data_loader import DataLoader
from ml_models.one_class_trainer import create_all_models, train_all_models
from ml_models.metrics import MetricsCalculator
from ml_models.ensemble import EnsembleVoting, create_weighted_ensemble, evaluate_ensemble_strategies
from ml_models.visualizer import ResultsVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for GPS spoofing detection"
    )
    parser.add_argument(
        '--train-dir', type=str, default='data/training',
        help='Directory with training data (default: data/training)'
    )
    parser.add_argument(
        '--val-dir', type=str, default=None,
        help='Directory with validation data (default: use train/test split)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='trained_models',
        help='Directory to save trained models (default: trained_models)'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory to save results and visualizations (default: results)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Test size for train/val split if val-dir not provided (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GPS SPOOFING DETECTION - MODEL TRAINING")
    print("="*80)
    print(f"Training data: {args.train_dir}")
    print(f"Validation data: {args.val_dir if args.val_dir else 'Split from training'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Results directory: {args.results_dir}")
    print("="*80 + "\n")
    
    # Step 1: Load and prepare data
    print("STEP 1: Loading and preparing data...")
    print("-" * 80)
    
    loader = DataLoader()
    data = loader.load_and_prepare(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_size=args.test_size
    )
    
    X_train_clean = data['X_train_clean']
    X_val = data['X_val']
    y_val = data['y_val']
    feature_names = data['feature_names']
    
    # Save scaler
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    loader.save_scaler(scaler_path)
    
    # Step 2: Create and train all models
    print("\nSTEP 2: Training one-class classifiers...")
    print("-" * 80)
    
    models = create_all_models()
    models = train_all_models(models, X_train_clean)
    
    # Step 3: Evaluate individual models
    print("\nSTEP 3: Evaluating individual models...")
    print("-" * 80)
    
    metrics_calc = MetricsCalculator()
    
    for name, model in models.items():
        metrics = metrics_calc.evaluate_model(model, X_val, y_val, name)
        
    metrics_calc.print_summary()
    
    # Step 4: Create and evaluate ensemble
    print("\nSTEP 4: Creating ensemble voting system...")
    print("-" * 80)
    
    # Get validation metrics for weighted ensemble
    validation_metrics = metrics_calc.results
    
    # Evaluate all ensemble strategies
    ensemble_preds = evaluate_ensemble_strategies(
        models, X_val, y_val, validation_metrics
    )
    
    # Evaluate ensemble performance
    for ensemble_name, ensemble_pred in ensemble_preds.items():
        metrics_calc.calculate_basic_metrics(y_val, ensemble_pred, ensemble_name)
        
    # Create the best ensemble (weighted by F1-score)
    best_ensemble = create_weighted_ensemble(models, validation_metrics)
    
    # Step 5: Save models
    print("\nSTEP 5: Saving trained models...")
    print("-" * 80)
    
    for name, model in models.items():
        model_path = os.path.join(args.output_dir, f"{name}.pkl")
        model.save_model(model_path)
        
    ensemble_path = os.path.join(args.output_dir, "ensemble.pkl")
    best_ensemble.save(ensemble_path)
    
    # Save metrics
    metrics_path = os.path.join(args.results_dir, "performance_report.json")
    metrics_calc.save_results(metrics_path)
    
    # Step 6: Generate visualizations
    print("\nSTEP 6: Generating visualizations...")
    print("-" * 80)
    
    try:
        visualizer = ResultsVisualizer(metrics_calc.results, args.results_dir)
        
        # Create all visualizations
        visualizer.plot_model_comparison()
        visualizer.plot_confusion_matrices(models, X_val, y_val)
        visualizer.plot_roc_curves()
        visualizer.plot_precision_recall_curves()
        
        print("Visualizations saved successfully")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
        print("This is likely due to missing matplotlib. Install with: pip install matplotlib")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModels saved to: {args.output_dir}/")
    print(f"Results saved to: {args.results_dir}/")
    print("\nModel Performance Summary:")
    
    comparison_df = metrics_calc.get_comparison_dataframe()
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review visualizations in results/")
    print("  2. Test models with: python detect_spoofing_live.py")
    print("  3. Collect more data if performance is insufficient")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

