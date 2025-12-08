"""
Data Loader for GPS Spoofing Detection ML Models

Handles loading, preprocessing, and feature selection for the ML training pipeline.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional


class DataLoader:
    """
    Load and preprocess GPS/IMU data for ML training.
    
    This class handles:
    - Loading multiple CSV files from a directory
    - Separating clean vs attack samples
    - Feature selection and engineering
    - Data normalization/standardization
    - Train/validation splits
    """
    
    # Primary features for anomaly detection
    PRIMARY_FEATURES = [
        'innovation_spoof',
        'innovation_spoof_ma',
        'innovation_spoof_std',
        'innovation_spoof_max',
        'position_error',
        'kf_tracking_error',
        'accel_magnitude',
        'gyro_magnitude',
        'jerk_magnitude',
        'position_error_ma',
        'position_error_std',
        'kf_tracking_error_ma',
        'kf_tracking_error_std',
        'innovation_diff',
        'kf_diff_magnitude',
    ]
    
    def __init__(self, feature_list: Optional[List[str]] = None):
        """
        Initialize the data loader.
        
        Args:
            feature_list: List of feature names to use. If None, uses PRIMARY_FEATURES.
        """
        self.feature_list = feature_list if feature_list is not None else self.PRIMARY_FEATURES
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_dataset(self, directory: str, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load all CSV files from a directory and combine them.
        
        Args:
            directory: Path to directory containing CSV files
            pattern: Glob pattern for CSV files (default: "*.csv")
            
        Returns:
            Combined DataFrame with all data
        """
        csv_files = glob.glob(os.path.join(directory, pattern))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {directory} matching pattern {pattern}")
            
        print(f"Loading {len(csv_files)} CSV files from {directory}")
        
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add source file for tracking
                df['source_file'] = os.path.basename(csv_file)
                dataframes.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                
        if not dataframes:
            raise ValueError(f"No valid CSV files could be loaded from {directory}")
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} total samples")
        
        return combined_df
        
    def prepare_one_class_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract clean (non-spoofed) samples for one-class training.
        
        Args:
            df: Full dataset
            
        Returns:
            DataFrame with only clean samples
        """
        clean_df = df[df['is_attack_active'] == 0].copy()
        print(f"Extracted {len(clean_df)} clean samples for training")
        return clean_df
        
    def prepare_test_data(self, df: pd.DataFrame, include_clean: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare test data with both clean and attack samples.
        
        Args:
            df: Full dataset
            include_clean: If True, includes clean samples in test set
            
        Returns:
            Tuple of (clean_df, attack_df)
        """
        clean_df = df[df['is_attack_active'] == 0].copy()
        attack_df = df[df['is_attack_active'] == 1].copy()
        
        print(f"Test data: {len(clean_df)} clean, {len(attack_df)} attack samples")
        
        if include_clean:
            return clean_df, attack_df
        else:
            return pd.DataFrame(), attack_df
            
    def get_feature_matrix(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Extract feature matrix from DataFrame.
        
        Args:
            df: DataFrame with features
            fit_scaler: If True, fits the scaler on this data
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Select only available features from the feature list
        available_features = [f for f in self.feature_list if f in df.columns]
        
        if not available_features:
            raise ValueError(f"None of the specified features found in DataFrame. "
                           f"Available columns: {df.columns.tolist()}")
                           
        if len(available_features) < len(self.feature_list):
            missing = set(self.feature_list) - set(available_features)
            print(f"Warning: Missing features: {missing}")
            
        print(f"Using {len(available_features)} features: {available_features}")
        
        # Extract feature matrix
        X = df[available_features].values
        
        # Handle any NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Normalize features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = available_features
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled, available_features
        
    def load_and_prepare(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Load and prepare data for one-class classifier training.
        
        Args:
            train_dir: Directory with training data
            val_dir: Directory with validation data (optional)
            test_size: Fraction of training data to use for validation if val_dir not provided
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with keys:
                - X_train_clean: Clean training samples (normalized)
                - X_val_clean: Clean validation samples
                - X_val_attack: Attack validation samples
                - y_val: Labels for validation (0=clean, 1=attack)
                - feature_names: List of feature names
        """
        # Load training data
        train_df = self.load_dataset(train_dir)
        
        # Extract clean samples for training
        train_clean_df = self.prepare_one_class_data(train_df)
        
        # Get feature matrix and fit scaler
        X_train_clean, feature_names = self.get_feature_matrix(train_clean_df, fit_scaler=True)
        
        # Load or split validation data
        if val_dir is not None:
            # Load separate validation set
            val_df = self.load_dataset(val_dir)
            val_clean_df, val_attack_df = self.prepare_test_data(val_df)
            
            X_val_clean, _ = self.get_feature_matrix(val_clean_df, fit_scaler=False)
            X_val_attack, _ = self.get_feature_matrix(val_attack_df, fit_scaler=False)
        else:
            # Split training data
            X_train_clean, X_val_clean = train_test_split(
                X_train_clean,
                test_size=test_size,
                random_state=random_state
            )
            
            # Get attack samples from training set for validation
            train_attack_df = train_df[train_df['is_attack_active'] == 1]
            if len(train_attack_df) > 0:
                X_val_attack, _ = self.get_feature_matrix(train_attack_df, fit_scaler=False)
            else:
                X_val_attack = np.array([])
                print("Warning: No attack samples found in training data")
                
        # Create validation labels
        y_val_clean = np.zeros(len(X_val_clean))
        y_val_attack = np.ones(len(X_val_attack))
        y_val = np.concatenate([y_val_clean, y_val_attack])
        
        # Combine validation data
        X_val = np.vstack([X_val_clean, X_val_attack]) if len(X_val_attack) > 0 else X_val_clean
        
        print(f"\nData preparation complete:")
        print(f"  Training samples (clean only): {len(X_train_clean)}")
        print(f"  Validation samples (clean): {len(X_val_clean)}")
        print(f"  Validation samples (attack): {len(X_val_attack)}")
        print(f"  Total validation: {len(X_val)}")
        print(f"  Features: {len(feature_names)}")
        
        return {
            'X_train_clean': X_train_clean,
            'X_val': X_val,
            'X_val_clean': X_val_clean,
            'X_val_attack': X_val_attack,
            'y_val': y_val,
            'feature_names': feature_names
        }
        
    def save_scaler(self, filepath: str):
        """Save the fitted scaler to disk."""
        import joblib
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")
        
    def load_scaler(self, filepath: str):
        """Load a fitted scaler from disk."""
        import joblib
        self.scaler = joblib.load(filepath)
        print(f"Scaler loaded from {filepath}")


def quick_data_summary(directory: str):
    """
    Print a quick summary of the data in a directory.
    
    Args:
        directory: Path to directory containing CSV files
    """
    loader = DataLoader()
    try:
        df = loader.load_dataset(directory)
        
        print(f"\n{'='*60}")
        print(f"DATA SUMMARY: {directory}")
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")
        print(f"Clean samples: {len(df[df['is_attack_active']==0])} ({100*len(df[df['is_attack_active']==0])/len(df):.1f}%)")
        print(f"Attack samples: {len(df[df['is_attack_active']==1])} ({100*len(df[df['is_attack_active']==1])/len(df):.1f}%)")
        print(f"\nFeature statistics:")
        print(df[loader.PRIMARY_FEATURES].describe())
        print(f"{'='*60}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        quick_data_summary(sys.argv[1])
    else:
        print("Usage: python data_loader.py <data_directory>")
        print("Example: python data_loader.py data/training")

