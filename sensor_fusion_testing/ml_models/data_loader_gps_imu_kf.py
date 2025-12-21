"""
Data Loader for GPS+IMU+KF-only Spoofing Detection

This module defines a feature set that uses ONLY victim-side sensor data:
- GPS position (from GNSS sensor, converted to local ENU meters)
- IMU acceleration and gyroscope
- Kalman filter state (position, velocity)
- Innovation statistics (vector, norm, NIS)
- Covariance diagnostics (S diag, P diag)

NO ground-truth vehicle position is used. This makes the detector independent
of any "golden" reference dataset.

The features are designed for unsupervised anomaly detection where the model
trains only on clean data and detects attacks as anomalies.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional


# Primary features for GPS+IMU+KF-only anomaly detection
# These are all computed from victim-side sensors without ground truth
PRIMARY_FEATURES_GPS_IMU_KF = [
    # GPS position (local ENU meters from GNSS sensor)
    'gps_x',
    'gps_y',
    'gps_z',
    
    # Kalman filter updated state
    'kf_x',
    'kf_y', 
    'kf_z',
    'kf_vx',
    'kf_vy',
    'kf_vz',
    
    # Innovation (measurement residual)
    'innov_x',
    'innov_y',
    'innov_z',
    'innov_norm',
    
    # Normalized Innovation Squared (chi-squared statistic)
    'nis',
    
    # Innovation covariance diagonal (uncertainty in innovation)
    'S_x',
    'S_y',
    'S_z',
    
    # IMU data
    'accel_magnitude',
    'gyro_magnitude',
    
    # Rolling statistics on innovation (computed from buffer)
    'innov_norm_ma',
    'innov_norm_std',
    'nis_ma',
    'nis_std',
]


class DataLoaderGPSIMUKF:
    """
    Load and preprocess GPS+IMU+KF-only data for ML training.
    
    This loader handles data that contains only victim-side features:
    - No true_gps_* columns (no ground truth position)
    - No position_error or kf_tracking_error (no comparison to truth)
    
    The data is expected to come from clean runs for training, and the
    model detects attacks as anomalies in the feature distribution.
    """
    
    def __init__(self, feature_list: Optional[List[str]] = None):
        """
        Initialize the data loader.
        
        Args:
            feature_list: List of feature names to use. If None, uses PRIMARY_FEATURES_GPS_IMU_KF.
        """
        self.feature_list = feature_list if feature_list is not None else PRIMARY_FEATURES_GPS_IMU_KF
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
            
        print(f"[DataLoaderGPSIMUKF] Loading {len(csv_files)} CSV files from {directory}")
        
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add source file for tracking
                df['source_file'] = os.path.basename(csv_file)
                dataframes.append(df)
            except Exception as e:
                print(f"  Warning: Failed to load {csv_file}: {e}")
                
        if not dataframes:
            raise ValueError(f"No valid CSV files could be loaded from {directory}")
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"[DataLoaderGPSIMUKF] Loaded {len(combined_df)} total samples")
        
        return combined_df
        
    def prepare_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract clean (non-spoofed) samples for one-class training.
        
        Args:
            df: Full dataset
            
        Returns:
            DataFrame with only clean samples
        """
        if 'is_attack_active' in df.columns:
            clean_df = df[df['is_attack_active'] == 0].copy()
            print(f"[DataLoaderGPSIMUKF] Extracted {len(clean_df)} clean samples for training")
        else:
            # If no label column, assume all data is clean
            clean_df = df.copy()
            print(f"[DataLoaderGPSIMUKF] No 'is_attack_active' column - using all {len(clean_df)} samples as clean")
        return clean_df
        
    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare test data with both clean and attack samples.
        
        Args:
            df: Full dataset
            
        Returns:
            Tuple of (clean_df, attack_df)
        """
        if 'is_attack_active' not in df.columns:
            print("[DataLoaderGPSIMUKF] Warning: No 'is_attack_active' column - cannot separate clean/attack")
            return df.copy(), pd.DataFrame()
            
        clean_df = df[df['is_attack_active'] == 0].copy()
        attack_df = df[df['is_attack_active'] == 1].copy()
        
        print(f"[DataLoaderGPSIMUKF] Test data: {len(clean_df)} clean, {len(attack_df)} attack samples")
        
        return clean_df, attack_df
        
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
                           f"Available columns: {list(df.columns)}")
                           
        if len(available_features) < len(self.feature_list):
            missing = set(self.feature_list) - set(available_features)
            print(f"[DataLoaderGPSIMUKF] Warning: Missing features: {missing}")
            
        print(f"[DataLoaderGPSIMUKF] Using {len(available_features)} features")
        
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
            train_dir: Directory with training data (clean runs)
            val_dir: Directory with validation data (optional, may have attacks)
            test_size: Fraction of training data to use for validation if val_dir not provided
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with keys:
                - X_train_clean: Clean training samples (normalized)
                - X_val_clean: Clean validation samples
                - X_val_attack: Attack validation samples (if available)
                - y_val: Labels for validation (0=clean, 1=attack)
                - feature_names: List of feature names
        """
        # Load training data
        train_df = self.load_dataset(train_dir)
        
        # Extract clean samples for training
        train_clean_df = self.prepare_clean_data(train_df)
        
        # Get feature matrix and fit scaler
        X_train_clean, feature_names = self.get_feature_matrix(train_clean_df, fit_scaler=True)
        
        # Load or split validation data
        if val_dir is not None and os.path.exists(val_dir):
            # Load separate validation set
            val_df = self.load_dataset(val_dir)
            val_clean_df, val_attack_df = self.prepare_test_data(val_df)
            
            X_val_clean = np.array([])
            X_val_attack = np.array([])
            
            if len(val_clean_df) > 0:
                X_val_clean, _ = self.get_feature_matrix(val_clean_df, fit_scaler=False)
            if len(val_attack_df) > 0:
                X_val_attack, _ = self.get_feature_matrix(val_attack_df, fit_scaler=False)
        else:
            # Split training data
            X_train_clean, X_val_clean = train_test_split(
                X_train_clean,
                test_size=test_size,
                random_state=random_state
            )
            
            # No attack samples available from training set alone
            X_val_attack = np.array([])
            print("[DataLoaderGPSIMUKF] No validation directory - using split of training data")
                
        # Create validation labels
        y_val_clean = np.zeros(len(X_val_clean)) if len(X_val_clean) > 0 else np.array([])
        y_val_attack = np.ones(len(X_val_attack)) if len(X_val_attack) > 0 else np.array([])
        y_val = np.concatenate([y_val_clean, y_val_attack])
        
        # Combine validation data
        if len(X_val_attack) > 0 and len(X_val_clean) > 0:
            X_val = np.vstack([X_val_clean, X_val_attack])
        elif len(X_val_clean) > 0:
            X_val = X_val_clean
        elif len(X_val_attack) > 0:
            X_val = X_val_attack
        else:
            X_val = np.array([])
        
        print(f"\n[DataLoaderGPSIMUKF] Data preparation complete:")
        print(f"  Training samples (clean only): {len(X_train_clean)}")
        print(f"  Validation samples (clean): {len(X_val_clean) if len(X_val_clean) > 0 else 0}")
        print(f"  Validation samples (attack): {len(X_val_attack) if len(X_val_attack) > 0 else 0}")
        print(f"  Total validation: {len(X_val) if len(X_val) > 0 else 0}")
        print(f"  Features: {len(feature_names)}")
        
        return {
            'X_train_clean': X_train_clean,
            'X_val': X_val if len(X_val) > 0 else np.array([]),
            'X_val_clean': X_val_clean if len(X_val_clean) > 0 else np.array([]),
            'X_val_attack': X_val_attack if len(X_val_attack) > 0 else np.array([]),
            'y_val': y_val,
            'feature_names': feature_names
        }
        
    def save_scaler(self, filepath: str):
        """Save the fitted scaler to disk."""
        import joblib
        joblib.dump(self.scaler, filepath)
        print(f"[DataLoaderGPSIMUKF] Scaler saved to {filepath}")
        
    def load_scaler(self, filepath: str):
        """Load a fitted scaler from disk."""
        import joblib
        self.scaler = joblib.load(filepath)
        print(f"[DataLoaderGPSIMUKF] Scaler loaded from {filepath}")


def get_feature_description() -> Dict[str, str]:
    """
    Get descriptions of all features in PRIMARY_FEATURES_GPS_IMU_KF.
    
    Returns:
        Dictionary mapping feature name to description
    """
    return {
        'gps_x': 'GPS East position in local ENU meters (from GNSS sensor)',
        'gps_y': 'GPS North position in local ENU meters (from GNSS sensor)',
        'gps_z': 'GPS Up position in local ENU meters (from GNSS sensor)',
        'kf_x': 'Kalman filter estimated East position (meters)',
        'kf_y': 'Kalman filter estimated North position (meters)',
        'kf_z': 'Kalman filter estimated Up position (meters)',
        'kf_vx': 'Kalman filter estimated East velocity (m/s)',
        'kf_vy': 'Kalman filter estimated North velocity (m/s)',
        'kf_vz': 'Kalman filter estimated Up velocity (m/s)',
        'innov_x': 'Innovation (residual) East component (meters)',
        'innov_y': 'Innovation (residual) North component (meters)',
        'innov_z': 'Innovation (residual) Up component (meters)',
        'innov_norm': 'Innovation Euclidean norm (meters)',
        'nis': 'Normalized Innovation Squared (chi-squared with 3 DOF)',
        'S_x': 'Innovation covariance diagonal - East (m^2)',
        'S_y': 'Innovation covariance diagonal - North (m^2)',
        'S_z': 'Innovation covariance diagonal - Up (m^2)',
        'accel_magnitude': 'IMU accelerometer magnitude (m/s^2)',
        'gyro_magnitude': 'IMU gyroscope magnitude (rad/s)',
        'innov_norm_ma': 'Rolling mean of innovation norm (window=10)',
        'innov_norm_std': 'Rolling std of innovation norm (window=10)',
        'nis_ma': 'Rolling mean of NIS (window=10)',
        'nis_std': 'Rolling std of NIS (window=10)',
    }


if __name__ == "__main__":
    print("GPS+IMU+KF-only Feature Set")
    print("=" * 60)
    print(f"Number of features: {len(PRIMARY_FEATURES_GPS_IMU_KF)}")
    print("\nFeatures:")
    descriptions = get_feature_description()
    for i, feat in enumerate(PRIMARY_FEATURES_GPS_IMU_KF, 1):
        desc = descriptions.get(feat, "No description")
        print(f"  {i:2d}. {feat:20s} - {desc}")

