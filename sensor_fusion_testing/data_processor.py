import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_dir: str = "test_results"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all test data from JSON files."""
        data_dict = {}
        
        for strategy_dir in os.listdir(self.data_dir):
            strategy_path = os.path.join(self.data_dir, strategy_dir)
            if not os.path.isdir(strategy_path):
                continue
                
            # Load raw data
            try:
                with open(os.path.join(strategy_path, 'raw_data.json'), 'r') as f:
                    raw_data = json.load(f)
                    
                # Validate data lengths
                required_fields = [
                    'timestamps', 'true_positions', 'fused_positions',
                    'true_velocities', 'fused_velocities',
                    'position_errors', 'velocity_errors'
                ]
                
                # Check if all required fields exist
                for field in required_fields:
                    if field not in raw_data:
                        print(f"Warning: Missing field '{field}' in {strategy_dir}")
                        continue
                        
                # Get the minimum length of all arrays
                lengths = {
                    'timestamps': len(raw_data['timestamps']),
                    'true_positions': len(raw_data['true_positions']),
                    'fused_positions': len(raw_data['fused_positions']),
                    'true_velocities': len(raw_data['true_velocities']),
                    'fused_velocities': len(raw_data['fused_velocities']),
                    'position_errors': len(raw_data['position_errors']),
                    'velocity_errors': len(raw_data['velocity_errors'])
                }
                
                min_length = min(lengths.values())
                if min_length == 0:
                    print(f"Warning: No data found for {strategy_dir}")
                    continue
                    
                # Print length information
                print(f"\nData lengths for {strategy_dir}:")
                for field, length in lengths.items():
                    print(f"  {field}: {length}")
                print(f"  Using minimum length: {min_length}")
                
                # Truncate all arrays to the minimum length
                df = pd.DataFrame({
                    'timestamp': raw_data['timestamps'][:min_length],
                    'true_x': [pos[0] for pos in raw_data['true_positions'][:min_length]],
                    'true_y': [pos[1] for pos in raw_data['true_positions'][:min_length]],
                    'true_z': [pos[2] for pos in raw_data['true_positions'][:min_length]],
                    'fused_x': [pos[0] for pos in raw_data['fused_positions'][:min_length]],
                    'fused_y': [pos[1] for pos in raw_data['fused_positions'][:min_length]],
                    'fused_z': [pos[2] for pos in raw_data['fused_positions'][:min_length]],
                    'true_vel_x': [vel[0] for vel in raw_data['true_velocities'][:min_length]],
                    'true_vel_y': [vel[1] for vel in raw_data['true_velocities'][:min_length]],
                    'true_vel_z': [vel[2] for vel in raw_data['true_velocities'][:min_length]],
                    'fused_vel_x': [vel[0] for vel in raw_data['fused_velocities'][:min_length]],
                    'fused_vel_y': [vel[1] for vel in raw_data['fused_velocities'][:min_length]],
                    'fused_vel_z': [vel[2] for vel in raw_data['fused_velocities'][:min_length]],
                    'position_error': raw_data['position_errors'][:min_length],
                    'velocity_error': raw_data['velocity_errors'][:min_length]
                })
                
                # Add derived features
                df['position_error_3d'] = np.sqrt(
                    (df['true_x'] - df['fused_x'])**2 +
                    (df['true_y'] - df['fused_y'])**2 +
                    (df['true_z'] - df['fused_z'])**2
                )
                
                df['velocity_error_3d'] = np.sqrt(
                    (df['true_vel_x'] - df['fused_vel_x'])**2 +
                    (df['true_vel_y'] - df['fused_vel_y'])**2 +
                    (df['true_vel_z'] - df['fused_vel_z'])**2
                )
                
                # Add rolling statistics with proper handling of NaN values
                window_size = min(10, min_length)  # Ensure window size is not larger than data
                
                # Calculate moving averages
                df['position_error_ma'] = df['position_error'].rolling(window=window_size, min_periods=1).mean()
                df['velocity_error_ma'] = df['velocity_error'].rolling(window=window_size, min_periods=1).mean()
                
                # Calculate standard deviations with proper handling of single values
                def safe_std(x):
                    if len(x) <= 1:
                        return 0.0
                    return x.std()
                    
                df['position_error_std'] = df['position_error'].rolling(window=window_size, min_periods=1).apply(safe_std)
                df['velocity_error_std'] = df['velocity_error'].rolling(window=window_size, min_periods=1).apply(safe_std)
                
                # Add attack label
                df['attack_type'] = strategy_dir
                
                data_dict[strategy_dir] = df
                
            except Exception as e:
                print(f"Error processing {strategy_dir}: {str(e)}")
                continue
                
        if not data_dict:
            raise ValueError("No valid data found in test_results directory")
            
        return data_dict
        
    def prepare_ml_data(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for ML training."""
        # Load all data
        data_dict = self.load_data()
        
        # Combine all data
        all_data = pd.concat(data_dict.values(), ignore_index=True)
        
        # Drop rows with NaN values (from rolling statistics)
        all_data = all_data.dropna()
        
        if len(all_data) == 0:
            raise ValueError("No valid data after processing")
            
        print(f"\nTotal samples after processing: {len(all_data)}")
        
        # Prepare features and labels
        feature_columns = [
            'position_error', 'velocity_error',
            'position_error_3d', 'velocity_error_3d',
            'position_error_ma', 'position_error_std',
            'velocity_error_ma', 'velocity_error_std'
        ]
        
        X = all_data[feature_columns].values
        y = pd.get_dummies(all_data['attack_type']).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
        
    def save_processed_data(self, output_dir: str = "ml_data"):
        """Save processed data for ML training."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_ml_data()
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save scaler parameters
        scaler_params = {
            'mean_': self.scaler.mean_.tolist(),
            'scale_': self.scaler.scale_.tolist()
        }
        with open(os.path.join(output_dir, 'scaler_params.json'), 'w') as f:
            json.dump(scaler_params, f, indent=2)
            
        print(f"\nProcessed data saved to {output_dir}")
            
    def generate_feature_importance(self) -> pd.DataFrame:
        """Generate feature importance analysis."""
        data_dict = self.load_data()
        all_data = pd.concat(data_dict.values(), ignore_index=True)
        
        # Calculate correlation with position error
        correlations = {}
        for col in all_data.columns:
            if col not in ['timestamp', 'attack_type']:
                try:
                    corr = np.corrcoef(all_data[col], all_data['position_error'])[0, 1]
                    correlations[col] = corr
                except Exception as e:
                    print(f"Warning: Could not calculate correlation for {col}: {str(e)}")
                    correlations[col] = np.nan
                
        # Convert to DataFrame and sort by absolute correlation
        importance_df = pd.DataFrame({
            'feature': list(correlations.keys()),
            'correlation': list(correlations.values())
        })
        importance_df['abs_correlation'] = importance_df['correlation'].abs()
        importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        # Add feature descriptions
        feature_descriptions = {
            'position_error': 'Direct position error',
            'position_error_3d': '3D Euclidean position error',
            'velocity_error': 'Direct velocity error',
            'velocity_error_3d': '3D Euclidean velocity error',
            'fused_vel_x': 'Fused velocity X component',
            'fused_vel_y': 'Fused velocity Y component',
            'fused_vel_z': 'Fused velocity Z component',
            'fused_x': 'Fused position X component',
            'fused_y': 'Fused position Y component',
            'fused_z': 'Fused position Z component',
            'true_vel_x': 'True velocity X component',
            'true_vel_y': 'True velocity Y component',
            'true_vel_z': 'True velocity Z component',
            'true_x': 'True position X component',
            'true_y': 'True position Y component',
            'true_z': 'True position Z component',
            'position_error_ma': 'Moving average of position error',
            'position_error_std': 'Standard deviation of position error',
            'velocity_error_ma': 'Moving average of velocity error',
            'velocity_error_std': 'Standard deviation of velocity error'
        }
        
        importance_df['description'] = importance_df['feature'].map(feature_descriptions)
        
        # Add statistical analysis
        print("\nStatistical Analysis by Attack Type:")
        print("===================================")
        for strategy in data_dict.keys():
            print(f"\n{strategy}:")
            strategy_data = data_dict[strategy]
            print(f"  Position Error:")
            print(f"    Mean: {strategy_data['position_error'].mean():.3f}")
            print(f"    Std: {strategy_data['position_error'].std():.3f}")
            print(f"    Max: {strategy_data['position_error'].max():.3f}")
            print(f"    Min: {strategy_data['position_error'].min():.3f}")
            print(f"  Velocity Error:")
            print(f"    Mean: {strategy_data['velocity_error'].mean():.3f}")
            print(f"    Std: {strategy_data['velocity_error'].std():.3f}")
            print(f"    Max: {strategy_data['velocity_error'].max():.3f}")
            print(f"    Min: {strategy_data['velocity_error'].min():.3f}")
        
        return importance_df

def main():
    # Example usage
    processor = DataProcessor()
    
    # Process and save data
    processor.save_processed_data()
    
    # Generate feature importance analysis
    importance_df = processor.generate_feature_importance()
    print("\nFeature Importance Analysis:")
    print(importance_df)
    
    # Save feature importance
    importance_df.to_csv('ml_data/feature_importance.csv', index=False)
    print("\nFeature importance saved to ml_data/feature_importance.csv")

if __name__ == '__main__':
    main() 