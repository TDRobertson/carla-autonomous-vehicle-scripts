#!/usr/bin/env python3
"""
Automated ML Dataset Collection Manager

This script provides a sophisticated, cross-platform interface for automating
data collection with progress tracking, error handling, and automatic CARLA
connection checking.

Features:
- Progress bars for visual feedback
- Automatic retry on collection failures
- CARLA connection health checks
- Dataset statistics after collection
- Cross-platform compatibility (Windows/Linux/Mac)

Usage:
    python automated_data_collection.py
"""

import sys
import os
import time
import glob
import subprocess
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import json

# Try to import optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Progress bars will be disabled.")
    print("Install with: pip install tqdm")

# CARLA imports for connection checking
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import carla
    HAS_CARLA = True
except ImportError:
    HAS_CARLA = False


class CollectionConfig:
    """Configuration for a collection run."""
    def __init__(self, name: str, num_runs: int, duration: int, 
                 attack_delay: int, output_dir: str, label_prefix: str,
                 random_attacks: bool = False, description: str = ""):
        self.name = name
        self.num_runs = num_runs
        self.duration = duration
        self.attack_delay = attack_delay
        self.output_dir = output_dir
        self.label_prefix = label_prefix
        self.random_attacks = random_attacks
        self.description = description


class DataCollectionManager:
    """Manager for automated data collection with progress tracking."""
    
    PRESETS = {
        'quick_test': CollectionConfig(
            name="Quick Test",
            num_runs=5,
            duration=60,
            attack_delay=10,
            output_dir="data/test",
            label_prefix="test_run",
            random_attacks=False,
            description="Quick 5-run test for validation"
        ),
        'one_class_training': CollectionConfig(
            name="One-Class Training",
            num_runs=25,
            duration=120,
            attack_delay=30,
            output_dir="data/training",
            label_prefix="train_run",
            random_attacks=False,
            description="Standard training dataset with clean baseline"
        ),
        'one_class_validation': CollectionConfig(
            name="One-Class Validation",
            num_runs=5,
            duration=180,
            attack_delay=10,
            output_dir="data/validation",
            label_prefix="val_run",
            random_attacks=True,
            description="Validation set with random attack timing"
        ),
        'supervised_training': CollectionConfig(
            name="Supervised Training",
            num_runs=20,
            duration=120,
            attack_delay=0,
            output_dir="data/supervised_training",
            label_prefix="sup_train_run",
            random_attacks=False,
            description="Balanced supervised learning dataset"
        ),
        'supervised_validation': CollectionConfig(
            name="Supervised Validation",
            num_runs=10,
            duration=150,
            attack_delay=0,
            output_dir="data/supervised_validation",
            label_prefix="sup_val_run",
            random_attacks=True,
            description="Supervised validation with timing diversity"
        ),
    }
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.failed_runs = []
        self.successful_runs = []
        
    def check_carla_connection(self) -> bool:
        """Check if CARLA server is accessible."""
        if not HAS_CARLA:
            print("Warning: CARLA Python API not found. Skipping connection check.")
            return True
            
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(5.0)
            world = client.get_world()
            print(f"✓ Connected to CARLA: {world.get_map().name}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to CARLA: {e}")
            print("  Make sure CarlaUE4.exe is running.")
            return False
            
    def run_collection(self, config: CollectionConfig, run_number: int) -> bool:
        """Run a single data collection."""
        label = f"{config.label_prefix}{run_number:02d}"
        
        # Build command
        cmd = [
            sys.executable,
            "ml_data_collector.py",
            "--duration", str(config.duration),
            "--attack-delay", str(config.attack_delay),
            "--warmup", "5",
            "--output-dir", config.output_dir,
            "--label", label
        ]
        
        if config.random_attacks:
            cmd.append("--random-attacks")
            
        # Run with retries
        for attempt in range(self.max_retries):
            try:
                print(f"  Starting run {run_number}/{config.num_runs} (attempt {attempt + 1}/{self.max_retries})...")
                result = subprocess.run(cmd, check=True, capture_output=False)
                
                self.successful_runs.append({
                    'run': run_number,
                    'label': label,
                    'config': config.name
                })
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Run {run_number} failed (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    print(f"  Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    self.failed_runs.append({
                        'run': run_number,
                        'label': label,
                        'config': config.name,
                        'error': str(e)
                    })
                    return False
                    
        return False
        
    def collect_dataset(self, config: CollectionConfig):
        """Collect a full dataset using the given configuration."""
        print("\n" + "="*80)
        print(f"COLLECTING: {config.name}")
        print("="*80)
        print(f"Description: {config.description}")
        print(f"Runs: {config.num_runs}")
        print(f"Duration: {config.duration}s per run")
        print(f"Attack delay: {config.attack_delay}s")
        print(f"Random attacks: {config.random_attacks}")
        print(f"Output: {config.output_dir}")
        print(f"Estimated time: ~{int(config.num_runs * (config.duration + 10) / 60)} minutes")
        print("="*80)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Confirm
        response = input("\nProceed with collection? (Y/N): ")
        if response.upper() != 'Y':
            print("Collection cancelled.")
            return
            
        # Check CARLA connection
        print("\nChecking CARLA connection...")
        if not self.check_carla_connection():
            response = input("CARLA not accessible. Continue anyway? (Y/N): ")
            if response.upper() != 'Y':
                print("Collection cancelled.")
                return
                
        print("\nStarting data collection...")
        start_time = time.time()
        
        # Progress bar or simple counter
        if HAS_TQDM:
            pbar = tqdm(range(1, config.num_runs + 1), desc="Progress", unit="run")
            for run_num in pbar:
                pbar.set_description(f"Run {run_num}/{config.num_runs}")
                success = self.run_collection(config, run_num)
                if not success:
                    pbar.write(f"  ✗ Run {run_num} failed after {self.max_retries} attempts")
                else:
                    pbar.write(f"  ✓ Run {run_num} completed successfully")
                time.sleep(1)  # Brief pause between runs
        else:
            for run_num in range(1, config.num_runs + 1):
                print(f"\n[{run_num}/{config.num_runs}] Collecting...")
                success = self.run_collection(config, run_num)
                if not success:
                    print(f"  ✗ Run {run_num} failed after {self.max_retries} attempts")
                else:
                    print(f"  ✓ Run {run_num} completed successfully")
                time.sleep(1)
                
        elapsed_time = time.time() - start_time
        
        # Summary
        print("\n" + "="*80)
        print("COLLECTION COMPLETE")
        print("="*80)
        print(f"Total runs: {config.num_runs}")
        print(f"Successful: {len(self.successful_runs)}")
        print(f"Failed: {len(self.failed_runs)}")
        print(f"Time elapsed: {int(elapsed_time / 60)} minutes {int(elapsed_time % 60)} seconds")
        print(f"Output location: {config.output_dir}/")
        
        # Show failed runs if any
        if self.failed_runs:
            print("\nFailed runs:")
            for fail in self.failed_runs:
                print(f"  - Run {fail['run']} ({fail['label']})")
                
        # Generate statistics
        self.print_dataset_statistics(config.output_dir)
        
    def print_dataset_statistics(self, output_dir: str):
        """Print statistics about collected dataset."""
        try:
            import pandas as pd
            
            csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
            if not csv_files:
                print("\nNo CSV files found for statistics.")
                return
                
            print("\n" + "="*80)
            print("DATASET STATISTICS")
            print("="*80)
            
            total_samples = 0
            clean_samples = 0
            attack_samples = 0
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    total_samples += len(df)
                    clean_samples += len(df[df['is_attack_active'] == 0])
                    attack_samples += len(df[df['is_attack_active'] == 1])
                except Exception as e:
                    print(f"Warning: Could not read {csv_file}: {e}")
                    
            print(f"Total files: {len(csv_files)}")
            print(f"Total samples: {total_samples:,}")
            print(f"Clean samples: {clean_samples:,} ({100*clean_samples/max(1,total_samples):.1f}%)")
            print(f"Attack samples: {attack_samples:,} ({100*attack_samples/max(1,total_samples):.1f}%)")
            print("="*80)
            
        except ImportError:
            print("\nNote: Install pandas to see dataset statistics (pip install pandas)")


def show_menu():
    """Show interactive menu."""
    print("\n" + "="*80)
    print("AUTOMATED ML DATA COLLECTION MANAGER")
    print("="*80)
    print("\nPreset Collection Modes:")
    print("  1. Quick Test (5 runs × 60s)")
    print("  2. One-Class Training (25 runs × 120s)")
    print("  3. One-Class Validation (5 runs × 180s, random)")
    print("  4. Supervised Training (20 runs × 120s)")
    print("  5. Supervised Validation (10 runs × 150s, random)")
    print("  6. Custom Configuration")
    print("  7. Exit")
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated ML Dataset Collection Manager"
    )
    parser.add_argument(
        '--preset', type=str, choices=list(DataCollectionManager.PRESETS.keys()),
        help='Run a preset configuration non-interactively'
    )
    parser.add_argument(
        '--max-retries', type=int, default=3,
        help='Maximum retries per run (default: 3)'
    )
    
    args = parser.parse_args()
    
    manager = DataCollectionManager(max_retries=args.max_retries)
    
    # Non-interactive mode
    if args.preset:
        config = DataCollectionManager.PRESETS[args.preset]
        manager.collect_dataset(config)
        return
        
    # Interactive mode
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            manager.collect_dataset(DataCollectionManager.PRESETS['quick_test'])
        elif choice == '2':
            manager.collect_dataset(DataCollectionManager.PRESETS['one_class_training'])
        elif choice == '3':
            manager.collect_dataset(DataCollectionManager.PRESETS['one_class_validation'])
        elif choice == '4':
            manager.collect_dataset(DataCollectionManager.PRESETS['supervised_training'])
        elif choice == '5':
            manager.collect_dataset(DataCollectionManager.PRESETS['supervised_validation'])
        elif choice == '6':
            # Custom configuration
            print("\n" + "="*80)
            print("CUSTOM CONFIGURATION")
            print("="*80)
            try:
                num_runs = int(input("Number of runs: "))
                duration = int(input("Duration per run (seconds): "))
                attack_delay = int(input("Attack delay (seconds): "))
                output_dir = input("Output directory: ")
                label_prefix = input("Label prefix: ")
                random = input("Use random attacks? (Y/N): ").upper() == 'Y'
                
                config = CollectionConfig(
                    name="Custom",
                    num_runs=num_runs,
                    duration=duration,
                    attack_delay=attack_delay,
                    output_dir=output_dir,
                    label_prefix=label_prefix,
                    random_attacks=random,
                    description="Custom user configuration"
                )
                manager.collect_dataset(config)
            except ValueError as e:
                print(f"Invalid input: {e}")
        elif choice == '7':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

