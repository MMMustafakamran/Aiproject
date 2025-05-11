#!/usr/bin/env python3
"""
train_models.py

1) Load racing log CSVs
2) Basic cleaning (dedupe, drop NaNs, filter outliers)
3) Convert continuous controls to discrete actions
4) Extract features/targets
5) Split into train/test and scale features
6) Train action classification model
7) Evaluate on test set
8) Save scaler + trained model
"""

import glob
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm
import multiprocessing

# ----- CONFIG -----
DATA_DIRS   = ["results/manual"]  # List of directories to load data from
MODEL_DIR   = "trained_models/Model-01"
TEST_SIZE   = 0.2
RANDOM_SEED = 42
BATCH_SIZE  = 10000  # Process data in batches
N_JOBS      = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core

# Define which columns to use
FEATURE_COLS = [
    "SpeedX", "SpeedY", "SpeedZ",
    "TrackPos", "Angle", "RPM",
    "CurLapTime", "DistFromStart", "DistRaced",
    "Fuel", "Damage"
]

# Action thresholds for converting continuous controls to discrete actions
STEER_THRESHOLD = 0.1
ACCEL_THRESHOLD = 0.1
BRAKE_THRESHOLD = 0.1
GEAR_THRESHOLD = 0.5

def convert_to_action(row):
    """Convert continuous controls to discrete action."""
    steer = row['Steer']
    accel = row['Accel']
    brake = row['Brake']
    gear = row['Gear_Control']
    prev_gear = row.get('Prev_Gear', 1)  # Default to 1 if not available
    
    # Determine steering action
    if steer < -STEER_THRESHOLD:
        steer_action = "left"
    elif steer > STEER_THRESHOLD:
        steer_action = "right"
    else:
        steer_action = "straight"
    
    # Determine acceleration/braking action
    if brake > BRAKE_THRESHOLD:
        accel_action = "brake"
    elif accel > ACCEL_THRESHOLD:
        accel_action = "throttle"
    else:
        accel_action = "coast"
    
    # Determine gear action
    if gear > prev_gear:
        gear_action = "gear_up"
    elif gear < prev_gear:
        gear_action = "gear_down"
    else:
        gear_action = "maintain_gear"
    
    # Combine actions (prioritize steering and acceleration over gear changes)
    if accel_action == "brake":
        return "brake"
    elif steer_action != "straight":
        return steer_action
    elif accel_action == "throttle":
        return "throttle"
    elif gear_action != "maintain_gear":
        return gear_action
    else:
        return "coast"

def load_csv_batch(file_path):
    """Load and clean a single CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Basic cleaning
        df.drop_duplicates(inplace=True)
        df.dropna(subset=FEATURE_COLS + ['Steer', 'Accel', 'Brake', 'Gear_Control'], inplace=True)
        # Filter outliers
        df = df[
            (df["SpeedX"] >= 0) & (df["SpeedX"] <= 400) &
            (df["DistFromStart"] >= 0) &
            (df["DistRaced"] >= 0)
        ]
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load all CSVs from all directories using parallel processing
    all_csv_files = []
    for data_dir in DATA_DIRS:
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {data_dir}")
            all_csv_files.extend(csv_files)
    
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIRS}")
    
    print(f"\nLoading {len(all_csv_files)} total CSV files in parallel using {N_JOBS} workers...")
    dfs = []
    with ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        future_to_file = {executor.submit(load_csv_batch, f): f for f in all_csv_files}
        for future in tqdm(as_completed(future_to_file), total=len(all_csv_files), desc="Loading files"):
            df = future.result()
            if df is not None:
                dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Initial dataset size: {len(df):,} rows")

    # Add previous gear column for gear change detection
    df['Prev_Gear'] = df['Gear_Control'].shift(1).fillna(1)

    # Convert continuous controls to discrete actions
    print("\nConverting continuous controls to discrete actions...")
    df['action'] = df.apply(convert_to_action, axis=1)
    
    # Create action mapping
    unique_actions = df['action'].unique()
    action_mapping = {action: idx for idx, action in enumerate(unique_actions)}
    print("\nAction mapping:")
    for action, idx in action_mapping.items():
        print(f"  {action}: {idx}")

    # Extract features & targets
    print("\nPreparing features and targets:")
    X = df[FEATURE_COLS].values
    y = df['action'].map(action_mapping).values
    print(f"  - Features shape: {X.shape}")
    print(f"  - Target shape: {y.shape}")

    # Train/test split
    print(f"\nSplitting into train/test (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    print(f"  - Training set size: {len(X_train):,} samples")
    print(f"  - Test set size: {len(X_test):,} samples")

    # Fit scaler on train features
    print("\nFitting StandardScaler on training features...")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train action classification model
    print("\nTraining RandomForestClassifier for action classification...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    print("\nEvaluating model:")
    train_acc = model.score(X_train_s, y_train)
    test_acc  = model.score(X_test_s, y_test)
    print(f"  - Training accuracy: {train_acc:.3%}")
    print(f"  - Test accuracy:     {test_acc:.3%}")

    # Print feature importances
    print("\nFeature importances:")
    importances = model.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
        print(f"  - {feat}: {imp:.3f}")

    # Save models and metadata
    print("\nSaving model, scaler, and metadata...")
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(model, os.path.join(MODEL_DIR, "action_model.joblib"))
    joblib.dump(action_mapping, os.path.join(MODEL_DIR, "action_mapping.joblib"))
    print("✓ Models saved successfully")

    print("\nData preprocessing and model training complete.")

if __name__ == "__main__":
    main()
