#!/usr/bin/env python3
"""
data_preprocessing.py

1) Load racing log CSVs
2) Basic cleaning (dedupe, drop NaNs, filter outliers)
3) Extract features/targets
4) Split into train/test and scale features
5) Train continuous and discrete control models
6) Evaluate on test set
7) Save scaler + trained models
"""

import glob
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
import joblib
from tqdm import tqdm
import multiprocessing

# ----- CONFIG -----
DATA_DIRS   = ["results/manual", "results/ruleai"]  # List of directories to load data from
MODEL_DIR   = "trained_models/Model-01"
TEST_SIZE   = 0.2
RANDOM_SEED = 42
BATCH_SIZE  = 10000  # Process data in batches
N_JOBS      = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core

# 1) Define which columns to use
FEATURE_COLS = [
    "SpeedX", "SpeedY", "SpeedZ",
    "TrackPos", "Angle", "RPM",
    "CurLapTime", "DistFromStart", "DistRaced",
    "Fuel", "Damage"
    # if you logged track sensors, add them here:
    # "TrackSensor0", ..., "TrackSensor18"
]
CONT_TARGETS = ["Accel", "Brake", "Steer", "Clutch"]
DISC_TARGET  = "Gear_Control"

def load_csv_batch(file_path):
    """Load and clean a single CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Basic cleaning
        df.drop_duplicates(inplace=True)
        df.dropna(subset=FEATURE_COLS + CONT_TARGETS + [DISC_TARGET], inplace=True)
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

    # 2) Load all CSVs from all directories using parallel processing
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

    # 3) Extract features & targets
    print("\nPreparing features and targets:")
    X = df[FEATURE_COLS].values
    y_cont = df[CONT_TARGETS].values
    y_disc = df[DISC_TARGET].values.astype(int)
    print(f"  - Features shape: {X.shape}")
    print(f"  - Continuous targets shape: {y_cont.shape}")
    print(f"  - Discrete target shape: {y_disc.shape}")

    # 4) Train/test split
    print(f"\nSplitting into train/test (test_size={TEST_SIZE})...")
    X_train, X_test, \
    y_cont_train, y_cont_test, \
    y_disc_train, y_disc_test = train_test_split(
        X, y_cont, y_disc,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )
    print(f"  - Training set size: {len(X_train):,} samples")
    print(f"  - Test set size: {len(X_test):,} samples")

    # 5) Fit scaler on train features
    print("\nFitting StandardScaler on training features...")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 6) Train continuous-output model (MLP) with optimized parameters
    print("\nTraining MultiOutput MLPRegressor for continuous controls...")
    mlp = MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=(128, 64),  # Increased first layer size
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.1,
            n_iter_no_change=10,
            batch_size='auto',
            random_state=RANDOM_SEED
        ),
        n_jobs=N_JOBS  # Enable parallel processing
    )
    mlp.fit(X_train_s, y_cont_train)

    # 7) Train discrete-output model (Random Forest) with optimized parameters
    print("\nTraining RandomForestClassifier for gear control...")
    rfc = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,  # Limit tree depth
        min_samples_split=10,  # Increase minimum samples for split
        min_samples_leaf=5,    # Increase minimum samples for leaf
        n_jobs=N_JOBS,         # Enable parallel processing
        random_state=RANDOM_SEED
    )
    rfc.fit(X_train_s, y_disc_train)

    # 8) Evaluate
    print("\nEvaluating models:")
    cont_score = mlp.score(X_test_s, y_cont_test)
    disc_acc   = rfc.score(X_test_s, y_disc_test)
    print(f"  - Continuous control R² score on test: {cont_score:.3f}")
    print(f"  - Gear-control accuracy on test:    {disc_acc:.3%}")

    # 9) Save scaler + models
    print("\nSaving models and scaler...")
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(mlp,    os.path.join(MODEL_DIR, "cont_model.joblib"))
    joblib.dump(rfc,    os.path.join(MODEL_DIR, "gear_model.joblib"))
    print("✓ Models saved successfully")

    print("\nData preprocessing and model training complete.")

if __name__ == "__main__":
    main()
