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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
import joblib

# ----- CONFIG -----
DATA_DIRS   = ["results/manual", "results/ruleai"]  # List of directories to load data from
MODEL_DIR   = "trained_models/Model-01"
TEST_SIZE   = 0.2
RANDOM_SEED = 42

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

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2) Load all CSVs from all directories
    all_csv_files = []
    for data_dir in DATA_DIRS:
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {data_dir}")
            all_csv_files.extend(csv_files)
    
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIRS}")
    
    print(f"\nLoading {len(all_csv_files)} total CSV files...")
    df = pd.concat((pd.read_csv(f) for f in all_csv_files), ignore_index=True)
    print(f"Initial dataset size: {len(df):,} rows")

    # 3) Basic cleaning
    print("\nCleaning data:")
    print(f"  - Before cleaning: {len(df):,} rows")
    
    # a) drop exact duplicates
    df.drop_duplicates(inplace=True)
    print(f"  - After removing duplicates: {len(df):,} rows")

    # b) drop rows with NaNs in any feature or target
    required = FEATURE_COLS + CONT_TARGETS + [DISC_TARGET]
    nan_before = df[required].isna().any(axis=1).sum()
    df.dropna(subset=required, inplace=True)
    print(f"  - Removed {nan_before:,} rows with missing values")
    print(f"  - After removing NaNs: {len(df):,} rows")

    # c) filter out obvious outliers
    outliers = df[
        ~((df["SpeedX"] >= 0) & (df["SpeedX"] <= 400) &   # reasonable speed range
          (df["DistFromStart"] >= 0) &
          (df["DistRaced"] >= 0))
    ]
    df = df[
        (df["SpeedX"] >= 0) & (df["SpeedX"] <= 400) &   # reasonable speed range
        (df["DistFromStart"] >= 0) &
        (df["DistRaced"] >= 0)
    ]
    print(f"  - Removed {len(outliers):,} rows with invalid values")
    print(f"  - Final cleaned dataset: {len(df):,} rows")

    # 4) Extract features & targets
    print("\nPreparing features and targets:")
    X = df[FEATURE_COLS].values
    y_cont = df[CONT_TARGETS].values
    y_disc = df[DISC_TARGET].values.astype(int)
    print(f"  - Features shape: {X.shape}")
    print(f"  - Continuous targets shape: {y_cont.shape}")
    print(f"  - Discrete target shape: {y_disc.shape}")

    # 5) Train/test split
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

    # 6) Fit scaler on train features
    print("\nFitting StandardScaler on training features...")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 7) Train continuous-output model (MLP)
    print("\nTraining MultiOutput MLPRegressor for continuous controls...")
    mlp = MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=RANDOM_SEED
        )
    )
    mlp.fit(X_train_s, y_cont_train)

    # 8) Train discrete-output model (Random Forest)
    print("\nTraining RandomForestClassifier for gear control...")
    rfc = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED
    )
    rfc.fit(X_train_s, y_disc_train)

    # 9) Evaluate
    print("\nEvaluating models:")
    cont_score = mlp.score(X_test_s, y_cont_test)
    disc_acc   = rfc.score(X_test_s, y_disc_test)
    print(f"  - Continuous control R² score on test: {cont_score:.3f}")
    print(f"  - Gear-control accuracy on test:    {disc_acc:.3%}")

    # 10) Save scaler + models
    print("\nSaving models and scaler...")
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(mlp,    os.path.join(MODEL_DIR, "cont_model.joblib"))
    joblib.dump(rfc,    os.path.join(MODEL_DIR, "gear_model.joblib"))
    print("✓ Models saved successfully")

    print("\nData preprocessing and model training complete.")

if __name__ == "__main__":
    main()
