#!/usr/bin/env python3
"""
train_models.py

1) Load racing log CSVs
2) Basic cleaning (dedupe, drop NaNs, filter outliers)
3) Data inspection and EDA
4) Convert continuous controls to discrete actions
5) Extract features/targets
6) Feature engineering (PCA, interactions, polynomials)
7) Split into train/test and scale features
8) Train action classification model
9) Evaluate on test set
10) Save scaler + trained model
"""

import glob
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ----- CONFIG -----
DATA_DIRS   = ["results/manual"]  # List of directories to load data from
MODEL_DIR   = "trained_models/Model-01"
TEST_SIZE   = 0.2
RANDOM_SEED = 42
BATCH_SIZE  = 10000  # Process data in batches
N_JOBS      = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core

# Feature engineering config
USE_PCA = True
N_COMPONENTS = 0.95  # Keep 95% of variance
USE_POLY_FEATURES = True
POLY_DEGREE = 2
USE_INTERACTIONS = True

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

def analyze_data_quality(df):
    """Analyze data quality and print summary statistics."""
    print("\n=== Data Quality Analysis ===")
    
    # Basic info
    print("\nDataset Info:")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns):,}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print("\nMissing Values:")
    for col, pct in zip(missing.index, missing_pct):
        if pct > 0:
            print(f"  {col}: {pct:.2f}%")
    
    # Duplicates
    n_dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dupes:,} ({n_dupes/len(df)*100:.2f}%)")
    
    # Value ranges
    print("\nValue Ranges:")
    for col in FEATURE_COLS:
        if col in df.columns:
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")

def analyze_distributions(df):
    """Analyze feature distributions and save plots."""
    print("\n=== Distribution Analysis ===")
    
    # Create distribution plots directory
    dist_dir = os.path.join(MODEL_DIR, "distributions")
    os.makedirs(dist_dir, exist_ok=True)
    
    # Analyze each feature
    for col in FEATURE_COLS:
        if col in df.columns:
            # Calculate statistics
            stats_dict = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            
            print(f"\n{col} Distribution:")
            for stat, value in stats_dict.items():
                print(f"  {stat}: {value:.3f}")
            
            # Create and save distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(dist_dir, f'{col}_distribution.png'))
            plt.close()

def analyze_correlations(df):
    """Analyze feature correlations and save correlation matrix."""
    print("\n=== Correlation Analysis ===")
    
    # Calculate correlation matrix
    corr_matrix = df[FEATURE_COLS].corr()
    
    # Print strong correlations
    print("\nStrong Correlations (|r| > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(f"  {corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr:.3f}")
    
    # Save correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'correlation_matrix.png'))
    plt.close()

def analyze_action_distribution(df):
    """Analyze the distribution of actions and their relationships with features."""
    print("\n=== Action Distribution Analysis ===")
    
    # Action counts
    action_counts = df['action'].value_counts()
    print("\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"  {action}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Analyze feature distributions by action
    action_dir = os.path.join(MODEL_DIR, "action_analysis")
    os.makedirs(action_dir, exist_ok=True)
    
    for col in FEATURE_COLS:
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='action', y=col)
            plt.title(f'{col} Distribution by Action')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(action_dir, f'{col}_by_action.png'))
            plt.close()

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

def create_feature_engineering_pipeline():
    """Create a pipeline for feature engineering."""
    steps = []
    
    # Always start with scaling
    steps.append(('scaler', StandardScaler()))
    
    # Add polynomial features if enabled
    if USE_POLY_FEATURES:
        steps.append(('poly', PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)))
    
    # Add PCA if enabled
    if USE_PCA:
        steps.append(('pca', PCA(n_components=N_COMPONENTS)))
    
    return Pipeline(steps)

def add_interaction_features(df):
    """Add interaction features between relevant columns."""
    if not USE_INTERACTIONS:
        return df
    
    # Speed-based interactions
    df['Speed_Magnitude'] = np.sqrt(df['SpeedX']**2 + df['SpeedY']**2 + df['SpeedZ']**2)
    df['Speed_Angle_Ratio'] = df['Speed_Magnitude'] / (df['Angle'] + 1e-6)
    
    # Track position interactions
    df['TrackPos_Speed'] = df['TrackPos'] * df['Speed_Magnitude']
    df['TrackPos_Angle'] = df['TrackPos'] * df['Angle']
    
    # Time-based interactions
    df['LapTime_Speed'] = df['CurLapTime'] * df['Speed_Magnitude']
    df['Dist_Speed'] = df['DistRaced'] * df['Speed_Magnitude']
    
    return df

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

    # Perform data inspection and EDA
    analyze_data_quality(df)
    analyze_distributions(df)
    analyze_correlations(df)

    # Add previous gear column for gear change detection
    df['Prev_Gear'] = df['Gear_Control'].shift(1).fillna(1)

    # Convert continuous controls to discrete actions
    print("\nConverting continuous controls to discrete actions...")
    df['action'] = df.apply(convert_to_action, axis=1)

    # Analyze action distribution
    analyze_action_distribution(df)

    # Add interaction features
    print("\nAdding interaction features...")
    df = add_interaction_features(df)

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
    print(f"  - Initial features shape: {X.shape}")
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

    # Create and fit feature engineering pipeline
    print("\nCreating feature engineering pipeline...")
    feature_pipeline = create_feature_engineering_pipeline()
    X_train_processed = feature_pipeline.fit_transform(X_train)
    X_test_processed = feature_pipeline.transform(X_test)
    print(f"  - Processed features shape: {X_train_processed.shape}")

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
    model.fit(X_train_processed, y_train)

    # Evaluate
    print("\nEvaluating model:")
    train_acc = model.score(X_train_processed, y_train)
    test_acc  = model.score(X_test_processed, y_test)
    print(f"  - Training accuracy: {train_acc:.3%}")
    print(f"  - Test accuracy:     {test_acc:.3%}")

    # Print feature importances
    print("\nFeature importances:")
    importances = model.feature_importances_
    feature_names = feature_pipeline.get_feature_names_out(FEATURE_COLS)
    for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {feat}: {imp:.3f}")

    # Save models and metadata
    print("\nSaving model, pipeline, and metadata...")
    joblib.dump(feature_pipeline, os.path.join(MODEL_DIR, "feature_pipeline.joblib"))
    joblib.dump(model, os.path.join(MODEL_DIR, "action_model.joblib"))
    joblib.dump(action_mapping, os.path.join(MODEL_DIR, "action_mapping.joblib"))
    print("✓ Models saved successfully")

    print("\nData preprocessing and model training complete.")

if __name__ == "__main__":
    main()
