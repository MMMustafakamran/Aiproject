import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_prepare_data():
    """Load the processed data and prepare features/targets"""
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load processed data
    input_file = os.path.join(current_dir, "processed_training_data.csv")
    df = pd.read_csv(input_file)
    
    # Define feature sets for each control
    common_features = [
        'Speed_Magnitude', 'SpeedX', 'SpeedY', 'SpeedZ',
        'Dist_From_Center', 'Angle', 'Angle_Change', 'Angle_Acceleration',
        'RPM', 'RPM_Change', 'RPM_Acceleration', 'TrackPos', 'TrackPos_Change',
        'Speed_Angle_Interaction', 'Speed_Position_Interaction', 'Speed_AngleChange_Interaction',
        'SpeedX_MA', 'SpeedY_MA', 'Angle_MA', 'TrackPos_MA', 'RPM_MA',
        'SpeedX_STD', 'SpeedY_STD', 'Angle_STD', 'TrackPos_STD', 'RPM_STD',
        'Speed_Position_Ratio', 'Angle_Position_Ratio'
    ]
    
    # Target variables
    targets = ['Steer', 'Accel', 'Brake']
    
    return df, common_features, targets

def train_and_evaluate_model(X, y, model_name):
    """Train and evaluate a model for a specific control"""
    # Split data using time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize models
    models = {
        'rf': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    best_model = None
    best_score = float('-inf')
    
    # Train and evaluate each model
    for name, model in models.items():
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        mean_cv_score = cv_scores.mean()
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model = model
    
    # Train the best model on the full dataset
    best_model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{model_name} Model Performance:")
    print(f"Best CV R² Score: {best_score:.4f}")
    print(f"\nTop 10 important features for {model_name}:")
    print(feature_importance.head(10))
    
    return best_model, best_score

def save_models(models, metadata):
    """Save trained models and their metadata to disk"""
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for name, model in models.items():
        # Save model
        model_file = os.path.join(current_dir, f'{name.lower()}_model.joblib')
        joblib.dump(model, model_file)
        
        # Save metadata
        meta_file = os.path.join(current_dir, f'{name.lower()}_metadata.joblib')
        joblib.dump(metadata[name], meta_file)
        
        print(f"Saved {name} model and metadata to {current_dir}")

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df, features, targets = load_and_prepare_data()
    
    # Dictionary to store models and metadata
    models = {}
    metadata = {}
    
    # Train models for each control
    for target in targets:
        print(f"\nTraining {target} model...")
        model, score = train_and_evaluate_model(
            df[features], df[target], target
        )
        
        models[target] = model
        metadata[target] = {
            'features': features,
            'cv_score': score,
            'feature_importance': pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
    
    # Save models and metadata
    print("\nSaving models and metadata...")
    save_models(models, metadata)
    
    # Print overall summary
    print("\nOverall Model Performance Summary:")
    for target in targets:
        print(f"\n{target}:")
        print(f"  CV R² Score: {metadata[target]['cv_score']:.4f}")
        print("\n  Top 5 important features:")
        print(metadata[target]['feature_importance'].head())

if __name__ == "__main__":
    main() 