import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_and_prepare_data():
    """Load the processed data and prepare features/targets"""
    # Load processed data
    df = pd.read_csv("processed_training_data.csv")
    
    # Define feature sets for each control
    common_features = [
        'Speed_Magnitude', 'SpeedX', 'SpeedY', 'SpeedZ',
        'Dist_From_Center', 'Angle', 'Angle_Change',
        'RPM', 'RPM_Change', 'TrackPos',
        'Speed_Angle_Interaction', 'Speed_Position_Interaction',
        'SpeedX_MA', 'SpeedY_MA', 'Angle_MA'
    ]
    
    # Target variables
    targets = ['Steer', 'Accel', 'Brake']
    
    return df, common_features, targets

def train_and_evaluate_model(X, y, model_name):
    """Train and evaluate a model for a specific control"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize model with optimized hyperparameters
    if model_name == 'Steer':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
    elif model_name == 'Accel':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
    else:  # Brake
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    # Print results
    print(f"\n{model_name} Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 important features for {model_name}:")
    print(feature_importance.head())
    
    return model, mse, r2, cv_scores.mean()

def save_models(models, model_dir='trained_models'):
    """Save trained models to disk"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for name, model in models.items():
        filename = os.path.join(model_dir, f'{name.lower()}_model.joblib')
        joblib.dump(model, filename)
        print(f"Saved {name} model to {filename}")

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    df, features, targets = load_and_prepare_data()
    
    # Dictionary to store models
    models = {}
    results = {}
    
    # Train models for each control
    for target in targets:
        print(f"\nTraining {target} model...")
        model, mse, r2, cv_r2 = train_and_evaluate_model(
            df[features], df[target], target
        )
        
        models[target] = model
        results[target] = {
            'mse': mse,
            'r2': r2,
            'cv_r2': cv_r2
        }
    
    # Save models
    print("\nSaving models...")
    save_models(models)
    
    # Print overall summary
    print("\nOverall Model Performance Summary:")
    for target, metrics in results.items():
        print(f"\n{target}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  CV R²: {metrics['cv_r2']:.4f}")

if __name__ == "__main__":
    main() 