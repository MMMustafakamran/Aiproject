"""
Model Training Script for TORCS AI Racing Controller

This script trains a machine learning model to predict driving actions
based on telemetry data. It uses a multi-output classifier approach
to predict multiple actions simultaneously.

Features:
- Multi-output classification
- Model evaluation and validation
- Hyperparameter tuning
- Model persistence
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

def load_preprocessed_data():
    """Load the preprocessed training and test data"""
    print("Loading preprocessed data...")
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    return X_train, X_test, y_train, y_test

def create_model():
    """Create and configure the model"""
    # Base Random Forest classifier
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    # Wrap in MultiOutputClassifier for multi-label classification
    model = MultiOutputClassifier(base_model)
    return model

def train_model(X_train, y_train):
    """Train the model with hyperparameter tuning"""
    print("\nTraining model...")
    start_time = time.time()
    
    # Define parameter grid for tuning
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [None, 10, 20],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2]
    }
    
    # Create base model
    base_model = create_model()
    
    # Perform grid search
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print("\nBest parameters:", grid_search.best_params_)
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report for each action
    action_names = ['steer_left', 'steer_right', 'accelerate', 'brake', 'gear_up', 'gear_down']
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=action_names))
    
    return accuracy

def save_model(model, scaler):
    """Save the trained model and scaler"""
    print("\nSaving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully!")

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Load scaler
    scaler = joblib.load('scaler.pkl')
    
    # Save model
    save_model(model, scaler)
    
    print("\nModel training and evaluation complete!")
    print(f"Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 