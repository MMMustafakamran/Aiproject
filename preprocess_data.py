import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_telemetry_data(file_path):
    # Read the data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Remove timestamp as it's not useful for prediction
    df = df.drop('timestamp', axis=1)
    
    # Create target variables (actions)
    # We'll create binary columns for each action
    actions = pd.DataFrame()
    
    # Steering (based on angle and track position)
    # Adjusted thresholds for better balance
    actions['steer_left'] = ((df['angle'] > 0.05) | (df['trackPos'] < -0.5)).astype(int)
    actions['steer_right'] = ((df['angle'] < -0.05) | (df['trackPos'] > 0.5)).astype(int)
    
    # Acceleration (based on speed and gear)
    actions['accelerate'] = ((df['speedX'] > 0) & (df['gear'] > 0)).astype(int)
    actions['brake'] = ((df['speedX'] < 0) | (df['gear'] < 0)).astype(int)
    
    # Gear changes (based on RPM and current gear)
    # Adjusted RPM threshold for gear up
    actions['gear_up'] = ((df['rpm'] > 6500) & (df['gear'] < 6)).astype(int)
    actions['gear_down'] = ((df['rpm'] < 3000) & (df['gear'] > 1)).astype(int)
    
    # Select features for training
    features = [
        'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced',
        'fuel', 'gear', 'lastLapTime', 'racePos', 'rpm',
        'speedX', 'speedY', 'speedZ', 'trackPos', 'z'
    ]
    
    X = df[features]
    y = actions
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the processed data
    np.save('X_train.npy', X_train_scaled)
    np.save('X_test.npy', X_test_scaled)
    np.save('y_train.npy', y_train.values)
    np.save('y_test.npy', y_test.values)
    
    # Save the scaler for later use
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\nData preprocessing complete!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print("\nAction distribution in training set:")
    print(y_train.sum())
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    telemetry_file = "telemetry_log.csv"
    X_train, X_test, y_train, y_test = preprocess_telemetry_data(telemetry_file) 