import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import joblib

def convert_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS) to seconds"""
    try:
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

def engineer_features(df):
    """
    Create new features from the existing data with enhanced track awareness
    """
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Convert time to seconds
    df_features['Time_Seconds'] = df_features['Time'].apply(convert_time_to_seconds)
    
    # 1. Speed-related features
    # Total speed magnitude
    df_features['Speed_Magnitude'] = np.sqrt(
        df_features['SpeedX']**2 + 
        df_features['SpeedY']**2 + 
        df_features['SpeedZ']**2
    )
    
    # Speed change rate (acceleration)
    df_features['SpeedX_Change'] = df_features['SpeedX'].diff()
    df_features['SpeedY_Change'] = df_features['SpeedY'].diff()
    df_features['SpeedZ_Change'] = df_features['SpeedZ'].diff()
    
    # 2. Track position features
    # Distance from center of track (absolute value)
    df_features['Dist_From_Center'] = abs(df_features['TrackPos'])
    
    # Track position change rate
    df_features['TrackPos_Change'] = df_features['TrackPos'].diff()
    
    # 3. Angle-related features
    # Rate of change of angle
    df_features['Angle_Change'] = df_features['Angle'].diff()
    
    # Angle acceleration (change in angle change)
    df_features['Angle_Acceleration'] = df_features['Angle_Change'].diff()
    
    # 4. RPM-related features
    # RPM change rate
    df_features['RPM_Change'] = df_features['RPM'].diff()
    
    # RPM acceleration
    df_features['RPM_Acceleration'] = df_features['RPM_Change'].diff()
    
    # 5. Control features
    # Combined acceleration (accel - brake)
    df_features['Net_Acceleration'] = df_features['Accel'] - df_features['Brake']
    
    # Control change rates
    df_features['Accel_Change'] = df_features['Accel'].diff()
    df_features['Brake_Change'] = df_features['Brake'].diff()
    df_features['Steer_Change'] = df_features['Steer'].diff()
    
    # 6. Time-based features
    # Time between steps
    df_features['Time_Delta'] = df_features['Time_Seconds'].diff()
    
    # 7. Gear-related features
    # Gear change indicator
    df_features['Gear_Change'] = df_features['Gear_State'].diff()
    
    # 8. Performance features
    # Distance covered per time unit
    df_features['Distance_Rate'] = df_features['DistRaced'] / (df_features['CurLapTime'] + 1e-6)
    
    # 9. Enhanced interaction features
    # Speed * Angle (indicates turning at speed)
    df_features['Speed_Angle_Interaction'] = df_features['Speed_Magnitude'] * df_features['Angle']
    
    # Speed * TrackPos (indicates position relative to speed)
    df_features['Speed_Position_Interaction'] = df_features['Speed_Magnitude'] * df_features['TrackPos']
    
    # Speed * Angle_Change (indicates turning rate at speed)
    df_features['Speed_AngleChange_Interaction'] = df_features['Speed_Magnitude'] * df_features['Angle_Change']
    
    # 10. Rolling window features (using 5-step window)
    window_size = 5
    for feature in ['SpeedX', 'SpeedY', 'Angle', 'TrackPos', 'RPM']:
        # Moving average
        df_features[f'{feature}_MA'] = df_features[feature].rolling(window=window_size, min_periods=1).mean()
        # Moving standard deviation
        df_features[f'{feature}_STD'] = df_features[feature].rolling(window=window_size, min_periods=1).std()
        # Moving min/max
        df_features[f'{feature}_MIN'] = df_features[feature].rolling(window=window_size, min_periods=1).min()
        df_features[f'{feature}_MAX'] = df_features[feature].rolling(window=window_size, min_periods=1).max()
    
    # 11. Safety features
    # Speed relative to track position (indicates if speed is appropriate for position)
    df_features['Speed_Position_Ratio'] = df_features['Speed_Magnitude'] / (abs(df_features['TrackPos']) + 1e-6)
    
    # Angle relative to track position (indicates if angle is appropriate for position)
    df_features['Angle_Position_Ratio'] = df_features['Angle'] / (abs(df_features['TrackPos']) + 1e-6)
    
    # Fill NaN values created by diff() and rolling operations
    df_features = df_features.fillna(0)
    
    return df_features

def scale_features(df):
    """
    Scale numerical features to have zero mean and unit variance.
    Only scales input features, not target variables.
    """
    # Define target variables
    target_vars = ['Accel', 'Brake', 'Steer', 'Gear_State']
    
    # Select numerical columns to scale, excluding target variables
    numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                     if col not in target_vars]
    
    # Create and fit the scaler
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_scaled, scaler

def main():
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the cleaned data
    print("Loading cleaned data...")
    input_file = os.path.join(current_dir, "cleaned_training_data.csv")
    df = pd.read_csv(input_file)
    
    # Engineer features
    print("\nEngineering features...")
    df_features = engineer_features(df)
    
    # Scale features
    print("\nScaling features...")
    df_scaled, scaler = scale_features(df_features)
    
    # Save the processed dataset
    output_file = os.path.join(current_dir, "processed_training_data.csv")
    df_scaled.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    
    # Save the scaler
    scaler_file = os.path.join(current_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")
    
    # Print feature information
    print("\nFeature Information:")
    print(f"Total number of features: {df_scaled.shape[1]}")
    print("\nFeature names:")
    print(df_scaled.columns.tolist())
    
    # Print some basic statistics
    print("\nDataset Statistics:")
    print(df_scaled.describe())

if __name__ == "__main__":
    main() 