import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def convert_time_to_seconds(time_str):
    """Convert time string (HH:MM:SS) to seconds"""
    try:
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

def engineer_features(df):
    """
    Create new features from the existing data
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
    
    # 3. Angle-related features
    # Rate of change of angle
    df_features['Angle_Change'] = df_features['Angle'].diff()
    
    # 4. RPM-related features
    # RPM change rate
    df_features['RPM_Change'] = df_features['RPM'].diff()
    
    # 5. Control features
    # Combined acceleration (accel - brake)
    df_features['Net_Acceleration'] = df_features['Accel'] - df_features['Brake']
    
    # 6. Time-based features
    # Time between steps
    df_features['Time_Delta'] = df_features['Time_Seconds'].diff()
    
    # 7. Gear-related features
    # Gear change indicator
    df_features['Gear_Change'] = df_features['Gear_State'].diff()
    
    # 8. Performance features
    # Distance covered per time unit
    df_features['Distance_Rate'] = df_features['DistRaced'] / (df_features['CurLapTime'] + 1e-6)  # Avoid division by zero
    
    # 9. Interaction features
    # Speed * Angle (indicates turning at speed)
    df_features['Speed_Angle_Interaction'] = df_features['Speed_Magnitude'] * df_features['Angle']
    
    # Speed * TrackPos (indicates position relative to speed)
    df_features['Speed_Position_Interaction'] = df_features['Speed_Magnitude'] * df_features['TrackPos']
    
    # 10. Rolling window features (using 5-step window)
    window_size = 5
    df_features['SpeedX_MA'] = df_features['SpeedX'].rolling(window=window_size, min_periods=1).mean()
    df_features['SpeedY_MA'] = df_features['SpeedY'].rolling(window=window_size, min_periods=1).mean()
    df_features['Angle_MA'] = df_features['Angle'].rolling(window=window_size, min_periods=1).mean()
    
    # Fill NaN values created by diff() and rolling operations
    df_features = df_features.fillna(0)
    
    return df_features

def scale_features(df):
    """
    Scale numerical features to have zero mean and unit variance
    """
    # Select numerical columns to scale
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create and fit the scaler
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_scaled, scaler

def main():
    # Load the cleaned data
    print("Loading cleaned data...")
    df = pd.read_csv("cleaned_training_data.csv")
    
    # Engineer features
    print("\nEngineering features...")
    df_features = engineer_features(df)
    
    # Scale features
    print("\nScaling features...")
    df_scaled, scaler = scale_features(df_features)
    
    # Save the processed dataset
    output_file = "processed_training_data.csv"
    df_scaled.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    
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