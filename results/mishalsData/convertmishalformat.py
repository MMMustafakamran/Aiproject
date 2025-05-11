#!/usr/bin/env python3
"""
convertmishalformat.py

Converts the driving_data.csv format to match the format used by pyclient.py
Input format (driving_data.csv):
    speedX,speedY,speedZ,angle,trackPos,rpm,gear,steer_input,accel_input,brake_input,...
Output format (car_data_*.csv):
    Step, Time, SpeedX, SpeedY, SpeedZ, TrackPos, Angle, RPM, Gear_State, ...
"""

import os
import pandas as pd
import glob
from datetime import datetime, timedelta

def convert_csv_format(input_file, output_dir):
    """Convert a single CSV file from driving_data format to pyclient format."""
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%d_%H%M%S")
    output_file = os.path.join(output_dir, f'Mishalsconverted_{timestamp}.csv')
    
    # Create new DataFrame with required columns
    new_df = pd.DataFrame()
    
    # Map the columns with exact names from input CSV
    new_df['Step'] = range(1, len(df) + 1)  # 1-based step counter
    
    # Create time column with proper formatting
    start_time = datetime.strptime('00:00:00', '%H:%M:%S')
    new_df['Time'] = [(start_time + timedelta(seconds=i)).strftime('%H:%M:%S') for i in range(len(df))]
    
    # Convert numeric columns with proper precision
    new_df['SpeedX'] = df['speedX'].round(2)
    new_df['SpeedY'] = df['speedY'].round(2)
    new_df['SpeedZ'] = df['speedZ'].round(2)
    new_df['TrackPos'] = df['trackPos'].round(2)
    new_df['Angle'] = df['angle'].round(2)
    new_df['RPM'] = df['rpm'].round(0).astype(int)  # RPM as integer
    new_df['Gear_State'] = df['gear'].astype(int)  # Gear as integer
    new_df['CurLapTime'] = 0.0  # Not available in input data
    new_df['DistFromStart'] = 0.0  # Not available in input data
    new_df['DistRaced'] = 0.0  # Not available in input data
    new_df['Fuel'] = 100.0  # Default value
    new_df['Damage'] = 0.0  # Default value
    new_df['RacePos'] = 1  # Default value
    
    # Control values with proper precision
    new_df['Accel'] = df['accel_input'].round(2)
    new_df['Brake'] = df['brake_input'].round(2)
    new_df['Steer'] = df['steer_input'].round(2)
    new_df['Gear_Control'] = df['gear'].astype(int)  # Gear as integer
    new_df['Clutch'] = 0.0  # Default value
    new_df['Meta'] = 0  # Default meta value as integer
    
    # Ensure all numeric columns have proper precision
    float_columns = ['SpeedX', 'SpeedY', 'SpeedZ', 'TrackPos', 'Angle', 'CurLapTime', 
                    'DistFromStart', 'DistRaced', 'Fuel', 'Damage', 'Accel', 'Brake', 
                    'Steer', 'Clutch']
    for col in float_columns:
        new_df[col] = new_df[col].round(2)
    
    # Save to CSV with proper formatting
    new_df.to_csv(output_file, index=False, float_format='%.2f')
    print(f"Converted {input_file} to {output_file}")
    
    # Print sample of converted data for verification
    print("\nSample of converted data (first row):")
    print(new_df.iloc[0].to_dict())
    
    return output_file

def main():
    # Input and output directories
    input_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing this script
    output_dir = input_dir  # Use the same directory for output
    
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    # Convert each file
    converted_files = []
    for csv_file in csv_files:
        try:
            # Skip if file is already in the new format
            if 'Mishalsconverted_' in os.path.basename(csv_file):
                print(f"Skipping already converted file: {csv_file}")
                continue
                
            output_file = convert_csv_format(csv_file, output_dir)
            converted_files.append(output_file)
        except Exception as e:
            print(f"Error converting {csv_file}: {e}")
            # Print the first few rows of the input file for debugging
            try:
                df = pd.read_csv(csv_file)
                print("\nFirst few rows of input file:")
                print(df.head())
            except Exception as e2:
                print(f"Could not read input file for debugging: {e2}")
    
    # Print summary
    print("\nConversion Summary:")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Successfully converted: {len(converted_files)}")
    print(f"Output directory: {output_dir}")
    
    if converted_files:
        print("\nConverted files:")
        for file in converted_files:
            print(f"- {os.path.basename(file)}")

if __name__ == "__main__":
    main()
