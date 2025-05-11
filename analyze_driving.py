#!/usr/bin/env python3
"""
Analyze driving data to identify steering issues and improve model performance.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_combine_data(data_dir):
    """Load and combine all CSV files from the data directory."""
    all_data = []
    for file in Path(data_dir).glob('*.csv'):
        try:
            df = pd.read_csv(file)
            df['file'] = file.name  # Add source file for tracking
            all_data.append(df)
            print(f"Loaded {file.name}")
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    
    if not all_data:
        raise ValueError("No data files found!")
    
    return pd.concat(all_data, ignore_index=True)

def analyze_steering(data):
    """Analyze steering behavior and identify potential issues."""
    # Basic statistics
    print("\nSteering Statistics:")
    print(f"Mean steering: {data['Steer'].mean():.3f}")
    print(f"Std steering: {data['Steer'].std():.3f}")
    print(f"Min steering: {data['Steer'].min():.3f}")
    print(f"Max steering: {data['Steer'].max():.3f}")
    
    # Check for steering bias
    left_steer = data[data['Steer'] < 0]['Steer'].mean()
    right_steer = data[data['Steer'] > 0]['Steer'].mean()
    print(f"\nSteering Bias:")
    print(f"Average left steering: {left_steer:.3f}")
    print(f"Average right steering: {right_steer:.3f}")
    
    # Analyze steering vs track position using custom bins
    print("\nSteering vs Track Position:")
    track_bins = [-1.0, -0.5, 0.0, 0.5, 1.0]
    track_pos_stats = data.groupby(pd.cut(data['TrackPos'], bins=track_bins))['Steer'].agg(['mean', 'std', 'count'])
    print(track_pos_stats)
    
    # Analyze steering vs speed
    print("\nSteering vs Speed:")
    speed_bins = [0, 50, 100, 150, 200, 250]
    speed_stats = data.groupby(pd.cut(data['SpeedX'], bins=speed_bins))['Steer'].agg(['mean', 'std', 'count'])
    print(speed_stats)
    
    # Analyze steering vs angle
    print("\nSteering vs Angle:")
    angle_bins = [-3.14, -1.57, 0, 1.57, 3.14]
    angle_stats = data.groupby(pd.cut(data['Angle'], bins=angle_bins))['Steer'].agg(['mean', 'std', 'count'])
    print(angle_stats)
    
    return {
        'mean_steer': data['Steer'].mean(),
        'std_steer': data['Steer'].std(),
        'left_bias': left_steer,
        'right_bias': right_steer,
        'track_pos_stats': track_pos_stats,
        'speed_stats': speed_stats,
        'angle_stats': angle_stats
    }

def analyze_correlations(data):
    """Analyze correlations between steering and other features."""
    # Select relevant features
    features = ['Steer', 'TrackPos', 'SpeedX', 'SpeedY', 'Angle', 'RPM']
    corr_matrix = data[features].corr()
    
    print("\nFeature Correlations with Steering:")
    for feature in features:
        if feature != 'Steer':
            corr = corr_matrix.loc['Steer', feature]
            print(f"{feature}: {corr:.3f}")
    
    return corr_matrix

def main():
    # Load data
    data_dir = 'results/learning'
    print(f"Loading data from {data_dir}...")
    data = load_and_combine_data(data_dir)
    
    # Basic data info
    print("\nData Overview:")
    print(f"Total samples: {len(data)}")
    print("\nColumns available:")
    print(data.columns.tolist())
    
    # Analyze steering
    print("\nAnalyzing steering behavior...")
    steering_stats = analyze_steering(data)
    
    # Analyze correlations
    print("\nAnalyzing feature correlations...")
    corr_matrix = analyze_correlations(data)
    
    # Print recommendations based on analysis
    print("\nRecommendations:")
    if abs(steering_stats['mean_steer']) > 0.1:
        print(f"- There appears to be a steering bias of {steering_stats['mean_steer']:.3f}")
        print("  Consider adjusting the model to compensate for this bias")
    
    if steering_stats['std_steer'] < 0.1:
        print("- Steering variation is very low, model might be too conservative")
    elif steering_stats['std_steer'] > 0.5:
        print("- Steering variation is very high, model might be too aggressive")
    
    if abs(steering_stats['left_bias']) > abs(steering_stats['right_bias']):
        print("- Model tends to steer left more than right")
    else:
        print("- Model tends to steer right more than left")
        
    # Additional insights
    print("\nAdditional Insights:")
    # Check if steering is more extreme at higher speeds
    high_speed_steer = data[data['SpeedX'] > 100]['Steer'].std()
    low_speed_steer = data[data['SpeedX'] < 50]['Steer'].std()
    if high_speed_steer > low_speed_steer * 1.5:
        print("- Steering is more aggressive at high speeds")
    
    # Check if steering is more extreme when far from center
    far_steer = data[abs(data['TrackPos']) > 0.5]['Steer'].std()
    center_steer = data[abs(data['TrackPos']) < 0.2]['Steer'].std()
    if far_steer > center_steer * 1.5:
        print("- Steering is more aggressive when far from track center")

if __name__ == "__main__":
    main() 