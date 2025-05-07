"""
Data Analysis Script for TORCS AI Racing Controller

This script analyzes the collected telemetry data from TORCS racing simulator.
It provides comprehensive analysis of:
- Basic data statistics and distributions
- Speed and track position analysis
- Gear usage patterns
- RPM distributions
- Feature correlations

The script generates visualizations saved as PNG files:
- telemetry_analysis.png: Basic distributions
- correlation_analysis.png: Feature correlations

Usage:
    python analyze_data.py

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_telemetry_data(file_path):
    # Read the telemetry data
    print("Loading telemetry data...")
    df = pd.read_csv(file_path)
    
    # Basic information
    print("\n=== Basic Information ===")
    print(f"Number of records: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    # Check for missing values
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Analyze speed distribution
    print("\n=== Speed Analysis ===")
    print(f"Average speed: {df['speedX'].mean():.2f}")
    print(f"Max speed: {df['speedX'].max():.2f}")
    print(f"Min speed: {df['speedX'].min():.2f}")
    
    # Analyze gear distribution
    print("\n=== Gear Analysis ===")
    gear_counts = df['gear'].value_counts()
    print("Gear distribution:")
    print(gear_counts)
    
    # Analyze track position
    print("\n=== Track Position Analysis ===")
    print(f"Average track position: {df['trackPos'].mean():.2f}")
    print(f"Track position range: [{df['trackPos'].min():.2f}, {df['trackPos'].max():.2f}]")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Speed distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='speedX', bins=50)
    plt.title('Speed Distribution')
    
    # Track position distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='trackPos', bins=50)
    plt.title('Track Position Distribution')
    
    # Gear distribution
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='gear')
    plt.title('Gear Distribution')
    
    # RPM distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=df, x='rpm', bins=50)
    plt.title('RPM Distribution')
    
    plt.tight_layout()
    plt.savefig('telemetry_analysis.png')
    print("\nAnalysis plots saved as 'telemetry_analysis.png'")
    
    # Correlation analysis
    print("\n=== Correlation Analysis ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('correlation_analysis.png')
    print("Correlation plot saved as 'correlation_analysis.png'")
    
    return df

if __name__ == "__main__":
    # Replace with your telemetry file path
    telemetry_file = "telemetry_log.csv"
    df = analyze_telemetry_data(telemetry_file) 