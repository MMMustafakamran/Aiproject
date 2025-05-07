"""
Threshold Verification Script for TORCS AI Racing Controller

This script verifies and analyzes the action thresholds used in preprocessing.
It provides:
- Visual analysis of threshold effectiveness
- Statistical validation of action triggers
- Distribution analysis for:
  * Steering angles
  * Track positions
  * RPM ranges
  * Speed thresholds

Output:
- threshold_analysis.png: Visual representation of thresholds
- Statistical analysis of threshold effectiveness

Usage:
    python verify_thresholds.py

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def verify_action_thresholds(file_path):
    # Read the data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Steering Analysis
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='angle', bins=50)
    plt.axvline(x=0.1, color='r', linestyle='--', label='Left threshold')
    plt.axvline(x=-0.1, color='g', linestyle='--', label='Right threshold')
    plt.title('Angle Distribution with Steering Thresholds')
    plt.legend()
    
    # 2. Track Position Analysis
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='trackPos', bins=50)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Right threshold')
    plt.axvline(x=-0.5, color='g', linestyle='--', label='Left threshold')
    plt.title('Track Position Distribution with Thresholds')
    plt.legend()
    
    # 3. RPM Analysis
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='rpm', bins=50)
    plt.axvline(x=7000, color='r', linestyle='--', label='Gear up threshold')
    plt.axvline(x=3000, color='g', linestyle='--', label='Gear down threshold')
    plt.title('RPM Distribution with Gear Change Thresholds')
    plt.legend()
    
    # 4. Speed Analysis
    plt.subplot(2, 2, 4)
    sns.histplot(data=df, x='speedX', bins=50)
    plt.axvline(x=0, color='r', linestyle='--', label='Acceleration threshold')
    plt.title('Speed Distribution with Acceleration Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png')
    print("Threshold analysis plot saved as 'threshold_analysis.png'")
    
    # Print statistics for each threshold
    print("\n=== Threshold Statistics ===")
    
    # Steering thresholds
    print("\nSteering Analysis:")
    print(f"Percentage of angles > 0.1: {(df['angle'] > 0.1).mean()*100:.2f}%")
    print(f"Percentage of angles < -0.1: {(df['angle'] < -0.1).mean()*100:.2f}%")
    
    # Track position thresholds
    print("\nTrack Position Analysis:")
    print(f"Percentage of positions > 0.5: {(df['trackPos'] > 0.5).mean()*100:.2f}%")
    print(f"Percentage of positions < -0.5: {(df['trackPos'] < -0.5).mean()*100:.2f}%")
    
    # RPM thresholds
    print("\nRPM Analysis:")
    print(f"Percentage of RPM > 7000: {(df['rpm'] > 7000).mean()*100:.2f}%")
    print(f"Percentage of RPM < 3000: {(df['rpm'] < 3000).mean()*100:.2f}%")
    
    # Speed thresholds
    print("\nSpeed Analysis:")
    print(f"Percentage of speeds > 0: {(df['speedX'] > 0).mean()*100:.2f}%")
    print(f"Percentage of speeds < 0: {(df['speedX'] < 0).mean()*100:.2f}%")
    
    # Gear analysis
    print("\nGear Analysis:")
    print("Gear distribution:")
    print(df['gear'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    telemetry_file = "telemetry_log.csv"
    verify_action_thresholds(telemetry_file) 