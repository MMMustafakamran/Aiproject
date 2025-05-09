import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def load_data():
    """Load the processed data"""
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "processed_training_data.csv")
    return pd.read_csv(input_file)

def analyze_feature_distributions(df):
    """Analyze and plot distributions of key features"""
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Select important features for driving
    key_features = [
        'Speed_Magnitude', 'Dist_From_Center', 'Angle_Change',
        'Net_Acceleration', 'Speed_Angle_Interaction', 'RPM_Change'
    ]
    
    # Create subplots for distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Key Features')
    
    for idx, feature in enumerate(key_features):
        row = idx // 3
        col = idx % 3
        
        # Plot histogram with KDE
        sns.histplot(data=df, x=feature, kde=True, ax=axes[row, col])
        axes[row, col].set_title(feature)
        
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'feature_distributions.png'))
    plt.close()
    
    # Print statistical summaries
    print("\nStatistical Summary of Key Features:")
    print(df[key_features].describe())
    
    # Calculate skewness and kurtosis
    print("\nSkewness and Kurtosis:")
    for feature in key_features:
        skew = stats.skew(df[feature].dropna())
        kurt = stats.kurtosis(df[feature].dropna())
        print(f"{feature}:")
        print(f"  Skewness: {skew:.3f}")
        print(f"  Kurtosis: {kurt:.3f}")

def analyze_feature_correlations(df):
    """Analyze correlations between features and target variables"""
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Target variables
    targets = ['Accel', 'Brake', 'Steer']
    
    # Features to analyze
    features = [
        'Speed_Magnitude', 'Dist_From_Center', 'Angle_Change',
        'RPM_Change', 'Net_Acceleration', 'Speed_Angle_Interaction',
        'Speed_Position_Interaction', 'SpeedX_MA', 'Angle_MA'
    ]
    
    # Create correlation matrix
    corr_matrix = df[features + targets].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'feature_correlations.png'))
    plt.close()
    
    # Print strongest correlations with target variables
    print("\nStrongest Correlations with Target Variables:")
    for target in targets:
        correlations = corr_matrix[target].sort_values(ascending=False)
        print(f"\n{target} correlations:")
        print(correlations[correlations.index != target].head())

def analyze_temporal_patterns(df):
    """Analyze patterns over time"""
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    temporal_features = ['Speed_Magnitude', 'Net_Acceleration', 'Angle_Change']
    
    plt.figure(figsize=(15, 5))
    for feature in temporal_features:
        plt.plot(df['Step'][:100], df[feature][:100], label=feature)
    
    plt.title('Temporal Patterns (First 100 Steps)')
    plt.xlabel('Step')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'temporal_patterns.png'))
    plt.close()

def main():
    # Load the data
    print("Loading processed data...")
    df = load_data()
    
    # Analyze feature distributions
    print("\nAnalyzing feature distributions...")
    analyze_feature_distributions(df)
    
    # Analyze feature correlations
    print("\nAnalyzing feature correlations...")
    analyze_feature_correlations(df)
    
    # Analyze temporal patterns
    print("\nAnalyzing temporal patterns...")
    analyze_temporal_patterns(df)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main() 