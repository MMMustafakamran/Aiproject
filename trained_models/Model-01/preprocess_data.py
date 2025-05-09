import pandas as pd
import numpy as np
import glob
import os

def load_and_combine_data(data_dir):
    """
    Load and combine all CSV files from the specified directory
    """
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # List to store individual dataframes
    dfs = []
    
    # Load each CSV file
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"Loaded {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(dfs)} files into one dataset")
        print(f"Total rows: {len(combined_df)}")
        return combined_df
    else:
        raise Exception("No CSV files found in the specified directory")

def clean_data(df):
    """
    Clean the dataset by:
    1. Removing any rows with missing values
    2. Removing any duplicate rows
    3. Removing any rows with invalid values
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    
    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Remove rows with invalid values
    # Speed should be reasonable
    df_clean = df_clean[df_clean['SpeedX'].abs() < 100]  # Assuming max speed is 100
    df_clean = df_clean[df_clean['SpeedY'].abs() < 100]
    df_clean = df_clean[df_clean['SpeedZ'].abs() < 100]
    
    # Track position should be between -1 and 1
    df_clean = df_clean[df_clean['TrackPos'].abs() <= 1]
    
    # RPM should be positive and reasonable
    df_clean = df_clean[df_clean['RPM'] >= 0]
    df_clean = df_clean[df_clean['RPM'] <= 10000]  # Assuming max RPM is 10000
    
    # Control values should be between -1 and 1
    df_clean = df_clean[df_clean['Accel'].between(-1, 1)]
    df_clean = df_clean[df_clean['Brake'].between(0, 1)]
    df_clean = df_clean[df_clean['Steer'].between(-1, 1)]
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def main():
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the data directory (relative to the script)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "results", "ruleai")
    
    # Load and combine data
    print("Loading and combining data...")
    df = load_and_combine_data(data_dir)
    
    # Clean the data
    print("\nCleaning data...")
    df_clean = clean_data(df)
    
    # Save the cleaned dataset in the Model-01 directory
    output_file = os.path.join(current_dir, "cleaned_training_data.csv")
    df_clean.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
    
    # Print some basic statistics
    print("\nDataset Statistics:")
    print(df_clean.describe())

if __name__ == "__main__":
    main()