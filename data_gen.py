import pandas as pd
import numpy as np
from datetime import datetime

def generate_synthetic_data(num_rows=10000, num_feature_columns=150):
    """
    Generate synthetic electricity load data for Tamil Nadu regions
    
    Parameters:
    num_rows: Number of data rows to generate
    num_feature_columns: Number of additional feature columns to include
    """
    np.random.seed(42)
    
    # Fixed regions for Tamil Nadu
    regions = ["Sivananda Colony", "Chinniyampalayam", "Ganapathy", "K K Pudur", "Ganapathy(Krishna Bharathi Power Systems)"]
    
    # Generate timestamps (starting from 2023-01-01)
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=num_rows, freq='H')
    
    # Generate random data
    data = []
    for i in range(num_rows):
        region = np.random.choice(regions)
        date = dates[i]
        
        # Base load with seasonal and time factors
        base_load = np.random.uniform(50, 499)
        seasonal_factor = 1.5 if date.month in [6, 7, 8] else 1.0
        time_factor = 1.2 if 18 <= date.hour <= 22 else 1.0
        
        # Calculate load
        load = base_load * seasonal_factor * time_factor + np.random.normal(0, 5)
        
        # Create row
        row = {
            'Timestamp': date,
            'Region': region,
            'Load': max(0, load)  # Ensure no negative values
        }
        
        # Add additional feature columns
        for j in range(1, num_feature_columns + 1):
            row[f'Feature_{j}'] = np.random.uniform(0, 1)
            
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    # Adjust these parameters as needed
    NUMBER_OF_ROWS = 1000      # Change this to generate more/fewer rows
    NUMBER_OF_FEATURES = 5     # Change this to add more/fewer feature columns
    
    # Generate data
    df = generate_synthetic_data(NUMBER_OF_ROWS, NUMBER_OF_FEATURES)
    
    # Print basic information
    print(f"Generated {len(df)} records with {len(df.columns)} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save to CSV
    output_file = 'dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\nData saved to {output_file}")
    
    # Basic statistics
    print("\nAverage Load by Region:")
    print(df.groupby('Region')['Load'].mean())

if __name__ == "__main__":
    main()