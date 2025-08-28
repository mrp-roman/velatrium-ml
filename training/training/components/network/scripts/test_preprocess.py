# test_preprocess_data.py
from preprocess_data import preprocess_data

import pandas as pd

# Provide the dataset path
dataset_path = "../data/raw/training_data.csv"

# Check column names
data = pd.read_csv(dataset_path)
print("Columns in the dataset:", data.columns)

# Print first few rows to confirm content
print("First few rows of the dataset:")
print(data.head())

def test_preprocessing():
    """
    Test the preprocess_data function with a sample dataset.
    """
    # Provide the path to your dataset
    dataset_path = "../data/raw/training_data.csv"  # Update with the correct path

    # Call the preprocess_data function
    try:
        X_scaled, y, scaler, label_encoder = preprocess_data(dataset_path, target_column="Label", balance_data=True)
        
        # Print results for debugging
        print("---------------------------------------")
        print("Preprocessing Successful!")
        print("Shape of X_scaled:", X_scaled.shape)
        print("Sample Encoded y:", y[:10])
        print("Scaler Info:", scaler)
        print("Encoded Labels:", label_encoder.classes_)
        

    except Exception as e:
        print("Preprocessing Failed!")
        print("Error:", e)

if __name__ == "__main__":
    test_preprocessing()