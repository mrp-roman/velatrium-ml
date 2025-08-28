import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
import numpy as np

def clean_labels(label):
    """
    Clean and standardize labels.
    :param label: Raw label string.
    :return: Cleaned and standardized label string.
    """
    replacements = {
        'Web Attack � ': 'WebAttack_',
        'Brute Force': 'BruteForce',
        'Sql Injection': 'SqlInjection',
        'XSS': 'XSS'
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.strip()

# def get_benign_data(data_path, target_column="Label"):
#     """
#     Extract BENIGN data from the dataset based on the target column.
    
#     Args:
#         data_path (str): Path to the dataset CSV.
#         target_column (str): Column containing the target labels.
    
#     Returns:
#         np.ndarray: Feature data filtered for BENIGN labels.
#     """
#     # Load the dataset
#     data = pd.read_csv(data_path)
#     # Strip whitespace from column names
#     data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace
#     print("Column names after stripping whitespace:", data.columns.tolist())

#     # Validate the target column
#     if target_column not in data.columns:
#         raise KeyError(f"'{target_column}' not found in dataset. Available columns: {list(data.columns)}")

#     # Clean and standardize labels
#     print("Cleaning and standardizing labels...")
#     data[target_column] = data[target_column].apply(clean_labels)

#     # Filter rows where the label is BENIGN
#     benign_data = data[data[target_column] == "BENIGN"]

#     # Drop the target column and convert to NumPy array
#     X = benign_features = benign_data.drop(columns=[target_column]).to_numpy()

#     # Replace infinities and NaNs
#     print("Handling missing or infinite values...")
#     X.replace([np.inf, -np.inf], np.nan, inplace=True)
#     X.fillna(X.median(), inplace=True)

#     # Scale features and retain feature names
#     print("Scaling features...")
#     scaler = MinMaxScaler()
#     X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#     # Print the shape and first 5 rows of the BENIGN data
#     print(f"Shape of BENIGN data: {benign_features.shape}")
#     print("First 5 rows of BENIGN data:")
#     print(benign_features[:5])

#     return benign_features

def preprocess_data(data_path, target_column="Label", balance_data=True):
    """
    Preprocess the dataset for training.
    - Cleans and standardizes labels.
    - Strips whitespace from column names.
    - Encodes non-numeric labels.
    - Balances the dataset if enabled.
    - Scales numerical features and retains feature names.

    :param data_path: Path to the raw dataset (CSV).
    :param target_column: Name of the column containing target labels.
    :param balance_data: Whether to balance the dataset.
    :return: Preprocessed X (DataFrame), y, scaler, and label encoder.
    """
    try:
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)

        # Display first few rows of the dataset for debugging
        print("First 5 rows of the dataset:")
        print(data.head())

        # Strip whitespace from column names
        data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace
        print("Column names after stripping whitespace:", data.columns.tolist())

        # Validate the target column
        if target_column not in data.columns:
            raise KeyError(f"'{target_column}' not found in dataset. Available columns: {list(data.columns)}")

        # Clean and standardize labels
        print("Cleaning and standardizing labels...")
        data[target_column] = data[target_column].apply(clean_labels)

        # Display label distribution
        print("Initial Label Distribution:")
        print(data[target_column].value_counts())

        # Check label diversity
        unique_labels = data[target_column].unique()
        print(f"Unique Labels Found: {unique_labels}")
        if len(unique_labels) <= 1:
            raise ValueError(f"Dataset contains only one unique label: {unique_labels}. Add more diverse examples.")

        # Balance the dataset if required
        if balance_data:
            print("Balancing the dataset...")
            majority_class = data[data[target_column] == 'BENIGN']
            minority_classes = data[data[target_column] != 'BENIGN']

            # Oversample minority classes to match majority class size
            minority_classes_oversampled = resample(
                minority_classes,
                replace=True,  # Sample with replacement
                n_samples=len(majority_class),  # Match majority class size
                random_state=42
            )

            # Combine majority class with oversampled minority classes
            data = pd.concat([majority_class, minority_classes_oversampled])
            print("Dataset successfully balanced.")

        # Shuffle the dataset to mix labels
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Display label distribution after balancing
        print("Post-Balance Label Distribution:")
        print(data[target_column].value_counts())

        # Extract BENIGN data for separate use
        print("Extracting BENIGN data...")
        benign_data = data[data[target_column] == "BENIGN"]
        X_benign = benign_data.drop(columns=[target_column]).copy()
        print(f"Shape of BENIGN data: {X_benign.shape}")
        print("First 5 rows of BENIGN data:")
        print(X_benign.head())

        # Separate features and target
        print("Separating features and target column...")
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Debugging
        print(f"Shape of X before scaling: {X.shape}")
        print(f"Shape of y: {y.shape}")
        print("Columns in X before scaling:", X.columns.tolist())

        # Encode target labels
        print("Encoding target labels...")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.values)
        print("Classes Found by LabelEncoder:", label_encoder.classes_)
        print("Encoded Labels (sample):", y_encoded[:10])

        # Convert non-numeric columns in X to numeric (if any)
        print("Converting non-numeric columns to numeric...")
        for col in X.select_dtypes(include=['object']).columns:
            print(f"Encoding column: {col}")
            X[col] = LabelEncoder().fit_transform(X[col])

        # Replace infinities and NaNs
        print("Handling missing or infinite values...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)

        X_benign.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_benign.fillna(X.median(), inplace=True)

        # Scale features and retain feature names
        print("Scaling features...")
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Scale BENIGN data
        print("Scaling BENIGN data...")
        X_benign_scaled = pd.DataFrame(scaler.transform(X_benign), columns=X_benign.columns)

        # Debugging: Check final shape
        print(f"Shape of X_scaled: {X_scaled.shape}")
        print(f"Shape of Labels (y): {y_encoded.shape}")

        print("Preprocessing complete!")
        return X_scaled, y_encoded, scaler, label_encoder, X_benign_scaled

    except Exception as e:
        print("An error occurred during preprocessing:")
        print(e)
        raise