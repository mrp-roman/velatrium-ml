from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(data, target_column=None):
    """
    Preprocess data: handle missing values, normalize features.
    :param data: Input DataFrame.
    :param target_column: Column to exclude from preprocessing (e.g., target variable).
    :return: Preprocessed DataFrame and scaler object.
    """
    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data
        y = None

    # Fill missing values
    X.fillna(0, inplace=True)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return (pd.DataFrame(X_scaled, columns=X.columns), y, scaler)