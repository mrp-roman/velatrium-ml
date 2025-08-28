# import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# import numpy as np
# import shap
# from sklearn.model_selection import train_test_split

# def preprocess_new_data(data, target_column="Label"):
#     """
#     Preprocess the network anomaly data.
#     - Scales numerical features.
#     - Encodes labels.

#     :param data: DataFrame containing raw network features.
#     :param target_column: Column containing the labels (e.g., "Label").
#     :return: Preprocessed X, y, and the scaler.
#     """
#     # Separate features and target
#     X = data.drop(columns=[target_column])
#     y = data[target_column]

#     # Encode labels
#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y)

#     # Scale numerical features
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y, scaler, label_encoder

# def train_network_model(new_data_path, company_id=None):
#     """
#     Train a network anomaly detection model for new data format.
#     - Performs SHAP analysis for feature importance.

#     :param new_data_path: Path to the new dataset (CSV file).
#     :param company_id: ID of the company for company-specific training.
#     """
#     # Load the dataset
#     data = pd.read_csv(new_data_path)

#     # Preprocess the data
#     print("Preprocessing data...")
#     X, y, scaler, label_encoder = preprocess_new_data(data, target_column="Label")

#     # Split into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Define Autoencoder
#     print("Training Autoencoder model...")
#     autoencoder = Sequential([
#         Dense(128, activation="relu", input_dim=X_train.shape[1]),
#         Dense(64, activation="relu"),
#         Dense(128, activation="relu"),
#         Dense(X_train.shape[1], activation="sigmoid"),
#     ])
#     autoencoder.compile(optimizer="adam", loss="mse")
#     autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2)

#     # Calculate reconstruction errors
#     X_reconstruction = autoencoder.predict(X_test)
#     reconstruction_error = np.mean((X_test - X_reconstruction) ** 2, axis=1)

#     # Normalize reconstruction errors for risk scoring
#     scaler_error = MinMaxScaler()
#     normalized_error = scaler_error.fit_transform(reconstruction_error.reshape(-1, 1)).flatten()
#     risk_score = np.mean(normalized_error)
#     print(f"\nOverall Risk Score: {risk_score:.4f}")

#     # SHAP Analysis
#     print("Performing SHAP analysis...")
#     explainer = shap.KernelExplainer(autoencoder.predict, X_train[:100])  # Subset for efficiency
#     shap_values = explainer.shap_values(X_test[:100])

#     # Aggregate SHAP values
#     mean_shap_values = np.mean(np.abs(shap_values[0]), axis=0)
#     feature_importance = sorted(zip(data.columns[:-1], mean_shap_values), key=lambda x: x[1], reverse=True)

#     # Report global feature importance
#     print("\nGlobal Feature Importance (Top 10):")
#     for feature, importance in feature_importance[:10]:
#         print(f"{feature}: {importance:.4f}")

#     # Save model, scaler, and SHAP results
#     model_type = "in_house" if not company_id else f"company/{company_id}"
#     autoencoder.save(f"models/{model_type}/network_autoencoder.h5")
#     joblib.dump(scaler, f"models/{model_type}/network_scaler.pkl")
#     joblib.dump(label_encoder, f"models/{model_type}/network_label_encoder.pkl")

#     shap.summary_plot(shap_values, X_test, show=False)
#     shap.save_html(f"models/{model_type}/network_shap.html", shap_values)

#     print("Model and results saved.")

# if __name__ == "__main__":
#     # Path to the new formatted dataset
#     new_data_path = "path/to/your/dataset.csv"

#     # Train global model
#     train_network_model(new_data_path)

#     # Train company-specific model
#     train_network_model(new_data_path, company_id="company_123")

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import shap
from preprocess_data import preprocess_data
from utils import save_model, save_scaler
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def train_network_model(data_path, model_output_path, target_column="Label", company_id=None):
    """
    Train a network anomaly detection model dynamically for global or company-specific data.
    - Calculates risk score.
    - Identifies global feature importance using SHAP.

    :param data_path: Path to the dataset (CSV).
    :param model_output_path: Base path for saving model and artifacts.
    :param target_column: Column containing the target labels.
    :param company_id: Optional company ID for company-specific training.
    """
    print(f"Training network model {'for company ' + company_id if company_id else 'globally'}.")

    # Preprocess the data
    X, y, scaler, label_encoder = preprocess_data(data_path, target_column=target_column)

    # Convert labels to one-hot encoding for multi-class classification
    num_classes = len(label_encoder.classes_)
    y_categorical = to_categorical(y, num_classes)

    # Define and train the Autoencoder
    autoencoder = Sequential([
        Dense(128, activation="relu", input_dim=X.shape[1]),
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(X.shape[1], activation="sigmoid"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X, X, epochs=50, batch_size=32, validation_split=0.2)
    # autoencoder.fit(X, X, epochs=20, batch_size=32, validation_split=0.2)

   # Calculate reconstruction errors
    X_reconstruction = autoencoder.predict(X)
    reconstruction_error = np.mean((X - X_reconstruction) ** 2, axis=1)

    # Create a new scaler for reconstruction errors
    error_scaler = MinMaxScaler()

    # Normalize reconstruction errors
    normalized_error = error_scaler.fit_transform(reconstruction_error.to_numpy().reshape(-1, 1)).flatten()
    risk_score = np.mean(normalized_error)  # Average normalized reconstruction error
    print(f"\nOverall Risk Score: {risk_score:.4f}")

    # SHAP Analysis
    print("Performing SHAP analysis...")
    # subset_X = X.iloc[:100]  # Use the first 100 samples for SHAP analysis
    subset_X = X.iloc[:100]  # Use the first 100 samples for SHAP analysis
    explainer = shap.KernelExplainer(autoencoder.predict, subset_X)

    # Ensure SHAP is applied to the same subset
    shap_values = explainer.shap_values(subset_X)

    # Aggregate SHAP values
    mean_shap_values = np.mean(np.abs(shap_values[0]), axis=0)
    feature_importance = sorted(zip(subset_X.columns, mean_shap_values), key=lambda x: x[1], reverse=True)

    # Display feature importance
    print("\nGlobal Feature Importance (Top 10):")
    for feature, importance in feature_importance[:10]:
        print(f"{feature}: {importance:.4f}")

    # Save the model, scaler, and SHAP results
    model_type = "in_house" if not company_id else f"company/{company_id}"
    model_path = os.path.join(model_output_path, model_type, "network_autoencoder.keras")
    scaler_path = os.path.join(model_output_path, model_type, "network_scaler.pkl")
    shap_path = os.path.join(model_output_path, model_type, "network_shap.html")

    # Save artifacts
    autoencoder.save(model_path)
    save_scaler(scaler, scaler_path)

    # Save SHAP force plot
    force_plot = shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0],
        subset_X.iloc[0],
        feature_names=subset_X.columns
    )
    shap.save_html(shap_path, force_plot)

    print("Model and artifacts saved.")

if __name__ == "__main__":
    # Paths
    data_path = "../data/raw/training_data.csv"
    model_output_path = "../models/"

    # Train global model
    train_network_model(data_path, model_output_path)

    # Train company-specific model
    # train_network_model(data_path, model_output_path, company_id="company_123")