import numpy as np
import pandas as pd
from utils import load_keras_model, load_scaler, load_centroids
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def preprocess_test_data(file_path, target_column="Label"):
    """
    Minimal preprocessing for the test data to align with training.
    - Removes the target column if present.
    - Removes leading/trailing whitespaces from column names.
    - Handles infinite values.
    """
    print(f"Loading test data from {file_path}...")
    data = pd.read_csv(file_path)

    # Remove leading/trailing whitespaces from column names
    data.columns = data.columns.str.strip()

    # Drop target column if it exists
    if target_column in data.columns:
        data.drop(columns=[target_column], inplace=True)

    # Handle infinite values and missing data
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.median(), inplace=True)

    print(f"Preprocessed test data shape: {data.shape}")
    return data


def calculate_risk_score(autoencoder, classifier, scaler, X, centroids=None):
    """
    Calculate the risk score for the input data using the trained Autoencoder and Random Forest.
    """
    # Scale features
    X_scaled = scaler.transform(X)

    # Predict class probabilities using Random Forest
    classifier_probs = classifier.predict_proba(X_scaled)
    benign_prob = classifier_probs[:, 0]  # Assuming "BENIGN" is the first class

    # Calculate reconstruction errors using the autoencoder
    X_reconstruction = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean((X_scaled - X_reconstruction) ** 2, axis=1)

    # Calculate distances to centroids (if provided)
    distances_to_centroids = [
        np.min([np.linalg.norm(X_scaled[i] - centroids[cls]) for cls in centroids])
        for i in range(len(X_scaled))
    ] if centroids else np.zeros(len(X_scaled))

    # Normalize reconstruction errors and distances
    normalized_reconstruction_error = MinMaxScaler().fit_transform(reconstruction_error.reshape(-1, 1)).flatten()
    normalized_centroid_distance = (
        MinMaxScaler().fit_transform(np.array(distances_to_centroids).reshape(-1, 1)).flatten()
        if centroids
        else np.zeros(len(X_scaled))
    )

    # Calculate risk scores
    risk_scores = (
        0.5 * (1 - benign_prob)  # Emphasis on Random Forest probabilities
        + 0.3 * normalized_reconstruction_error  # Contribution of reconstruction error
        + 0.2 * (1 - normalized_centroid_distance)  # Contribution of centroid distances
    )

    return risk_scores, reconstruction_error


def test_network_model(data_path, keras_model_path, scaler_path, centroids_path=None):
    """
    Test the trained Autoencoder model.
    """
    print(f"Testing network model with data from {data_path}...")

    # Preprocess the test data
    X = preprocess_test_data(data_path)

    # Load models and scaler
    print("Loading trained models and artifacts...")
    autoencoder = load_keras_model(keras_model_path)
    classifier = joblib.load("../models/in_house/network_classifier.pkl")
    test_scaler = load_scaler(scaler_path)
    centroids = load_centroids(centroids_path) if centroids_path else None

    # Calculate risk scores and reconstruction errors
    risk_scores, reconstruction_error = calculate_risk_score(
        autoencoder, classifier, test_scaler, X, centroids
    )

    # Save risk scores to a CSV
    risk_scores_df = pd.DataFrame({
        "Risk Score": risk_scores
    })
    risk_scores_csv_path = "outputs/risk_scores.csv"
    risk_scores_df.to_csv(risk_scores_csv_path, index=False)
    print(f"Risk scores saved to {risk_scores_csv_path}.")

    # Output reconstruction error and risk score analysis
    print("\nAnalyzing most pressing factors contributing to risk scores...")

    # Perform SHAP analysis on the highest-risk samples
    print("Performing SHAP analysis on the highest-risk samples...")

    # Combine risk scores with the original data
    data_with_risk = X.copy()
    data_with_risk["Risk Score"] = risk_scores
    print(data_with_risk)

    # Define the number of samples to analyze
    top_n = 5

    # Select top 50 high-risk samples
    high_risk_samples = data_with_risk.sort_values("Risk Score", ascending=False).head(top_n)
    subset_X_high = high_risk_samples.drop(columns=["Risk Score"])  # Drop the risk score column to get features
    print("High-risk samples:")
    print(high_risk_samples)

    # Select 50 samples closest to the 75th percentile
    seventy_fifth_percentile = data_with_risk["Risk Score"].quantile(0.75)
    subset_75th = data_with_risk.iloc[
        (data_with_risk["Risk Score"] - seventy_fifth_percentile).abs().argsort()[:top_n]
    ]
    subset_X_75th = subset_75th.drop(columns=["Risk Score"])  # Drop the risk score column to get features
    print("75th percentile samples:")
    print(subset_75th)

    # Select 50 samples closest to the 50th percentile (median)
    median_risk_score = data_with_risk["Risk Score"].median()
    subset_median = data_with_risk.iloc[
        (data_with_risk["Risk Score"] - median_risk_score).abs().argsort()[:top_n]
    ]
    subset_X_median = subset_median.drop(columns=["Risk Score"])  # Drop the risk score column to get features
    print("Median-risk samples:")
    print(subset_median)

    # Function to process SHAP values for each subset
    def process_shap(subset_X, filename_prefix):
        explainer = shap.KernelExplainer(autoencoder.predict, subset_X)
        shap_values = explainer.shap_values(subset_X)

        # Save SHAP values to a CSV
        shap_df = pd.DataFrame(
            shap_values[0],  # Select the first set of SHAP values
            index=range(len(shap_values[0])),  # Match rows with the subset indices
            columns=subset_X.columns
        )
        shap_csv_path = f"outputs/{filename_prefix}_shap_values_autoencoder.csv"
        shap_df.to_csv(shap_csv_path, index=False)
        print(f"SHAP values for {filename_prefix} samples saved to {shap_csv_path}.")

        # Generate and save SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, subset_X, feature_names=subset_X.columns, show=False)
        shap_summary_path = f"outputs/{filename_prefix}_shap_summary_autoencoder.png"
        plt.savefig(shap_summary_path, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary for {filename_prefix} samples saved to '{shap_summary_path}'.")

    # Process SHAP for each subset
    process_shap(subset_X_high, "high_risk")
    process_shap(subset_X_75th, "75th_percentile")
    process_shap(subset_X_median, "median_risk")

if __name__ == "__main__":
    # Paths to test data and model artifacts
    data_path = "collected_data/network_data.csv"
    keras_model_path = "../models/in_house/network_autoencoder.keras"
    scaler_path = "../models/in_house/network_scaler.pkl"
    centroids_path = "../models/in_house/centroids.pkl"  # Optional

    # Test the models
    test_network_model(data_path, keras_model_path, scaler_path, centroids_path)