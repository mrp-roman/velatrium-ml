import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def compute_dataset_features(data):
    """
    Compute aggregate features for the dataset to predict the likelihood of a threat.
    :param data: DataFrame containing risk scores, predictions, trends, and magnitudes.
    :return: Feature vector for the dataset.
    """
    features = {}

    # Aggregates for Combined Risk Score
    features["mean_combined_risk"] = data["Combined Risk Score"].mean()
    features["max_combined_risk"] = data["Combined Risk Score"].max()
    features["std_combined_risk"] = data["Combined Risk Score"].std()

    # Aggregates for Predicted Risk Score
    features["mean_predicted_risk"] = data["Predicted Risk Score"].mean()
    features["max_predicted_risk"] = data["Predicted Risk Score"].max()
    features["std_predicted_risk"] = data["Predicted Risk Score"].std()

    # Aggregates for Magnitude
    features["mean_magnitude"] = data["Magnitude"].mean()
    features["max_magnitude"] = data["Magnitude"].max()
    features["min_magnitude"] = data["Magnitude"].min()
    features["std_magnitude"] = data["Magnitude"].std()
    features["normalized_magnitude"] = (features["mean_magnitude"] - features["min_magnitude"]) / (features["max_magnitude"] - features["min_magnitude"] + 1e-6)

    # Proportion of trends that are increasing
    features["proportion_increasing_trend"] = (data["Trend"] == "Increasing").mean()

    # Proportion of high likelihood threats
    features["proportion_high_likelihood"] = (data["Likelihood"] == "High").mean()

    return pd.DataFrame([features])


def compute_threat_likelihood(features):
    """
    Compute the likelihood of a threat for the dataset using a weighted average.
    :param features: DataFrame containing aggregated features for the dataset.
    :return: Probability of a future threat.
    """
    # Define feature weights (adjust based on importance)
    weights = {
        "mean_combined_risk": 0.65,
        "max_combined_risk": 0.05,
        "normalized_magnitude": 0.1,
        "proportion_increasing_trend": 0.1,
        "proportion_high_likelihood": 0.1
    }

    # Compute the weighted risk score
    risk_score = 0
    for feature, weight in weights.items():
        risk_score += features[feature].iloc[0] * weight

    # Log intermediate values for debugging
    print("Original Features:")
    print(features)
    print(f"Weighted Risk Score: {risk_score}")

    # Clamp the risk score to ensure it's within [0, 1]
    risk_probability = min(max(risk_score, 0), 1)

    return risk_probability

def compute_dataset_threat_likelihood(input_csv):
    """
    Compute the likelihood of a threat for the entire dataset using weighted risk scoring.
    :param input_csv: Path to the input CSV file containing sample-level features.
    """
    # Load the data
    data = pd.read_csv(input_csv)

    # Generate dataset-level features
    features = compute_dataset_features(data)

    # Compute threat likelihood
    threat_probability = compute_threat_likelihood(features)

    # Print the likelihood
    print(f"Weighted risk score: {threat_probability * 100:.2f}")
    return threat_probability


if __name__ == "__main__":
    # File path to risk score CSV
    input_csv = "trend_results.csv"  # Input file with sample-level features

    # Compute the likelihood of a threat for the dataset
    compute_dataset_threat_likelihood(input_csv)