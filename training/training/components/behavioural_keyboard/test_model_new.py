import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import networkx as nx

def test_models(input_csv, autoencoder_model_file, isolation_forest_model_file, output_csv, output_moving_avg, output_plot_file):
    # Load the testing data
    data = pd.read_csv(input_csv)

    # Select relevant features for anomaly detection
    feature_columns = ["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    X_test = data[feature_columns].values

    # Load the trained autoencoder model and scaler
    autoencoder_data = joblib.load(autoencoder_model_file)
    autoencoder = autoencoder_data["autoencoder"]
    scaler_autoencoder = autoencoder_data["scaler"]

    # Load the trained Isolation Forest model and scaler
    isolation_forest_data = joblib.load(isolation_forest_model_file)
    isolation_forest = isolation_forest_data["model"]
    scaler_isolation_forest = isolation_forest_data["scaler"]

    # Normalize the testing data for both models
    X_test_scaled_autoencoder = scaler_autoencoder.transform(X_test)
    X_test_scaled_isolation_forest = scaler_isolation_forest.transform(X_test)

    # ------------------ Autoencoder Risk Scores ------------------
    # Get reconstruction errors from Autoencoder
    X_test_reconstructed = autoencoder.predict(X_test_scaled_autoencoder)
    reconstruction_errors = np.mean(np.square(X_test_scaled_autoencoder - X_test_reconstructed), axis=1)

    # Normalize reconstruction errors to [0, 1]
    epsilon_value = 1e-9
    normalized_autoencoder_scores = (reconstruction_errors - reconstruction_errors.min()) / (reconstruction_errors.max() - reconstruction_errors.min() + epsilon_value)

    # Amplify and normalize risk scores
    autoencoder_risk_scores = 1 - np.exp(-normalized_autoencoder_scores * 3)  # Amplify small differences

    # ---------------- Isolation Forest Risk Scores ----------------
    # Predict anomaly scores using Isolation Forest
    isolation_forest_scores = isolation_forest.decision_function(X_test_scaled_isolation_forest)

    # Normalize and invert Isolation Forest scores
    normalized_isolation_forest_scores = (isolation_forest_scores - isolation_forest_scores.min()) / (isolation_forest_scores.max() - isolation_forest_scores.min() + epsilon_value)
    isolation_forest_risk_scores = 1 - normalized_isolation_forest_scores  # Higher score = more risk

    # ---------------- Combined Risk Scores ----------------
    # Combine the risk scores from both models
    combined_risk_scores = autoencoder_risk_scores * 0.6 + isolation_forest_risk_scores * 0.4

    # Calculate the average risk score
    avg_risk_score = np.mean(combined_risk_scores)
    percent_anomaly = avg_risk_score * 100
    print(f"Percent anomaly: {percent_anomaly:.2f}%")

    # Classify based on a threshold
    threshold = 0.5  # Adjust this based on your use case
    predictions = ["Normal" if score < threshold else "Anomaly" for score in combined_risk_scores]

    # Calculate moving average of combined risk scores every 25 samples
    window_size = 25
    num_chunks = len(combined_risk_scores) // window_size  # Number of full 25-sample chunks
    # Create a new array for chunked averages
    chunked_moving_avg_risk_scores = []

    # Calculate average for each chunk
    for i in range(num_chunks):
        chunk = combined_risk_scores[i * window_size:(i + 1) * window_size]
        chunk_avg = np.mean(chunk)
        chunked_moving_avg_risk_scores.append(chunk_avg)

    # If there are leftover samples, calculate their average as well
    if len(combined_risk_scores) % window_size != 0:
        leftover_chunk = combined_risk_scores[num_chunks * window_size:]
        leftover_avg = np.mean(leftover_chunk)
        chunked_moving_avg_risk_scores.append(leftover_avg)
    
    # Convert to NumPy array for consistency
    chunked_moving_avg_df = pd.DataFrame(chunked_moving_avg_risk_scores, columns=["Chunked Moving Avg Risk Score"])


    # Add predictions and risk scores to the original data
    data["Autoencoder Risk Score"] = autoencoder_risk_scores
    data["Isolation Forest Risk Score"] = isolation_forest_risk_scores
    data["Combined Risk Score"] = combined_risk_scores
    data["Prediction"] = predictions

    # Save the results to a new CSV file
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    chunked_moving_avg_df.to_csv(output_moving_avg, index=False)
    print(f"Moving avg saved to {output_moving_avg}")

    # ---------------- Visualization ----------------
    # Plot the risk scores for visualization
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder_risk_scores, label="Autoencoder Risk Score", linestyle="-", marker="o")
    plt.plot(isolation_forest_risk_scores, label="Isolation Forest Risk Score", linestyle="--", marker="x")
    plt.plot(combined_risk_scores, label="Combined Risk Score", linestyle="-.", marker="s")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold (0.5)")

    plt.title("Risk Scores from Models")
    plt.xlabel("Data Point Index")
    plt.ylabel("Risk Score (0 to 1)")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(output_plot_file)
    plt.show()
    print(f"Risk score plot saved to {output_plot_file}")


if __name__ == "__main__":
    # Input testing data CSV
    input_csv = "data/keylogger_data.csv"  # Replace with your actual testing CSV file
    # Trained model file
    autoencoder_file = "models/autoencoder_model.joblib"  # Replace with your saved autoencoder model file
    isolation_file = "models/isolation_forest_model.joblib"
    # Output results CSV
    output_csv = "testing_results.csv"
    output_moving_avg = "moving_avg.csv"
    output_plot_file = "fingerprint/risk_graph.png"

    test_models(input_csv, autoencoder_file, isolation_file, output_csv, output_moving_avg, output_plot_file)