import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def preprocess_risk_scores(data, window_size=25):
    """
    Prepare sliding windows of risk scores for ARIMA predictions.
    :param data: DataFrame containing risk scores.
    :param window_size: Number of time steps in each window.
    :return: List of windows.
    """
    risk_scores = data["Combined Risk Score"].values
    windows = []
    for i in range(len(risk_scores) - window_size):
        windows.append(risk_scores[i:i + window_size])
    return np.array(windows)


def prepare_lstm_data(data, sequence_length):
    """
    Prepare sliding windows of risk scores for LSTM predictions.
    :param data: Array of risk scores.
    :param sequence_length: Number of time steps in each window.
    :return: X (input sequences), y (next value).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)


def build_lstm_model(sequence_length):
    """
    Build and compile an LSTM model for time-series forecasting.
    :param sequence_length: Number of time steps in each input sequence.
    :return: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(64, activation="relu", input_shape=(sequence_length, 1)),
        Dense(1)  # Output the next value
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_predict_lstm(data, sequence_length=25, epochs=20, batch_size=16):
    """
    Fit LSTM model on data and predict the next value for each sequence.
    :param data: Array of risk scores.
    :param sequence_length: Number of time steps in each input sequence.
    :param epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :return: Predictions for the next risk score after each window.
    """
    # Prepare data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = prepare_lstm_data(data_scaled, sequence_length)

    # Build and train the model
    model = build_lstm_model(sequence_length)

    early_stopping = EarlyStopping(
        monitor="loss",  # Monitor validation loss
        patience=5,          # Stop after 5 epochs of no improvement
        restore_best_weights=True,  # Restore weights from the best epoch
        verbose=1
    )

    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

    # Predict the next value for each sequence
    predictions_scaled = model.predict(X, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)

    return predictions.flatten()

def detect_trends_and_magnitude(predictions, actual, data, magnitude_threshold=0.45):
    """
    Analyze trends and calculate the magnitude of risk increase.
    :param predictions: Predicted risk scores.
    :param actual: Actual risk scores.
    :param data: Original data for referencing risk scores.
    :param magnitude_threshold: Threshold for determining high-likelihood threats.
    :return: Trend labels, magnitudes, and threat likelihoods.
    """
    trends = []
    magnitudes = []
    likelihoods = []

    for i, (pred, act) in enumerate(zip(predictions, actual)):
        if np.isnan(pred):
            trends.append(0)
            magnitudes.append(0)
            likelihoods.append("Low")
        else:
            trend = 1 if pred > act else 0  # 1 if increasing, 0 if stable/decreasing
            magnitude = pred - data["Combined Risk Score"].iloc[i]
            normalized_magnitude = magnitude / max(1e-6, data["Combined Risk Score"].iloc[i])  # Normalize
            
            # Classify likelihood
            likelihood = "High" if normalized_magnitude > magnitude_threshold else "Low"

            trends.append(trend)
            magnitudes.append(normalized_magnitude)
            likelihoods.append(likelihood)

    return trends, magnitudes, likelihoods


# def plot_trends_with_magnitudes(data, trends, predictions, magnitudes, likelihoods, window_size):
#     """
#     Visualize the trends, predictions, and magnitudes of risk increase.
#     """
#     plt.figure(figsize=(14, 8))
#     plt.plot(data["Combined Risk Score"], label="Actual Risk Score")
#     plt.plot(range(window_size, len(predictions) + window_size), predictions, label="Predicted Risk Score", linestyle="--")
#     for i, (trend, magnitude, likelihood) in enumerate(zip(trends, magnitudes, likelihoods)):
#         color = "green" if trend == 0 else ("orange" if likelihood == "Low" else "red")
#         plt.axvline(x=i + window_size, color=color, alpha=0.3)
#     plt.legend()
#     plt.title("Risk Score Trends with Magnitudes")
#     plt.xlabel("Time")
#     plt.ylabel("Risk Score")
#     plt.grid(True)
#     plt.show()


def compute_trends_and_magnitudes(input_csv, output_csv, window_size=25, magnitude_threshold=0.45):
    """
    Compute moving predictive trends and magnitudes in risk scores using ARIMA.
    :param input_csv: Path to the CSV file containing risk scores.
    :param window_size: Number of time steps in each window.
    :param magnitude_threshold: Threshold for determining high-likelihood threats.
    """
    # Load the data
    data = pd.read_csv(input_csv)

    # Prepare windows
    windows = preprocess_risk_scores(data, window_size)

    # Fit ARIMA and predict
    predictions = fit_predict_lstm(windows)

    # Compare predictions to actual next risk score
    actual = data["Combined Risk Score"].values[window_size:]
    trends, magnitudes, likelihoods = detect_trends_and_magnitude(predictions, actual, data, magnitude_threshold)

    # # Append predictions and trends to the DataFrame
    # data = data.iloc[window_size:].copy()
    # data["Predicted Risk Score"] = predictions
    # data["Trend"] = ["Increasing" if t == 1 else "Stable/Decreasing" for t in trends]
    # data["Magnitude"] = magnitudes
    # data["Likelihood"] = likelihoods

    # Prepare a new DataFrame for the output
    output_data = data.iloc[window_size:].copy()
    output_data["Predicted Risk Score"] = predictions[:len(output_data)]
    output_data["Trend"] = ["Increasing" if t == 1 else "Stable/Decreasing" for t in trends]
    output_data["Magnitude"] = magnitudes
    output_data["Likelihood"] = likelihoods

    # Save the results to a new CSV file
    output_data.to_csv(output_csv, index=False)
    print(f"Trends and magnitudes saved to {output_csv}")

    # Visualize the results
    # plot_trends_with_magnitudes(output_data, trends, predictions, magnitudes, likelihoods, window_size)



if __name__ == "__main__":
   # File path to risk score CSV
    input_csv = "testing_results.csv"  # Update with your file path
    output_csv = "trend_results.csv"

    # Parameters
    window_size = 25  # Size of each sliding window for trend prediction
    magnitude_threshold = 0.45  # Threshold for high-likelihood threats

    # Compute trends and magnitudes
    compute_trends_and_magnitudes(input_csv, output_csv, window_size, magnitude_threshold)