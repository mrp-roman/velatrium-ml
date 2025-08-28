import numpy as np
import pandas as pd
from keras.models import load_model
import joblib  # For loading scaler
from preprocess_data import preprocess_data  # Ensure this matches your structure

def predict_single(input_data, model_path, scaler_path, label_encoder_path=None):
    """
    Predict on a single input value using a trained model.
    
    :param input_data: Dictionary or DataFrame containing feature values.
    :param model_path: Path to the trained model file.
    :param scaler_path: Path to the scaler file used during training.
    :param label_encoder_path: Path to the label encoder (optional, for classification).
    :return: Predicted output (reconstruction error or class probabilities).
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Load the scaler
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)

    # Preprocess the single input
    print("Processing input data...")
    input_df = pd.DataFrame([input_data])  # Create DataFrame from input
    input_scaled = scaler.transform(input_df)  # Scale features

    # Make prediction
    print("Making prediction...")
    prediction = model.predict(input_scaled)

    # If it's an autoencoder, calculate reconstruction error
    reconstruction_error = np.mean((input_scaled - prediction) ** 2)
    print(f"Reconstruction Error: {reconstruction_error}")

    # For classification models, use label encoder to interpret output (if provided)
    if label_encoder_path:
        label_encoder = joblib.load(label_encoder_path)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        print(f"Predicted Class: {predicted_class[0]}")

    return reconstruction_error

if __name__ == "__main__":
    # Example single input (adjust to match your feature structure)
    single_input = {
        "Destination Port": 80,
        "Flow Duration": 500000,
        "Total Fwd Packets": 1000,
        "Total Backward Packets": 10,
        "Total Length of Fwd Packets": 500000,
        "Total Length of Bwd Packets": 5000,
        "Fwd Packet Length Max": 1500,
        "Fwd Packet Length Min": 1000,
        "Fwd Packet Length Mean": 1250,
        "Fwd Packet Length Std": 50,
        "Bwd Packet Length Max": 500,
        "Bwd Packet Length Min": 50,
        "Bwd Packet Length Mean": 275,
        "Bwd Packet Length Std": 150,
        "Flow Bytes/s": 1e7,
        "Flow Packets/s": 4000,
        "Flow IAT Mean": 2000,
        "Flow IAT Std": 500,
        "Flow IAT Max": 5000,
        "Flow IAT Min": 100,
        "Fwd IAT Total": 20000,
        "Fwd IAT Mean": 1000,
        "Fwd IAT Std": 300,
        "Fwd IAT Max": 5000,
        "Fwd IAT Min": 100,
        "Bwd IAT Total": 5000,
        "Bwd IAT Mean": 500,
        "Bwd IAT Std": 100,
        "Bwd IAT Max": 2000,
        "Bwd IAT Min": 50,
        "Fwd PSH Flags": 1,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": 64,
        "Bwd Header Length": 32,
        "Fwd Packets/s": 1800,
        "Bwd Packets/s": 50,
        "Min Packet Length": 50,
        "Max Packet Length": 1500,
        "Packet Length Mean": 750,
        "Packet Length Std": 300,
        "Packet Length Variance": 90000,
        "FIN Flag Count": 0,
        "SYN Flag Count": 1,
        "RST Flag Count": 0,
        "PSH Flag Count": 1,
        "ACK Flag Count": 1,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": 2,
        "Average Packet Size": 1000,
        "Avg Fwd Segment Size": 1250,
        "Avg Bwd Segment Size": 275,
        "Fwd Header Length.1": 64,
        "Fwd Avg Bytes/Bulk": 0,
        "Fwd Avg Packets/Bulk": 0,
        "Fwd Avg Bulk Rate": 0,
        "Bwd Avg Bytes/Bulk": 0,
        "Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0,
        "Subflow Fwd Packets": 1000,
        "Subflow Fwd Bytes": 500000,
        "Subflow Bwd Packets": 10,
        "Subflow Bwd Bytes": 5000,
        "Init_Win_bytes_forward": 65535,
        "Init_Win_bytes_backward": 8192,
        "act_data_pkt_fwd": 980,
        "min_seg_size_forward": 20,
        "Active Mean": 1000,
        "Active Std": 200,
        "Active Max": 2000,
        "Active Min": 500,
        "Idle Mean": 100,
        "Idle Std": 50,
        "Idle Max": 500,
        "Idle Min": 50
    }

    # Paths to artifacts
    model_path = "../models/in_house/network_autoencoder.keras"
    scaler_path = "../models/in_house/network_scaler.pkl"

    # Make prediction
    predict_single(single_input, model_path, scaler_path)