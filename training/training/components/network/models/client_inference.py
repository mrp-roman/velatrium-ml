import os
import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

EXAMPLE_ORGID = "company1"
EXAMPLE_CLIENTID = "michael.johnson2225"
EXAMPLE_TIME = "12:27:22"

# Define the feature order to match training
FEATURE_ORDER = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

RISK_WEIGHTS = {
    "BENIGN": 0.1,  # Low base risk
    "DDoS": 0.9,    # High base risk
    "Botnet": 0.8,  # Moderately high risk
    "PortScan": 0.7,  # Moderate risk
    "Other": 0.6     # Catch-all for unknown classes
}

def cumulative_risk_dynamic_improved(risk_scores, spread_penalty=True):
    """
    Calculate cumulative risk with improved dynamic weights.
    
    :param risk_scores: List of risk scores (0 to 1).
    :param spread_penalty: Whether to penalize spread in scores (default: True).
    :return: Cumulative risk score (0 to 1).
    """
    # Step 1: Dynamic weights (quadratic function)
    weights = 1 + (2 * np.square(risk_scores))  # Higher scores get more weight

    # Step 2: Weighted sum
    weighted_sum = np.sum(weights * risk_scores)

    # Step 3: Normalize by the total weight
    normalization_factor = np.sum(weights)
    cumulative_risk = weighted_sum / normalization_factor

    # Step 4: Optional: Penalize for spread in risk scores
    if spread_penalty:
        variance_penalty = np.std(risk_scores) * 0.1  # 10% penalty based on spread
        cumulative_risk += variance_penalty
        cumulative_risk = np.clip(cumulative_risk, 0, 1)  # Ensure within bounds

    return cumulative_risk

def model_fn(model_dir):
    """
    Load the models and artifacts from the model directory.
    """
    print("Loading models and artifacts...")
    # Load the TensorFlow SavedModel
    autoencoder = tf.saved_model.load(os.path.join(model_dir, "autoencoder_model/1"))  # Path to the folder

    # autoencoder = tf.keras.models.load_model(os.path.join(model_dir, "network_autoencoder.keras"))
    classifier = joblib.load(os.path.join(model_dir, "network_classifier.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "network_scaler.pkl"))
    centroids = joblib.load(os.path.join(model_dir, "centroids.pkl"))
    label_mapping = joblib.load(os.path.join(model_dir, "label_mapping.pkl"))  # Load label mapping
    return {"autoencoder": autoencoder, "classifier": classifier, "scaler": scaler, "centroids": centroids, "label_mapping": label_mapping}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):  # Single record
            data = [data]
        data_frame = pd.DataFrame(data, columns=FEATURE_ORDER)  # Ensure correct feature order
        return data_frame.values  # Convert to NumPy array
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make predictions using the loaded models.
    """
    autoencoder = model["autoencoder"]
    classifier = model["classifier"]
    scaler = model["scaler"]
    centroids = model["centroids"]
    label_mapping = model["label_mapping"]

    print("Labels in label_mapping:")
    for index, label in label_mapping.items():
        print(f"Index {index}: {label}")

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Autoencoder reconstruction error
    infer = autoencoder.signatures["serving_default"]
    print("Available output keys:")
    print(infer.structured_outputs)
    # Prepare input tensor for the SavedModel
    input_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)

    reconstructed_data = infer(input_tensor)["output_0"].numpy()
    reconstruction_error = np.mean((scaled_data - reconstructed_data) ** 2, axis=1)

    # Classification probabilities
    classifier_probs = classifier.predict_proba(scaled_data)
    predicted_label_indices = classifier.predict(scaled_data)
    print(predicted_label_indices)
    predicted_labels = [label_mapping[i] for i in predicted_label_indices]

    # Confidence in classification (highest probability for the predicted label)
    classification_confidence = classifier_probs[
        np.arange(len(predicted_label_indices)), predicted_label_indices
    ]

    print(f"Predicted label index: {predicted_label_indices}")
    # print(f"Classifier prob of label index: {}")

    # benign_prob = classifier_probs[:, 0]  # Assuming "BENIGN" is the first class
    predicted_labels = [label_mapping[i] for i in predicted_label_indices]

    # Centroid-based distances
    min_distances = []
    for sample, label_idx in zip(scaled_data, predicted_label_indices):
        label = label_mapping[label_idx]
        centroid = centroids[label]
        distance = np.linalg.norm(sample - centroid)
        min_distances.append(distance)
    min_distances = np.array(min_distances)

    max_centroid_distance = 3.798771973840376
    max_reconstruction_error = 0.12747319435066515

    # Normalize errors and distances
    # normalized_error = MinMaxScaler().fit_transform(reconstruction_error.reshape(-1, 1)).flatten()
    # normalized_distances = MinMaxScaler().fit_transform(min_distances.reshape(-1, 1)).flatten()

    normalized_error = np.clip(reconstruction_error / max_reconstruction_error, 0, 1)
    normalized_distances = np.clip(min_distances / max_centroid_distance, 0, 1)

    tanh_normalized_error = np.tanh(normalized_error)
    tanh_normalized_distances = np.tanh(normalized_distances)

    # Risk scores
    # risk_scores = 0.4 * (1 - benign_prob) + 0.4 * tanh_normalized_error + 0.2 * (1 - tanh_normalized_distances)
    # risk_scores = np.clip(risk_scores, 0, 1)

    # Combine metrics with classification-based weights
    # Calculate risk scores
    # Combine metrics with classification-based weights
    # risk_scores = []
    # for i, label in enumerate(predicted_labels):
    #     label_risk_weight = RISK_WEIGHTS.get(label, 0.6)  # Default risk weight for unknown labels
    #     risk_score = (
    #         0.5 * label_risk_weight +
    #         0.4 * tanh_normalized_error[i] +
    #         0.3 * tanh_normalized_distances[i] +
    #         0.3 * (1 - benign_prob[i])
    #     )
    #     # Boost risk score for extremely high reconstruction errors
    #     if reconstruction_error[i] > max_reconstruction_error * 0.9:  # Adjust threshold as needed
    #         risk_score += 0.2
    #     risk_scores.append(np.clip(risk_score, 0, 1))

    # return {
    #     "reconstruction_error": reconstruction_error.tolist(),
    #     "risk_scores": risk_scores,
    #     "benign_prob": benign_prob.tolist(),
    #     "predicted_labels": predicted_labels,
    # }
    # Risk scores based on classification confidence and risk weights
    risk_scores = []
    for i, (label, confidence) in enumerate(zip(predicted_labels, classification_confidence)):
        # Risk weight for the predicted label
        label_risk_weight = RISK_WEIGHTS.get(label, 0.6)  # Default weight for unknown labels

        # Adjust risk weight based on confidence
        # Special handling for "BENIGN" predictions
        if label == "BENIGN":
            if confidence < 0.88:  # Low confidence in "BENIGN"
                benign_adjustment = (0.88 - confidence) * 0.5  # Moderate increase in risk
            else:  # High confidence in "BENIGN"
                benign_adjustment = 0  # No adjustment for normal confidence
        else:  # Non-"BENIGN" predictions
            benign_adjustment = 0.5  # Major increase in risk for misclassification

        # Compute the final risk score
        risk_score = (
            0.5 * label_risk_weight +
            0.4 * tanh_normalized_error[i] +  # Weighted reconstruction error
            0.3 * tanh_normalized_distances[i] +  # Weighted centroid distance
            benign_adjustment
        )
       # Boost risk score for extremely high reconstruction errors
        if reconstruction_error[i] > max_reconstruction_error * 0.9:
            risk_score += 0.2

        # Boost risk score for extremely high centroid distances
        if min_distances[i] > max_centroid_distance * 0.9:
            risk_score += 0.2

        # Ensure risk score is bounded between 0 and 1
        risk_scores.append(np.clip(risk_score, 0, 1))
    
    cumulative_risk = cumulative_risk_dynamic_improved(risk_scores)
    print(f"Cumulative risk: {cumulative_risk}")

    return {
    "data": [
        {
            "org_id": EXAMPLE_ORGID,
            "client_id": EXAMPLE_CLIENTID,
            "time_ex": EXAMPLE_TIME,
            "reconstruction_error": reconstruction_error.tolist(),
            "risk_scores": risk_scores,
            "classification_confidence": classification_confidence.tolist(),
            "predicted_labels": predicted_labels,
            "cumulative_risk": cumulative_risk,
        }
    ]
}

def output_fn(prediction, response_content_type):
    """
    Format the predictions for the response.
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")

# Testing section
if __name__ == "__main__":
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "models_client/model")

    model = model_fn(model_dir)

    # Sample input data
    sample_data = [
    {
        "Destination Port": 80,
        "Flow Duration": 10000,
        "Total Fwd Packets": 5000,
        "Total Backward Packets": 0,
        "Total Length of Fwd Packets": 3000000,
        "Total Length of Bwd Packets": 0,
        "Fwd Packet Length Max": 1500,
        "Fwd Packet Length Min": 1500,
        "Fwd Packet Length Mean": 1500.0,
        "Fwd Packet Length Std": 0.0,
        "Bwd Packet Length Max": 0,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 0,
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": 300000000,
        "Flow Packets/s": 5000,
        "Flow IAT Mean": 2.0,
        "Flow IAT Std": 1.0,
        "Flow IAT Max": 5.0,
        "Flow IAT Min": 1.0,
        "Fwd IAT Total": 10000,
        "Fwd IAT Mean": 2.0,
        "Fwd IAT Std": 1.0,
        "Fwd IAT Max": 5.0,
        "Fwd IAT Min": 1.0,
        "Bwd IAT Total": 0,
        "Bwd IAT Mean": 0,
        "Bwd IAT Std": 0,
        "Bwd IAT Max": 0,
        "Bwd IAT Min": 0,
        "Fwd PSH Flags": 1,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": 100,
        "Bwd Header Length": 0,
        "Fwd Packets/s": 5000,
        "Bwd Packets/s": 0,
        "Min Packet Length": 1500,
        "Max Packet Length": 1500,
        "Packet Length Mean": 1500.0,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "FIN Flag Count": 100000000,
        "SYN Flag Count": 1,
        "RST Flag Count": 0,
        "PSH Flag Count": 1,
        "ACK Flag Count": 1,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": 0,
        "Average Packet Size": 1500.0,
        "Avg Fwd Segment Size": 1500.0,
        "Avg Bwd Segment Size": 0,
        "Fwd Header Length.1": 100,
        "Fwd Avg Bytes/Bulk": 0,
        "Fwd Avg Packets/Bulk": 0,
        "Fwd Avg Bulk Rate": 0.0,
        "Bwd Avg Bytes/Bulk": 0,
        "Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0.0,
        "Subflow Fwd Packets": 5000,
        "Subflow Fwd Bytes": 3000000,
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
        "Init_Win_bytes_forward": 29200,
        "Init_Win_bytes_backward": 0,
        "act_data_pkt_fwd": 4999,
        "min_seg_size_forward": 60,
        "Active Mean": 2.5,
        "Active Std": 1.0,
        "Active Max": 5.0,
        "Active Min": 1.0,
        "Idle Mean": 0,
        "Idle Std": 0,
        "Idle Max": 0,
        "Idle Min": 0,
    },
    {
        "Destination Port": 443,
        "Flow Duration": 15000,
        "Total Fwd Packets": 10000,
        "Total Backward Packets": 0,
        "Total Length of Fwd Packets": 6000000,
        "Total Length of Bwd Packets": 0,
        "Fwd Packet Length Max": 1500,
        "Fwd Packet Length Min": 1500000000,
        "Fwd Packet Length Mean": 1500.0,
        "Fwd Packet Length Std": 15251500.0,
        "Bwd Packet Length Max": 0,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 0,
        "Bwd Packet Length Std": 0,
        "Flow Bytes/s": 455555500000000,
        "Flow Packets/s": 6666,
        "Flow IAT Mean": 2.0,
        "Flow IAT Std": 1.0,
        "Flow IAT Max": 5.0,
        "Flow IAT Min": 1.0,
        "Fwd IAT Total": 15000,
        "Fwd IAT Mean": 2.0,
        "Fwd IAT Std": 1.0,
        "Fwd IAT Max": 5.0,
        "Fwd IAT Min": 1.0,
        "Bwd IAT Total": 0,
        "Bwd IAT Mean": 0,
        "Bwd IAT Std": 0,
        "Bwd IAT Max": 0,
        "Bwd IAT Min": 0,
        "Fwd PSH Flags": 1,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": 200,
        "Bwd Header Length": 0,
        "Fwd Packets/s": 6666,
        "Bwd Packets/s": 0,
        "Min Packet Length": 1500,
        "Max Packet Length": 1500,
        "Packet Length Mean": 1500.0,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "FIN Flag Count": 0,
        "SYN Flag Count": 1,
        "RST Flag Count": 0,
        "PSH Flag Count": 1,
        "ACK Flag Count": 1,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": 0,
        "Average Packet Size": 1500.0,
        "Avg Fwd Segment Size": 1500.0,
        "Avg Bwd Segment Size": 0,
        "Fwd Header Length.1": 200,
        "Fwd Avg Bytes/Bulk": 0,
        "Fwd Avg Packets/Bulk": 0,
        "Fwd Avg Bulk Rate": 0.0,
        "Bwd Avg Bytes/Bulk": 0,
        "Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0.0,
        "Subflow Fwd Packets": 10000000000,
        "Subflow Fwd Bytes": 6000000,
        "Subflow Bwd Packets": 0,
        "Subflow Bwd Bytes": 0,
        "Init_Win_bytes_forward": 29200,
        "Init_Win_bytes_backward": 0,
        "act_data_pkt_fwd": 9999,
        "min_seg_size_forward": 60,
        "Active Mean": 2.5,
        "Active Std": 1.0,
        "Active Max": 5.0,
        "Active Min": 1.0,
        "Idle Mean": 0,
        "Idle Std": 0,
        "Idle Max": 0,
        "Idle Min": 0,
    },   
]
    
    # Define ranges for benign data
    benign_data = [
    {
        "Destination Port": 80,
        "Flow Duration": 50000,
        "Total Fwd Packets": 200,
        "Total Backward Packets": 150,
        "Total Length of Fwd Packets": 30000,
        "Total Length of Bwd Packets": 25000,
        "Fwd Packet Length Max": 1200,
        "Fwd Packet Length Min": 50,
        "Fwd Packet Length Mean": 600.0,
        "Fwd Packet Length Std": 30.0,
        "Bwd Packet Length Max": 800,
        "Bwd Packet Length Min": 40,
        "Bwd Packet Length Mean": 400.0,
        "Bwd Packet Length Std": 20.0,
        "Flow Bytes/s": 100000.0,
        "Flow Packets/s": 50.0,
        "Flow IAT Mean": 0.5,
        "Flow IAT Std": 0.2,
        "Flow IAT Max": 2.0,
        "Flow IAT Min": 0.1,
        "Fwd IAT Total": 5000,
        "Fwd IAT Mean": 0.5,
        "Fwd IAT Std": 0.2,
        "Fwd IAT Max": 2.0,
        "Fwd IAT Min": 0.1,
        "Bwd IAT Total": 4000,
        "Bwd IAT Mean": 0.4,
        "Bwd IAT Std": 0.2,
        "Bwd IAT Max": 1.5,
        "Bwd IAT Min": 0.1,
        "Fwd PSH Flags": 0,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": 60,
        "Bwd Header Length": 60,
        "Fwd Packets/s": 25.0,
        "Bwd Packets/s": 20.0,
        "Min Packet Length": 50,
        "Max Packet Length": 1200,
        "Packet Length Mean": 600.0,
        "Packet Length Std": 30.0,
        "Packet Length Variance": 900.0,
        "FIN Flag Count": 0,
        "SYN Flag Count": 1,
        "RST Flag Count": 0,
        "PSH Flag Count": 1,
        "ACK Flag Count": 1,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": 1.0,
        "Average Packet Size": 600.0,
        "Avg Fwd Segment Size": 600.0,
        "Avg Bwd Segment Size": 400.0,
        "Fwd Header Length.1": 60,
        "Fwd Avg Bytes/Bulk": 0,
        "Fwd Avg Packets/Bulk": 0,
        "Fwd Avg Bulk Rate": 0.0,
        "Bwd Avg Bytes/Bulk": 0,
        "Bwd Avg Packets/Bulk": 0,
        "Bwd Avg Bulk Rate": 0.0,
        "Subflow Fwd Packets": 200,
        "Subflow Fwd Bytes": 30000,
        "Subflow Bwd Packets": 150,
        "Subflow Bwd Bytes": 25000,
        "Init_Win_bytes_forward": 1000,
        "Init_Win_bytes_backward": 500,
        "act_data_pkt_fwd": 100,
        "min_seg_size_forward": 40,
        "Active Mean": 1.0,
        "Active Std": 0.5,
        "Active Max": 2.0,
        "Active Min": 0.5,
        "Idle Mean": 1.0,
        "Idle Std": 0.5,
        "Idle Max": 2.0,
        "Idle Min": 0.5
    }]

    # Simulate input processing
    input_data = input_fn(json.dumps(sample_data + benign_data), "application/json")

    # Perform prediction
    prediction = predict_fn(input_data, model)

    # Format output
    response = output_fn(prediction, "application/json")
    print("Prediction Response:")
    print(response)