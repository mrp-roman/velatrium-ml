import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
# from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocess_data import preprocess_data
from utils import save_scaler, save_centroids
import os
import joblib


def calculate_centroids(X, y, label_encoder):
    """
    Calculate centroids (mean feature values) for each class in the dataset.

    :param X: Feature matrix (NumPy array).
    :param y: Encoded labels (NumPy array).
    :param label_encoder: Fitted label encoder.
    :return: Dictionary of centroids for each class.
    """
    centroids = {}
    for class_idx, class_label in enumerate(label_encoder.classes_):
        # Filter data by the current class
        class_data = X[y == class_idx]  # Ensure this is already numeric
        if len(class_data) > 0:
            centroids[class_label] = np.mean(class_data, axis=0)  # Calculate mean along columns
        else:
            print(f"Warning: No data for class {class_label}. Defaulting centroid to zeros.")
            centroids[class_label] = np.zeros(X.shape[1])  # Default centroid as zeros if no data
    print("Calculated centroids:")
    for label, centroid in centroids.items():
        print(f"Label: {label}, Centroid: {centroid}")
    return centroids


def train_network_model(data_path, model_output_path, target_column="Label", company_id=None):
    """
    Train a network anomaly detection model dynamically for global or company-specific data.
    - Combines Random Forest classification and Autoencoder anomaly detection.
    - Calculates risk scores.
    - Identifies global feature importance using SHAP.

    :param data_path: Path to the dataset (CSV).
    :param model_output_path: Base path for saving model and artifacts.
    :param target_column: Column containing the target labels.
    :param company_id: Optional company ID for company-specific training.
    """
    print(f"Training network model {'for company ' + company_id if company_id else 'globally'}.")

    # Preprocess the data
    X, y, scaler, label_encoder, X_benign_scaled = preprocess_data(data_path, target_column=target_column)
    print(label_encoder.classes_)

    # Compute class imbalance ratios for XGBoost
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    print(f"Class weights: {class_weights}")

    # Train a Random Forest classifier
    print("Training classification model...")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # classifier = RandomForestClassifier(random_state=42, n_estimators=100)
    # classifier.fit(X_train, y_train)

    print("Training classification model with XGBoost...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = XGBClassifier(
        objective='multi:softprob', 
        scale_pos_weight=class_weights,  # Balancing
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    classifier.fit(X_train, y_train)

    # Evaluate the classification model
    # y_pred = classifier.predict(X_test)
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    # print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Evaluate the classification model
    y_pred = np.argmax(classifier.predict_proba(X_test), axis=1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Calculate centroids for malicious behavior
    print("Calculating centroids...")
    centroids = calculate_centroids(X_train, y_train, label_encoder)
    print(enumerate(label_encoder.classes_))
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print(f"Label mapping: {label_mapping}, \n label encoder classes: {label_encoder.classes_}")

    # Train an Autoencoder for anomaly detection
    print("Training anomaly detection Autoencoder...")
    autoencoder = Sequential([
        Dense(128, activation="relu", input_dim=X.shape[1]),
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(X.shape[1], activation="sigmoid"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=5,          # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restore weights from the best epoch
    )
    autoencoder.fit(X_benign_scaled, X_benign_scaled, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Calculate maximum reconstruction error
    print("Calculating maximum reconstruction error...")
    reconstructed_data = autoencoder.predict(X_benign_scaled)
    reconstruction_error = np.mean((X_benign_scaled - reconstructed_data) ** 2, axis=1)
    max_reconstruction_error = np.max(reconstruction_error)
    print(f"Maximum Reconstruction Error: {max_reconstruction_error}")

    X_train = X_train.to_numpy()  # Convert DataFrame to NumPy array if needed
    y_train = np.array(y_train)   # Ensure labels are also NumPy arrays
    # Calculate maximum centroid distance
    print("Calculating maximum centroid distance...")
    centroids = calculate_centroids(X_train, y_train, label_encoder)
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

    max_centroid_distance = 0
    for sample, label_idx in zip(X_train, y_train):
        label = label_mapping[label_idx]
        centroid = centroids[label]
        
        # Debugging: Log data types
        # print(f"Sample: {sample}, label: {label}, centroid: {centroid}")
        
        # Convert to NumPy arrays for safe computation
        sample = np.array(sample, dtype=np.float64)
        centroid = np.array(centroid, dtype=np.float64)
        
        distance = np.linalg.norm(sample - centroid)
        # print(distance)
        max_centroid_distance = max(max_centroid_distance, distance)

    print(f"Maximum Centroid Distance: {max_centroid_distance}")

    # # Calculate reconstruction errors for anomaly detection
    # print("Calculating reconstruction errors...")
    # X_reconstruction = autoencoder.predict(X_test)
    # reconstruction_error = np.mean((X_test - X_reconstruction) ** 2, axis=1)

    # # Compute risk scores combining classifier probabilities and reconstruction error
    # print("Calculating risk scores...")
    # classifier_probs = classifier.predict_proba(X_test)
    # benign_prob = classifier_probs[:, 0]  # Assuming BENIGN is the first class

    # min_distances = []
    # for i in range(len(X_test)):
    #     distances = [np.linalg.norm(X_test[i] - centroids[class_label]) for class_label in centroids.keys()]
    #     min_distances.append(np.min(distances))
    # min_distances = np.array(min_distances)

    # normalized_error = MinMaxScaler().fit_transform(reconstruction_error.reshape(-1, 1)).flatten()
    # risk_scores = 0.5 * (1 - benign_prob) + 0.3 * normalized_error + 0.2 * (1 - MinMaxScaler().fit_transform(min_distances.reshape(-1, 1)).flatten())
    # print(f"Average Risk Score: {np.mean(risk_scores):.4f}")

    # # SHAP Analysis
    # print("Performing SHAP analysis on a random subset...")
    # subset_indices = np.random.choice(len(X_test), size=100, replace=False)
    # # subset_indices = np.random.choice(len(X_test), size=5, replace=False)
    # subset_X_test = X_test[subset_indices]
    # explainer = shap.TreeExplainer(classifier)
    # shap_values = explainer.shap_values(subset_X_test)

    # # Aggregate SHAP values and visualize feature importance
    # print("Generating SHAP visualizations...")
    # shap.summary_plot(shap_values, subset_X_test, feature_names=label_encoder.classes_, show=False)
    # shap_path = os.path.join(model_output_path, "network_shap_summary.png")
    # plt.savefig(shap_path)
    # print(f"SHAP visualizations saved at {shap_path}.")

    # Save the models and artifacts
    model_type = "in_house_new" if not company_id else f"company/{company_id}"
    os.makedirs(os.path.join(model_output_path, model_type), exist_ok=True)

    classifier_path = os.path.join(model_output_path, model_type, "network_classifier.pkl")
    joblib.dump(classifier, classifier_path)
    print(f"XGBoost classifier saved at {classifier_path}.")
    joblib.dump(label_mapping, "new_models/label_mapping.pkl")

    save_scaler(scaler, os.path.join(model_output_path, model_type, "network_scaler.pkl"))
    autoencoder.export(os.path.join(model_output_path, model_type, "autoencoder_model/1"))
    save_centroids(centroids, os.path.join(model_output_path, model_type, "centroids.pkl"))

    print("Model and artifacts saved successfully!")


if __name__ == "__main__":
    # Paths
    data_path = "../data/raw/training_data.csv"
    model_output_path = "new_models/"

    # Train global model
    train_network_model(data_path, model_output_path)

    # Train company-specific model
    # train_network_model(data_path, model_output_path, company_id="company_123")