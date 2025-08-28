import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def train_and_export_models(input_csv, output_autoencoder_model, output_isolation_forest_model):
    # Load data
    data = pd.read_csv(input_csv)

    # Select relevant features for anomaly detection
    feature_columns = ["DU.key1.key1", "DD.key1.key2", "DU.key1.key2", "UD.key1.key2", "UU.key1.key2"]
    X = data[feature_columns]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------- Autoencoder Training -----------------------

    # Define autoencoder model
    input_dim = X_scaled.shape[1]
    encoding_dim = 3  # Size of the bottleneck layer (can be tuned)

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_layer)

    # Decoder
    decoded = Dense(input_dim, activation="linear")(encoded)

    # Build the autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # EarlyStopping callback to stop training when no improvement
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=20,          # Stop after 5 epochs of no improvement
        restore_best_weights=True,  # Restore weights from the best epoch
        verbose=1
    )



    # Train the autoencoder
    autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=100,  # Set a high max epoch
        batch_size=16,
        shuffle=True,
        validation_split=0.1,
        callbacks=[early_stopping],  # Include early stopping
        verbose=1
    )

    # Save the autoencoder model and scaler
    joblib.dump({"autoencoder": autoencoder, "scaler": scaler}, output_autoencoder_model)
    print(f"Autoencoder model and scaler saved to {output_autoencoder_model}")

    # ---------------- Isolation Forest Training ------------------------

    # Train Isolation Forest model
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    isolation_forest.fit(X_scaled)

    # Save the Isolation Forest model and scaler
    joblib.dump({"model": isolation_forest, "scaler": scaler}, output_isolation_forest_model)
    print(f"Isolation Forest model and scaler saved to {output_isolation_forest_model}")


if __name__ == "__main__":
    # Input training data CSV
    input_csv = "data/keylogger_data_new_training.csv"  # Replace with your actual input CSV file

    # Output model files
    output_autoencoder_model = "models/autoencoder_model.joblib"
    output_isolation_forest_model = "models/isolation_forest_model.joblib"

    train_and_export_models(input_csv, output_autoencoder_model, output_isolation_forest_model)