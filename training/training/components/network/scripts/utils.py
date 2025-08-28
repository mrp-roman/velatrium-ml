import joblib
import os
import pickle
from keras.models import load_model as keras_load_model


def save_model(model, path):
    """
    Save a trained model to the specified path.
    :param model: Model to be saved.
    :param path: Path to save the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")

def save_scaler(scaler, path):
    """
    Save a scaler object to the specified path.
    :param scaler: Scaler object to be saved.
    :param path: Path to save the scaler.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def save_centroids(centroids, file_path):
    """
    Save centroids to a pickle file.
    
    :param centroids: Dictionary of centroids.
    :param file_path: Path to save the centroids.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(centroids, f)
    print(f"Centroids saved to {file_path}")


def load_model(path):
    """
    Load a Keras model from the specified path.
    :param path: Path to the saved model.
    :return: Loaded Keras model.
    """
    model = keras_load_model(path)
    print(f"Model loaded from {path}")
    return model


def save_scaler(scaler, path):
    """
    Save a scaler (e.g., MinMaxScaler) to the specified path using pickle.
    :param scaler: Scaler object to save.
    :param path: Path to save the scaler.
    """
    with open(path, "wb") as file:
        pickle.dump(scaler, file)
    print(f"Scaler saved at {path}")


def load_scaler(path):
    """
    Load a scaler (e.g., MinMaxScaler) from the specified path using pickle.
    :param path: Path to the saved scaler.
    :return: Loaded scaler object.
    """
    with open(path, "rb") as file:
        scaler = pickle.load(file)
    print(f"Scaler loaded from {path}")
    return scaler


def save_centroids(centroids, path):
    """
    Save centroids to the specified path using pickle.
    :param centroids: Dictionary of centroids to save.
    :param path: Path to save the centroids.
    """
    with open(path, "wb") as file:
        pickle.dump(centroids, file)
    print(f"Centroids saved at {path}")


def load_centroids(path):
    """
    Load centroids from the specified path using pickle.
    :param path: Path to the saved centroids.
    :return: Loaded centroids dictionary.
    """
    with open(path, "rb") as file:
        centroids = pickle.load(file)
    print(f"Centroids loaded from {path}")
    return centroids

def load_keras_model(path):
    """
    Load a Keras model from the specified path.
    :param path: Path to the saved model.
    :return: Loaded Keras model.
    """
    if not (path.endswith(".keras") or path.endswith(".h5")):
        raise ValueError(f"File format not supported: {path}. Keras models must be saved with `.keras` or `.h5` extensions.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Keras model file not found at: {path}")
    
    model = keras_load_model(path)
    print(f"Keras model loaded from: {path}")
    return model


def save_pickle_model(model, path):
    """
    Save a model as a `.pkl` file.
    :param model: Model to save (e.g., Random Forest, Scaler).
    :param path: Path to save the model.
    """
    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Pickle model saved at: {path}")


def load_pickle_model(path):
    """
    Load a model saved as a `.pkl` file.
    :param path: Path to the saved model.
    :return: Loaded model.
    """
    if not path.endswith(".pkl"):
        raise ValueError("Pickle models must be saved with the '.pkl' extension.")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Pickle model loaded from: {path}")
    return model