from pymongo import MongoClient
import pandas as pd
import joblib
import os
import pickle
from keras.models import load_model as keras_load_model

# # MongoDB connection
# client = MongoClient("mongodb://localhost:27017/")
# db = client.velatrium

def fetch_company_data(company_id, dataset_type):
    """
    Fetch company-specific data for a specific dataset type.
    :param company_id: ID of the company.
    :param dataset_type: Dataset type (e.g., network, employee_access, phishing).
    :return: DataFrame containing company-specific data.
    """
    records = list(db.company_data.find({"company_id": company_id, "dataset_type": dataset_type}))
    return pd.DataFrame(records)

def fetch_global_data(dataset_type):
    """
    Fetch global data for a specific dataset type.
    :param dataset_type: Dataset type (e.g., network, employee_access, phishing).
    :return: DataFrame containing global data.
    """
    records = list(db.global_data.find({"dataset_type": dataset_type}))
    return pd.DataFrame(records)

def save_model(model, path):
    """
    Save a trained model to the filesystem.
    :param model: Trained model object.
    :param path: File path to save the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved at {path}")

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