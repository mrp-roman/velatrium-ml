import requests
import tarfile
import os

# Presigned URL from the server
SIGNED_URL = ""

# Local paths
DOWNLOAD_PATH = "models_client.tar.gz"
EXTRACT_PATH = "./models_client"

# Download the file
def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded to {save_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

# Extract the .tar.gz file
def extract_tar_file(file_path, extract_to):
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"File extracted to {extract_to}")
    except Exception as e:
        print(f"Error extracting file: {e}")

# Run the download and extraction process
download_file(SIGNED_URL, DOWNLOAD_PATH)
extract_tar_file(DOWNLOAD_PATH, EXTRACT_PATH)