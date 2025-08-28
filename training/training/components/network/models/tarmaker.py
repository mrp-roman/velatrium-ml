import tarfile

artifacts = ["sagemaker_inference/model/network_classifier.pkl", "sagemaker_inference/model/network_scaler.pkl","sagemaker_inference/model/label_mapping.pkl", "sagemaker_inference/model/centroids.pkl", "sagemaker_inference/inference.py", "sagemaker_inference/requirements.txt", "sagemaker_inference/model/autoencoder_model"]
with tarfile.open("model.tar.gz", "w:gz") as tar:
    for artifact in artifacts:
        tar.add(artifact)
print("Artifacts packaged into model.tar.gz.")