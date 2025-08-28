import os

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")