from fastapi import FastAPI
from app.routers import data_ingestion, prediction, auth
from app.utils.logger import setup_logger

app = FastAPI()

# Initialize logging
setup_logger()

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(data_ingestion.router, prefix="/api/v1/data", tags=["Data Ingestion"])
app.include_router(prediction.router, prefix="/api/v1/predictions", tags=["Predictions"])

@app.get("/")
def root():
    return {"message": "Velatrium Backend is running"}