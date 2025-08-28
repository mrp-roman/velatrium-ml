from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from app.services.data_processing import validate_and_sort_data
from app.services.storage import save_to_mongodb

router = APIRouter()

@router.post("/")
async def ingest_data(request: Request, background_tasks: BackgroundTasks):
    try:
        # Parse incoming data
        data = await request.json()

        # Validate and sort the data
        validated_data = validate_and_sort_data(data)

        # Save to MongoDB in the background
        background_tasks.add_task(save_to_mongodb, validated_data)

        return {"message": "Data ingestion successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))