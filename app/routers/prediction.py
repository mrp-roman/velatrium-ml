from fastapi import APIRouter, HTTPException
from app.services.model_integration import invoke_model
from app.services.storage import fetch_sorted_data

router = APIRouter()

@router.post("/risk-score")
async def get_risk_score(company_id: str):
    try:
        # Fetch data for the specified company
        company_data = fetch_sorted_data(company_id)

        # Invoke the SageMaker model
        prediction = invoke_model(company_data)

        return {"company_id": company_id, "risk_score": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))