from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import os

# Path to the CSV file
CSV_FILE_PATH = "cattle_muzzle_database.csv"

router = APIRouter()

@router.post("/clear-muzzle-database")
async def clear_muzzle_database():
    global muzzle_df
    
    # Clear the in-memory DataFrame
    muzzle_df = pd.DataFrame(columns=['cattle_id', 'feature_vector'])
    
    # Clear the CSV file by overwriting it with the empty DataFrame
    muzzle_df.to_csv(CSV_FILE_PATH, index=False)
    
    return JSONResponse(content={"success": True, "message": "Muzzle database cleared successfully."})
