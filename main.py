
import os
from fastapi import FastAPI
import uvicorn
from routes.muzzle import router as muzzle_router
from routes.disease import router as disease_router
from routes.detect import router as detect_router
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Initialize FastAPI app
app = FastAPI()

app.include_router(muzzle_router, prefix="/api", tags=["Muzzle Detection"])
app.include_router(disease_router, prefix="/api", tags=["Disease Detection"])
app.include_router(detect_router, prefix="/api", tags=["Disease Detection part2"])

def run_app():
    # Start the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 9000)))  # Use PORT from environment variable

if __name__ == "__main__":
    run_app()
