import os
from fastapi import FastAPI
import uvicorn


from routes.detect import router as detect_router
from routes.identify_muzzle import router as identify_muzzle_router
from routes.register_muzzle import router as register_muzzle_router
from routes.clear_muzzle_database import router as clear_muzzle_database_router


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Initialize FastAPI app
app = FastAPI()

app.include_router(
    clear_muzzle_database_router, prefix="/api", tags=[" clear muzzle db "]
)


app.include_router(detect_router, prefix="/api", tags=["Disease Detection part2"])
app.include_router(identify_muzzle_router, prefix="/api", tags=[" Register Muzzle "])

app.include_router(register_muzzle_router, prefix="/api", tags=["Identify Muzzle"])


def run_app():
    # Start the FastAPI app with Uvicorn
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.environ.get("PORT", 9000))
    )  # Use PORT from environment variable


if __name__ == "__main__":
    run_app()
