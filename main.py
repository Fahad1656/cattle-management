from fastapi import FastAPI
import uvicorn
from routes.muzzle import router as muzzle_router
from routes.disease import router as disease_router
from routes.detect import router as detect_router
from pyngrok import ngrok
import nest_asyncio

# Initialize FastAPI app
app = FastAPI()


app.include_router(muzzle_router, prefix="/api", tags=["Muzzle Detection"])
app.include_router(disease_router, prefix="/api", tags=["Disease Detection"])
app.include_router(detect_router, prefix="/api", tags=["Disease Detection part2"])


nest_asyncio.apply()


def run_app():
    # Set your ngrok authtoken
    ngrok.set_auth_token(
        "2n6ZvXx7eRmcjBVwMcC1O0dVLnR_3vPKENWYrRLRyi2rJ8Rcv"
    )  # Replace with your actual authtoken

    # Expose the FastAPI app via ngrokpy 
    public_url = ngrok.connect(9000)
    print(f"Public URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=9000)


if __name__ == "__main__":
    run_app()
