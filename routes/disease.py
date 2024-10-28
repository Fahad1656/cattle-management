from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from inference_sdk import InferenceHTTPClient

router = APIRouter()

# New endpoint for disease detection using Roboflow
@router.post("/detect-disease")
async def detect_disease(file: UploadFile):
    # Save the uploaded file to disk temporarily
    temp_image_path = tempfile.mktemp(suffix='.jpg')  # Create a temp file with a .jpg suffix

    try:
        with open(temp_image_path, "wb") as buffer:
            content = await file.read()  # Save file to the specified path
            buffer.write(content)

        # Initialize the inference client
        CLIENT = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="apbBAk49Wfw2sjRtMmcH"  # Replace with your actual API key
        )

        # Call the infer method
        print("ok")
        result = CLIENT.infer(temp_image_path, model_id="cattle-disease-pnjdc/3")
        print(result,"vvvvvvvvv")

        # Extract prediction
        if 'predictions' in result and len(result['predictions']) > 0:
            prediction = result['predictions'][0]
            response = {
                "succcess":True,
                "predicted_class": prediction['class'],
                "confidence": prediction['confidence'],
                
            }
            return JSONResponse(content=response)
        else:
            return JSONResponse(content={"error": "No predictions found"}, status_code=404)

    except Exception as e:
        print(f"Error in /detect-disease: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
