from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from io import BytesIO
from PIL import Image, UnidentifiedImageError

from utils.image_utils import preprocess_for_disease

router = APIRouter()
model_save_path = "./models/cattle_disease_modelx.h5"
model = load_model(model_save_path)

#DONT CHANGE THE CLASS NAME SERIAL
class_names = [
    "foot_infected",
    "healthy",
    "healthy_cow_mouth",
    "healthy_foot",
    "lumpy skin",
    "mouth_infected",
]


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_image_path = temp_file.name
        try:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()  # Ensure all data is written
        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to save the uploaded file: {e}"},
                status_code=500,
            )

    try:
        # Preprocess the uploaded image
        new_image = preprocess_for_disease(temp_image_path)

        if new_image is not None:
            # Add batch dimension to the image
            new_image = np.expand_dims(new_image, axis=0)
            # Make predictions
            predictions = model.predict(new_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]

            # Create response with the prediction result
            response = {
                "success": True,
                "predicted_class": predicted_class_name,
                "confidence": float(confidence),
            }
            return JSONResponse(content=response)

        else:
            return JSONResponse(
                content={"error": "Image preprocessing failed"}, status_code=400
            )

    except Exception as e:
        # Log and return error details
        print(f"Error in /detect: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Clean up the temporary file
        try:
            os.remove(temp_image_path)
        except OSError as e:
            print(f"Error removing temporary file: {e}")
