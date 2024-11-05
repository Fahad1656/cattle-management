from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import tempfile
import os
from utils.image_utils import preprocess_image
from utils.muzzle_classes import class_names
from utils.muzzle_matching_helper import register_muzzle_image

router = APIRouter()
model_path = "./models/muzzle_model.h5"
model = load_model(model_path)

@router.post("/register-single-image-muzzle")
async def register_single_muzzle(file: UploadFile, cattleID: str):
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_image_path = temp_file.name

    try:
        # Save uploaded file content to the temporary file
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Preprocess the image for model prediction
        new_image = preprocess_image(temp_image_path)

        # If image preprocessing was successful
        if new_image is not None:
            # Expand dimensions for model compatibility and make predictions
            new_image = np.expand_dims(new_image, axis=0)
            predictions = model.predict(new_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]

            # If confidence is below 0.80, reject the image as invalid
            if confidence < 0.70:
                return JSONResponse(
                    content={
                        "success": False,
                        "message": "This is not a perfectly captured muzzle. Please upload a valid one!"
                    },
                    status_code=400
                )

            # If confidence is high enough, proceed to register the muzzle
            register_response = register_muzzle_image(temp_image_path, cattleID)
            if register_response:
                return JSONResponse(
                    content={"success": True, "cattle_id": register_response},
                    status_code=201
                )
            else:
                return JSONResponse(
                    content={"error": "Failed to register muzzle image."},
                    status_code=500
                )

        # Handle case where image preprocessing failed
        else:
            return JSONResponse(
                content={"error": "Image preprocessing failed"},
                status_code=400
            )

    except Exception as e:
        # Catch any unexpected errors
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Clean up the temporary file
        try:
            os.remove(temp_image_path)
        except OSError as e:
            print(f"Error removing temporary file: {e}")
