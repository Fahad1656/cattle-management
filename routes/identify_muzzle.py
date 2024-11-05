from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
import os
from utils.image_utils import preprocess_image
from utils.muzzle_classes import class_names
from utils.muzzle_matching_helper import identify_cattle_from_single_image

router = APIRouter()
model_path = "./models/muzzle_model.h5"
model = load_model(model_path)

@router.post("/identify-single-image-muzzle")
async def identify_single_muzzle(file: UploadFile):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_image_path = temp_file.name

    try:
        # Save uploaded image content to the temporary file
        with open(temp_image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Preprocess the image for model input
        new_image = preprocess_image(temp_image_path)

        # Check if preprocessing was successful
        if new_image is not None:
            # Add batch dimension for model input and make predictions
            new_image = np.expand_dims(new_image, axis=0)
            predictions = model.predict(new_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]

            # If confidence is below threshold, reject the image
            if confidence < 0.45:
                return JSONResponse(
                    content={
                        "success": False,
                        "message": "This is not a perfectly captured muzzle. Please upload a valid one!"
                    },
                    status_code=400
                )

            # If confidence is above threshold, proceed to identify cattle ID
            cattle_id, distance = identify_cattle_from_single_image(temp_image_path)
            if cattle_id:
                return JSONResponse(
                    content={
                        "success": True,
                        "cattle_id": cattle_id,
                        "distance": float(distance)  # Ensure distance is a standard float
                    },
                    status_code=200
                )
            else:
                return JSONResponse(
                    content={"error": "No matching cattle ID identified"},
                    status_code=404
                )

        # Handle case where image preprocessing fails
        else:
            return JSONResponse(
                content={"error": "Image preprocessing failed"},
                status_code=400
            )

    except Exception as e:
        # Catch and log unexpected errors
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Clean up temporary file
        try:
            os.remove(temp_image_path)
        except OSError as e:
            print(f"Error removing temporary file: {e}")
