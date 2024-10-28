# routes/muzzle.py

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import os
import tempfile
from tensorflow.keras.models import load_model  # Import model loading from TensorFlow
from utils.image_utils import preprocess_image  # Import the preprocess function
from utils.muzzle_classes import class_names

router = APIRouter()

# Load the trained model
model_path = "./models/muzzle_model.h5"
model = load_model(model_path)


@router.post("/check-muzzle")
async def create_upload_file(file: UploadFile):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_image_path = temp_file.name
        try:

            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

        except Exception as e:
            return JSONResponse(
                content={"error": f"Failed to save the uploaded file: {e}"},
                status_code=500,
            )

    try:

        new_image = preprocess_image(temp_image_path)

        if new_image is not None:

            new_image = np.expand_dims(new_image, axis=0)

            predictions = model.predict(new_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]
            if confidence < 0.50:
                return JSONResponse(
                    content={"success": False, "error": "Muzzle not identified"},
                    status_code=404,
                )

            response = {
                "succcess": True,
                "predicted_class": predicted_class_name,
                "confidence": float(confidence),
            }

            return JSONResponse(content=response)

        else:
            return JSONResponse(
                content={"error": "Image preprocessing failed"}, status_code=400
            )

    except Exception as e:

        print(f"Error in /check_muzzle: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:

        try:
            os.remove(temp_image_path)
        except OSError as e:
            print(f"Error removing temporary file: {e}")
