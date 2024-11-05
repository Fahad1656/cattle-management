
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
import numpy as np
from utils.image_utils import preprocess_for_disease
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import torch
from ultralytics import YOLO

router = APIRouter()
model_save_path = "./models/cattle_disease_modelx2.h5"
model = load_model(model_save_path)
yolo_model = YOLO('yolov8n.pt')

# DO NOT CHANGE THE CLASS NAME SERIAL
class_names = [
    "foot_infected",
    "healthy",
    "healthy_cow_mouth",
    "healthy_foot",
    "lumpy_skin",
    "mouth_infected",
]

# Function to extract class names and confidence scores from YOLO results
def get_detections(results):
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = box.conf
            class_name = result.names[class_id]
            detections.append({
                "class_name": class_name,
                "confidence": float(confidence)
            })
    return detections

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
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
        # Run YOLO detection and get class names and confidence scores
        results = yolo_model.predict(source=temp_image_path, save=True)
        detections = get_detections(results)
        print(detections)

        # Check if any high-confidence, non-cow objects are detected
        if len(detections)!=0:
             for detection in detections:
                if detection["class_name"] != "cow" and detection["confidence"] > 0.80:
                    return JSONResponse(
                    content={"error": "Please provide a valid cow image"},
                    status_code=400,
                )
       

        # Proceed with disease prediction if confidence for "cow" is below 0.60 or no "cow" detected
        new_image = preprocess_for_disease(temp_image_path)
        if new_image is not None:
            new_image = np.expand_dims(new_image, axis=0)
            predictions = model.predict(new_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]

            response = {
                "success": True,
                "predicted_class": predicted_class_name,
                "confidence": float(confidence),
            }
            return JSONResponse(content=response,status_code=200)
        else:
            return JSONResponse(
                content={"error": "Image preprocessing failed"}, status_code=400
            )

    except Exception as e:
        print(f"Error in /detect: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        try:
            os.remove(temp_image_path)
        except OSError as e:
            print(f"Error removing temporary file: {e}")
