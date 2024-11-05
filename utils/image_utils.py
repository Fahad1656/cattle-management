# utils/image_utils.py

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import torch

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess the image for model prediction."""
    try:
        with open(image_path, 'rb') as f:
            img = BytesIO(f.read())  # Read the image file
            image = Image.open(img)  # Open the image with PIL
            image = image.convert('RGB')  # Convert to RGB if needed
            image = image.resize(target_size, Image.LANCZOS)  # Resize
            image = img_to_array(image)  # Convert image to array
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess for model
            return image
    except UnidentifiedImageError:
        print(f"Skipping corrupted image: {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
def preprocess_for_disease(image_path, target_size=(256, 256)):
    try:
        with open(image_path, "rb") as f:
            img = BytesIO(f.read())
            image = Image.open(img)
            image = image.convert("RGB")
            image = image.resize(target_size, Image.LANCZOS)
            image = img_to_array(image)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return image
    except UnidentifiedImageError:
        print(f"Skipping corrupted image: {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None