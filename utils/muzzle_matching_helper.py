import cv2
import numpy as np
import pandas as pd
import os
import random
import tempfile

global muzzle_df
muzzle_df=None

def preprocess_cowMuzzle(image_path):
    """Preprocess the image by resizing it to the required dimensions."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image at path {image_path} could not be loaded.")
        return None  # Return None if the image is not loaded

    image = cv2.resize(image, (128, 128))  # Resize to fixed dimensions
    return image

def extract_composite_features(image):
    """Extract HOG features from the image."""
    features = cv2.HOGDescriptor().compute(image).flatten()
    return features

def load_muzzle_database():
    """Load the muzzle database from a CSV file."""
    try:
        df = pd.read_csv("cattle_muzzle_database.csv")
        print("Muzzle database loaded successfully.")
        return df
    except FileNotFoundError:
        print("Cattle muzzle database CSV file not found. Creating a new DataFrame.")
        return pd.DataFrame(columns=['cattle_id', 'feature_vector'])

# Load the muzzle database at the beginning of your script



# def generate_cattle_id():
#     """Generate a random cattle ID."""
#     return f"cattle_{random.randint(1000, 9999)}"

def register_muzzle_image(image_path, cattle_id):
    """Register a single muzzle image with an optional cattle_id parameter."""
    global muzzle_df
    muzzle_df = load_muzzle_database()

   

    gray_image = preprocess_cowMuzzle(image_path)
    if gray_image is None:
        return None

    feature_vector = extract_composite_features(gray_image)
    if feature_vector.size == 0:
        return None

    feature_vector /= (np.linalg.norm(feature_vector) + 1e-6)
    feature_hex = feature_vector.tobytes().hex()
    new_row = pd.Series({'cattle_id': cattle_id, 'feature_vector': feature_hex})

    muzzle_df = pd.concat([muzzle_df, new_row.to_frame().T], ignore_index=True)
    muzzle_df.to_csv("cattle_muzzle_database.csv", index=False)

    return cattle_id  # Return the generated cattle_id to indicate success

def identify_cattle_from_single_image(image_path, threshold=0.47):
    """Identify cattle based on a single test image using feature matching."""
    global muzzle_df
    muzzle_df = load_muzzle_database()
    gray_image = preprocess_cowMuzzle(image_path)
    if gray_image is None:
        return None, None

    test_vector = extract_composite_features(gray_image)
    if test_vector.size == 0:
        return None, None

    test_vector /= (np.linalg.norm(test_vector) + 1e-6)

    closest_cattle_id = None
    closest_distance = float('inf')

    for _, row in muzzle_df.iterrows():
        stored_vector = np.frombuffer(bytes.fromhex(row['feature_vector']), dtype=np.float32)

        # Skip if the feature vectors are not the same size
        if stored_vector.size != test_vector.size:
            continue

        # Calculate Euclidean distance
        distance = np.linalg.norm(stored_vector - test_vector)
        print(f"Distance: {distance}")

        # Track the closest match
        if distance < closest_distance:
            closest_distance = distance
            closest_cattle_id = row['cattle_id']

    if closest_distance < threshold:
        return closest_cattle_id, closest_distance
    else:
        return None, None
