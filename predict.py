import cv2 as cv
import numpy as np
import pickle
import sys
from skimage.feature.texture import graycomatrix, graycoprops

# ----------------------------------------------
# Helper Functions for Feature Extraction
# ----------------------------------------------

def extract_color_features(img):
    """
    Extract mean and standard deviation from RGB, HSV, and Lab color spaces.
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    features = []
    for space in [img, hsv, lab]:
        for i in range(3):
            channel = space[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
    return features

def extract_texture_features(gray):
    """
    Extract GLCM texture features from a grayscale image.
    """
    gray = np.uint8(gray)
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        features.extend(graycoprops(glcm, prop)[0])
    return features

def extract_spot_features(gray):
    """
    Analyze black spot count and area from the grayscale image.
    """
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    spot_areas = [cv.contourArea(c) for c in contours if cv.contourArea(c) > 5]
    total_spot_area = sum(spot_areas)
    num_spots = len(spot_areas)
    avg_spot_area = total_spot_area / num_spots if num_spots > 0 else 0
    return [num_spots, total_spot_area, avg_spot_area]

def extract_shape_features(gray):
    """
    Compute aspect ratio, extent, and solidity from the largest contour.
    """
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv.contourArea)
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)
        rect_area = w * h
        aspect_ratio = float(w) / h
        extent = float(area) / rect_area
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        return [aspect_ratio, extent, solidity]
    return [0, 0, 0]

def extract_features(image_path):
    """
    Complete feature pipeline: color, texture, spot, and shape.
    """
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv.resize(img, (500, 500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Collect all features
    color_feat = extract_color_features(img)
    texture_feat = extract_texture_features(gray)
    spot_feat = extract_spot_features(gray)
    shape_feat = extract_shape_features(gray)

    return np.array(color_feat + texture_feat + spot_feat + shape_feat).reshape(1, -1)

# ----------------------------------------------
# Main Script
# ----------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Step 1: Extract features
        features = extract_features(image_path)

        # Step 2: Load the trained model
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        # Step 3: Predict the class
        prediction = model.predict(features)[0]

        print(f"\nPrediction: This leaf is most likely '{prediction}'.\n")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
