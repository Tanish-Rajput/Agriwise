# This script extracts color, texture, spot, and shape features from training images and saves them to a CSV file for model training.

import cv2 as cv
import numpy as np
import pandas as pd
import mahotas
import os
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.color import rgb2gray
from glob import glob
import time  # for pauses between images

def extract_color_features(img):
    cv.imshow("Original Color Image", img)
    cv.waitKey(500)  # Pause for 500 ms for visualization

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
    cv.imshow("Grayscale for Texture", gray)
    cv.waitKey(500)

    gray = np.uint8(gray)
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        features.extend(graycoprops(glcm, prop)[0])
    return features

def extract_spot_features(gray):
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    cv.imshow("Threshold for Spots", thresh)
    cv.waitKey(500)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    spot_areas = [cv.contourArea(c) for c in contours if cv.contourArea(c) > 5]
    total_spot_area = sum(spot_areas)
    num_spots = len(spot_areas)
    avg_spot_area = total_spot_area / num_spots if num_spots > 0 else 0
    return [num_spots, total_spot_area, avg_spot_area]

def extract_shape_features(gray):
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imshow("Threshold for Shape", thresh)
    cv.waitKey(500)

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
    img = cv.imread(image_path)
    img = cv.resize(img, (500, 500))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    print(f"🔍 Extracting features from: {image_path}")
    cv.imshow("Input Image", img)
    cv.waitKey(500)

    color_feat = extract_color_features(img)
    texture_feat = extract_texture_features(gray)
    spot_feat = extract_spot_features(gray)
    shape_feat = extract_shape_features(gray)

    cv.destroyAllWindows()
    return color_feat + texture_feat + spot_feat + shape_feat

# Dataset paths
dataset_paths = {
    "healthy": "train_folder/Healthy",
    "septoria": "train_folder/septoria",
    "stripe_rust": "train_folder/stripe_rust"
}

# Prepare dataset
data = []
labels = []

for label, folder in dataset_paths.items():
    for img_path in glob(os.path.join(folder, "*.jpg")):
        try:
            features = extract_features(img_path)
            data.append(features)
            labels.append(label)
        except Exception as e:
            print(f"Failed on {img_path}: {e}")

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("leaf_disease_features.csv", index=False)

print("Feature extraction complete. Saved to 'leaf_disease_features.csv'")
