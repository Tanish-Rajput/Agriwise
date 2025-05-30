# AgriWise - Wheat Leaf Disease Classification

AgriWise is a machine learning project that detects diseases in wheat leaves using image processing and feature extraction techniques. It classifies leaves into three categories: **Healthy**, **Septoria**, and **Stripe Rust**.

This project was built with the goal of supporting farmers and agricultural experts in identifying plant diseases early for better crop management.

---

## 📁 Project Structure
```
AGRIWISE/
│
├── data/
│ └── leaf_disease_features.csv # Pre-extracted features from training images (optional)
│
├── train_folder/ # (Optional) Contains training images categorized by class
├── test_folder/ # (Optional) Contains test images for evaluation
│
├── wheat_leaf/ # Contains sample test images for prediction
│
├── model.pkl # Trained ML model saved with Pickle
├── feature_extraction.py # Extracts features from images and saves them to CSV
├── training_model.py # Trains a model using the extracted features
├── predict.py # Takes an image path and predicts the disease class
└── README.md # You're reading it!
```


---

## ⚙️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/agriwise.git
cd agriwise
```

2. **Install the dependencies:**
```bash
pip install -r requirements.txt
```
**(Optional) Extract features from training data:**
If you've deleted the images but want to see how feature extraction works, re-add your training images in train_folder/ structured like:
```bash
train_folder/
├── Healthy/
├── septoria/
└── stripe_rust/
```
3. **Then run:**
```bash
python feature_extraction.py
```

4. **Train the model:**
```bash
python training_model.py
```
This will generate model.pkl.

5. **Predict disease from a new image:**
```bash
python predict.py wheat_leaf/sample.jpg
```

**Output will be:**
```bash
Prediction: This leaf is infected with stripe_rust.
```

**How It Works**

The pipeline extracts the following features from each image:

Color statistics in RGB, HSV, Lab color spaces

Texture descriptors using GLCM (contrast, correlation, etc.)

Spot detection and count

Shape-based features like aspect ratio and solidity

These features are used to train a classical machine learning classifier (e.g., Random Forest or SVM).

**Notes**

Images have been excluded from the repo to keep it lightweight.

If you want to test or retrain, provide your own dataset in the same folder structure.

If you're just exploring, use the provided leaf_disease_features.csv for training.

**Contributions**

Contributions, ideas, or feedback are welcome! Feel free to open an issue or pull request.

