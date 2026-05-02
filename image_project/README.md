# CIFAR-10 Image Classifier Web App

A complete web application for image classification using SIFT + ORB + PCA + SVM trained on CIFAR-10 dataset.

## Features

- **Backend**: Flask API with image classification endpoint
- **Frontend**: Modern dark theme single-page application
- **Model**: SIFT (500) + ORB (300) + Color Histogram (96) → 896D → PCA (200) → SVM (RBF, C=20)
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (this will take some time):
```bash
cd image_project
python train.py
```

This will create the `models/` directory with all trained models.

3. Start the Flask server:
```bash
python app.py
```

4. Open `index.html` in your browser or serve it:
```bash
python -m http.server 8000
```

Then visit: http://localhost:8000

## Usage

1. Drag and drop an image or click to browse
2. Click "Classify Image"
3. View the predicted class, confidence, and top 3 predictions

## API Endpoint

**POST** `/predict`
- Input: multipart/form-data with `image` file
- Output: JSON with class, confidence, and all scores

Example response:
```json
{
  "class": "airplane",
  "confidence": 0.87,
  "all_scores": {
    "airplane": 0.87,
    "automobile": 0.05,
    ...
  },
  "top_predictions": [
    {"class": "airplane", "score": 0.87},
    {"class": "ship", "score": 0.05},
    {"class": "bird", "score": 0.03}
  ]
}
```

## Project Structure

```
image_project/
├── train.py              # Training script
├── app.py                # Flask API server
├── index.html            # Frontend UI
├── requirements.txt      # Python dependencies
├── models/               # Trained models (created after training)
│   ├── kmeans_sift.pkl
│   ├── kmeans_orb.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── svm.pkl
└── cifar-10-python/      # CIFAR-10 dataset
    └── cifar-10-batches-py/
```

## Model Pipeline

1. Resize image to 64x64
2. Extract SIFT features (100 keypoints) → BoW histogram (500 bins)
3. Extract ORB features (100 keypoints) → BoW histogram (300 bins)
4. Extract Color Histogram (3 channels × 32 bins = 96 features)
5. Concatenate: 500 + 300 + 96 = 896 dimensions
6. Apply StandardScaler
7. Apply PCA (reduce to 200 dimensions)
8. Classify with SVM (RBF kernel, C=20)
