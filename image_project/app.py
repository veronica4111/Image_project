from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import joblib
import os
import base64
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Configure Gemini API
client = genai.Client()

# Load trained models
print("Loading models...")
kmeans_sift = joblib.load('models/kmeans_sift.pkl')
kmeans_orb = joblib.load('models/kmeans_orb.pkl')
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca.pkl')
svm = joblib.load('models/svm.pkl')
print("Models loaded successfully!")

# Initialize feature extractors
sift = cv2.SIFT_create(nfeatures=100)
orb = cv2.ORB_create(nfeatures=100)

def extract_sift_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)
    return desc

def extract_orb_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, desc = orb.detectAndCompute(gray, None)
    return desc

def extract_color_histogram(image, bins=32):
    hist = []
    for i in range(3):
        h = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist.extend(h.flatten())
    return np.array(hist)

def compute_bow_histogram(descriptors, kmeans):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(kmeans.n_clusters)
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

def process_image(image_bytes):
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 64x64
    image = cv2.resize(image, (64, 64))
    
    # Extract SIFT features
    sift_desc = extract_sift_features(image)
    sift_hist = compute_bow_histogram(sift_desc, kmeans_sift)
    
    # Extract ORB features
    orb_desc = extract_orb_features(image)
    orb_hist = compute_bow_histogram(orb_desc, kmeans_orb)
    
    # Extract Color Histogram
    color_hist = extract_color_histogram(image, bins=32)
    
    # Combine features (500 + 300 + 96 = 896)
    feature_vector = np.concatenate([sift_hist, orb_hist, color_hist])
    
    return feature_vector.reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        
        # Process image and extract features
        features = process_image(image_bytes)
        
        # Apply StandardScaler
        features_scaled = scaler.transform(features)
        
        # Apply PCA
        features_pca = pca.transform(features_scaled)
        
        # Predict with SVM
        prediction = svm.predict(features_pca)[0]
        probabilities = svm.predict_proba(features_pca)[0]
        
        # Get predicted class
        predicted_class = CLASS_NAMES[prediction]
        confidence = float(probabilities[prediction])
        
        # Get all scores
        all_scores = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
        
        # Sort scores for top predictions
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Call Gemini API for AI analysis
        gemini_analysis = None
        try:
            # Check if API key is loaded
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                gemini_analysis = "API key not configured."
            else:
                # Prepare prompt for Gemini - STRICT CIFAR-10 ONLY
                prompt = f"""You are a CIFAR-10 image classifier. You MUST classify this image into ONE of these 10 classes ONLY:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

RULES:
- You MUST choose ONE class from the list above
- DO NOT mention any other objects or animals outside this list
- DO NOT say "I see a person" or "human" or anything not in CIFAR-10
- If the image contains something not in the list, choose the CLOSEST matching class

The SVM model predicted: {predicted_class}
Top 3 predictions: {', '.join([f"{cls} ({score:.1%})" for cls, score in sorted_scores[:3]])}

Provide a 2-3 sentence analysis:
1. State which CIFAR-10 class you think it is (MUST be from the 10 classes above)
2. Explain what visual features led you to this classification
3. Mention if you agree or disagree with the SVM prediction

Remember: ONLY use the 10 CIFAR-10 class names in your response."""

                # Prepare image part for Gemini
                import PIL.Image
                import io
                
                # Convert bytes to PIL Image
                pil_image = PIL.Image.open(io.BytesIO(image_bytes))
                
                # Send to Gemini with image and prompt
                response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, pil_image])
                
                gemini_analysis = response.text
            
        except Exception as gemini_error:
                error_msg = str(gemini_error)
                import traceback
                traceback.print_exc()
                
                # Handle specific error types
                if "429" in error_msg or "quota" in error_msg.lower():
                    gemini_analysis = "⏳ Rate limit reached. Please wait a moment and try again. (Free tier: 5 requests/minute)"
                elif "404" in error_msg:
                    gemini_analysis = "❌ Model not available. Please check your API configuration."
                elif "API key" in error_msg:
                    gemini_analysis = "🔑 API key error. Please verify your Gemini API key."
                elif "getaddrinfo failed" in error_msg or "ConnectError" in error_msg or "Network" in error_msg:
                    gemini_analysis = "🌐 Network error: could not reach Gemini API. Check your internet/DNS/proxy settings."
                else:
                    gemini_analysis = f"⚠️ AI analysis temporarily unavailable."
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            'all_scores': all_scores,
            'top_predictions': [{'class': cls, 'score': score} for cls, score in sorted_scores[:3]],
            'gemini_analysis': gemini_analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
