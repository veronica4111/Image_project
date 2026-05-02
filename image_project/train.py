import pickle
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import os

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(batch[b'labels'])
    return images, labels

def extract_sift_features(image, sift):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, desc = sift.detectAndCompute(gray, None)
    return desc

def extract_orb_features(image, orb):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kp, desc = orb.detectAndCompute(gray, None)
    return desc

def extract_color_histogram(image, bins=32):
    hist = []
    for i in range(3):
        h = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist.extend(h.flatten())
    return np.array(hist)

def build_vocabulary(descriptors, vocab_size):
    all_desc = np.vstack([d for d in descriptors if d is not None])
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=42, batch_size=1000)
    kmeans.fit(all_desc)
    return kmeans

def compute_bow_histogram(descriptors, kmeans):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(kmeans.n_clusters)
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist

print("Loading CIFAR-10 dataset...")
train_images = []
train_labels = []
for i in range(1, 6):
    images, labels = load_cifar_batch(f'../cifar-10-python/cifar-10-batches-py/data_batch_{i}')
    train_images.append(images)
    train_labels.append(labels)

train_images = np.vstack(train_images)
train_labels = np.hstack(train_labels)

print(f"Loaded {len(train_images)} training images")

# Resize images to 64x64
print("Resizing images to 64x64...")
train_images_resized = np.array([cv2.resize(img, (64, 64)) for img in train_images])

# Initialize feature extractors
sift = cv2.SIFT_create(nfeatures=100)
orb = cv2.ORB_create(nfeatures=100)

print("Extracting SIFT descriptors...")
sift_descriptors = []
for i, img in enumerate(train_images_resized):
    if i % 5000 == 0:
        print(f"  Processing image {i}/{len(train_images_resized)}")
    desc = extract_sift_features(img, sift)
    sift_descriptors.append(desc)

print("Building SIFT vocabulary (500 clusters)...")
kmeans_sift = build_vocabulary(sift_descriptors, vocab_size=500)
joblib.dump(kmeans_sift, 'models/kmeans_sift.pkl')

print("Extracting ORB descriptors...")
orb_descriptors = []
for i, img in enumerate(train_images_resized):
    if i % 5000 == 0:
        print(f"  Processing image {i}/{len(train_images_resized)}")
    desc = extract_orb_features(img, orb)
    orb_descriptors.append(desc)

print("Building ORB vocabulary (300 clusters)...")
kmeans_orb = build_vocabulary(orb_descriptors, vocab_size=300)
joblib.dump(kmeans_orb, 'models/kmeans_orb.pkl')

print("Computing feature vectors...")
X_train = []
for i in range(len(train_images_resized)):
    if i % 5000 == 0:
        print(f"  Processing image {i}/{len(train_images_resized)}")
    
    sift_hist = compute_bow_histogram(sift_descriptors[i], kmeans_sift)
    orb_hist = compute_bow_histogram(orb_descriptors[i], kmeans_orb)
    color_hist = extract_color_histogram(train_images_resized[i], bins=32)
    
    feature_vector = np.concatenate([sift_hist, orb_hist, color_hist])
    X_train.append(feature_vector)

X_train = np.array(X_train)
print(f"Feature vector shape: {X_train.shape}")

print("Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'models/scaler.pkl')

print("Applying PCA (200 components)...")
pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
joblib.dump(pca, 'models/pca.pkl')

print("Training SVM (RBF kernel, C=20)...")
svm = SVC(kernel='rbf', C=20, probability=True, random_state=42)
svm.fit(X_train_pca, train_labels)
joblib.dump(svm, 'models/svm.pkl')

print("\nTraining complete! All models saved to models/ directory")
print("Models saved:")
print("  - models/kmeans_sift.pkl")
print("  - models/kmeans_orb.pkl")
print("  - models/scaler.pkl")
print("  - models/pca.pkl")
print("  - models/svm.pkl")
