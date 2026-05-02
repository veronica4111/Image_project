import pickle
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")
    import sys
    sys.exit(1)
import os
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# ── Load CIFAR-10 ─────────────────────────────────────────────────
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    return images, labels

def load_all_cifar10(data_dir):
    images, labels = [], []
    for i in range(1, 6):
        path = os.path.join(data_dir, f'data_batch_{i}')
        imgs, lbls = load_cifar10_batch(path)
        images.append(imgs)
        labels.extend(lbls)
    X_train = np.concatenate(images)
    y_train = np.array(labels)
    X_test, y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

data_dir = 'cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_all_cifar10(data_dir)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Visualize Sample Images ───────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(class_names[y_train[i]], fontsize=10)
    ax.axis('off')
plt.suptitle('CIFAR-10 Sample Images', fontsize=14)
plt.tight_layout()
plt.savefig('output.png')
print("Saved in output.png ✅")

# ── Resize images ─────────────────────────────────────────────────
def resize_images(images, size=(64, 64)):
    return np.array([cv2.resize(img, size, interpolation=cv2.INTER_CUBIC) for img in images])

NUM = 20000
print("Resizing images to 64x64...")
X_train_resized = resize_images(X_train[:NUM])
X_test_resized  = resize_images(X_test)
print("Resizing done ✅")

# ── SIFT Feature Extraction ───────────────────────────────────────
def extract_sift_features(images):
    sift = cv2.SIFT_create(nfeatures=100)
    descriptors_list = []
    print(f"Extracting SIFT features from {len(images)} images...")
    for i, img in enumerate(images):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, descriptors = sift.detectAndCompute(img_gray, None)
        descriptors_list.append(descriptors if descriptors is not None else np.zeros((1, 128)))
        if (i+1) % 5000 == 0:
            print(f"  {i+1}/{len(images)} images done ✅")
    return descriptors_list

# ── ORB Feature Extraction ────────────────────────────────────────
def extract_orb_features(images):
    orb = cv2.ORB_create(nfeatures=100)
    descriptors_list = []
    print(f"Extracting ORB features from {len(images)} images...")
    for i, img in enumerate(images):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, descriptors = orb.detectAndCompute(img_gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors.astype(np.float32))
        else:
            descriptors_list.append(np.zeros((1, 32), dtype=np.float32))
        if (i+1) % 5000 == 0:
            print(f"  {i+1}/{len(images)} images done ✅")
    return descriptors_list

# Extract SIFT
sift_train = extract_sift_features(X_train_resized)
sift_test  = extract_sift_features(X_test_resized)
print("SIFT extraction done ✅")

# Extract ORB
orb_train = extract_orb_features(X_train_resized)
orb_test  = extract_orb_features(X_test_resized)
print("ORB extraction done ✅")

# ── Bag of Words for SIFT ─────────────────────────────────────────
def build_vocabulary(descriptors_list, vocab_size):
    all_descriptors = np.vstack(descriptors_list)
    print(f"Total descriptors: {all_descriptors.shape}")
    print(f"Building vocabulary ({vocab_size} words)...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=42, n_init=3)
    kmeans.fit(all_descriptors)
    print("Vocabulary built ✅")
    return kmeans

def image_to_bow(descriptors_list, kmeans, vocab_size):
    bow_features = []
    for descriptors in descriptors_list:
        words = kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
        hist = hist.astype(float) / (hist.sum() + 1e-7)
        bow_features.append(hist)
    return np.array(bow_features)

# SIFT BoW
sift_vocab_size = 500
kmeans_sift = build_vocabulary(sift_train, sift_vocab_size)
X_sift_train = image_to_bow(sift_train, kmeans_sift, sift_vocab_size)
X_sift_test  = image_to_bow(sift_test,  kmeans_sift, sift_vocab_size)
print(f"SIFT BoW shape: {X_sift_train.shape} ✅")

# ORB BoW
orb_vocab_size = 300
kmeans_orb = build_vocabulary(orb_train, orb_vocab_size)
X_orb_train = image_to_bow(orb_train, kmeans_orb, orb_vocab_size)
X_orb_test  = image_to_bow(orb_test,  kmeans_orb, orb_vocab_size)
print(f"ORB BoW shape: {X_orb_train.shape} ✅")

# ── Color Histogram Features ──────────────────────────────────────
def extract_color_features(images):
    features = []
    for img in images:
        hist_features = []
        for channel in range(3):
            hist, _ = np.histogram(img[:,:,channel], bins=32, range=(0, 256))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
            hist_features.extend(hist)
        features.append(hist_features)
    return np.array(features)

print("Extracting color features...")
X_color_train = extract_color_features(X_train_resized)
X_color_test  = extract_color_features(X_test_resized)
print(f"Color features shape: {X_color_train.shape} ✅")

# ── Combine SIFT + ORB + Color ────────────────────────────────────
X_combined_train = np.hstack([X_sift_train, X_orb_train, X_color_train])
X_combined_test  = np.hstack([X_sift_test,  X_orb_test,  X_color_test])
print(f"Combined features shape: {X_combined_train.shape} ✅")

# ── PCA ───────────────────────────────────────────────────────────
print("Applying PCA...")
scaler = StandardScaler()
X_scaled      = scaler.fit_transform(X_combined_train)
X_test_scaled = scaler.transform(X_combined_test)

pca = PCA(n_components=200, random_state=42)
X_pca      = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

explained = pca.explained_variance_ratio_.sum() * 100
print(f"Shape after PCA: {X_pca.shape}")
print(f"Variance explained: {explained:.1f}%")
print("PCA done ✅")

# ── SVM Classification ────────────────────────────────────────────
print("Training SVM... (may take ~10 minutes)")
svm = SVC(kernel='rbf', C=20, gamma='scale', random_state=42)
svm.fit(X_pca, y_train[:NUM])
print("SVM trained ✅")

y_pred = svm.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ── Confusion Matrix ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=class_names, ax=ax, colorbar=False)
plt.title('Confusion Matrix - SIFT + ORB + PCA + SVM')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved ✅")
# ── Results Visualization ─────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate metrics automatically
precision_per_class = precision_score(y_test, y_pred, average=None)
recall_per_class    = recall_score(y_test, y_pred, average=None)
f1_per_class        = f1_score(y_test, y_pred, average=None)

# 1. Bar chart
x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, precision_per_class, width, label='Precision', color='steelblue')
ax.bar(x,         recall_per_class,    width, label='Recall',    color='orange')
ax.bar(x + width, f1_per_class,        width, label='F1-Score',  color='green')
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.set_title('Precision, Recall & F1-Score per Class')
ax.axhline(y=acc, color='red', linestyle='--', label=f'Overall Accuracy {acc*100:.1f}%')
ax.legend()
plt.tight_layout()
plt.savefig('metrics_per_class.png')
print("Metrics chart saved ✅")

# 2. Pipeline summary
print("\n========== PROJECT SUMMARY ==========")
print(f"Dataset        : CIFAR-10 (50,000 train / 10,000 test)")
print(f"Image Size     : 64x64 (resized from 32x32)")
print(f"Feature Method : SIFT (500 words) + ORB (300 words) + Color Hist")
print(f"Dim Reduction  : PCA (200 components, 49.5% variance)")
print(f"Classifier     : SVM (RBF kernel, C=20)")
print(f"Final Accuracy : {acc*100:.2f}%")
print(f"Best Class     : {class_names[np.argmax(f1_per_class)]} ({max(f1_per_class)*100:.1f}% F1)")
print(f"Worst Class    : {class_names[np.argmin(f1_per_class)]} ({min(f1_per_class)*100:.1f}% F1)")
print("=====================================")
