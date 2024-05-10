import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Define constants
ROOT_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_train'
IMG_SIZE = (100, 100)
BATCH_SIZE = 8
NUM_CLASSES = 29
NUM_EPOCHS = 10
NUM_IMAGES = 100  # Number of images to load per class

# Model paths
model_paths = [
    "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\my_model.h5",
    "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\model_fixed_snr.h5",
    "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\model_variable_snr.h5"
]

# Define function to load images and labels
def load_data(root_dir, num_images):
    images = []
    labels = []
    class_folders = os.listdir(root_dir)
    for class_folder in class_folders:
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):
            class_images = os.listdir(class_path)[:num_images]
            for image_name in class_images:
                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, IMG_SIZE)
                images.append(image)
                labels.append(class_folder)
    return np.array(images), np.array(labels)

# Function to add SNR noise to images
def add_noise(images, snr):
    noisy_images = np.zeros_like(images)
    linear_snr = 10**(snr/10)
    for i in range(len(images)):
        avg_power = np.mean(images[i]**2)
        noise_power = avg_power/linear_snr
        noise = np.random.normal(0, np.sqrt(noise_power), images[i].shape)
        img_tmp = images[i].astype('float64')+noise
        img_tmp = np.clip(img_tmp, 0, 255)
        noisy_images[i] = img_tmp.astype('uint8')
    return noisy_images

def add_variable_noise(images, snr_min, snr_max):
    noisy_images = np.zeros_like(images)
    for i in range(len(images)):
        snr = np.random.uniform(snr_min, snr_max)
        linear_snr = 10**(snr/10)
        avg_power = np.mean(images[i]**2)
        noise_power = avg_power/linear_snr
        noise = np.random.normal(0, np.sqrt(noise_power), images[i].shape)
        img_tmp = images[i].astype('float64')+noise
        img_tmp = np.clip(img_tmp, 0, 255)
        noisy_images[i] = img_tmp.astype('uint8')
    return noisy_images

# Load data
images, labels = load_data(ROOT_DIR, NUM_IMAGES)

# Convert labels to one-hot encoding
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
index_to_label = {idx: label for label, idx in label_to_index.items()}
labels = np.array([label_to_index[label] for label in labels])
labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Split data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.25, stratify=labels, random_state=42
)


# SNR range
snr_range = range(-10, 41, 10)

# Initialize lists to store accuracies
accuracies = [[] for _ in range(len(model_paths))]

# Load test images and labels
def load_test_data(root_dir, num_images):
    images = []
    labels = []
    class_folders = os.listdir(root_dir)
    for class_folder in class_folders:
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):
            class_images = os.listdir(class_path)[:num_images]
            for image_name in class_images:
                image_path = os.path.join(class_path, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, IMG_SIZE)
                images.append(image)
                labels.append(class_folder)
    return np.array(images), np.array(labels)

# Load test data
test_images, test_labels = load_test_data(ROOT_DIR, NUM_IMAGES)

# Convert test labels to numerical indices
test_labels_numeric = np.array([label_to_index[label] for label in test_labels])

# Convert test labels to one-hot encoding
test_labels = to_categorical(test_labels_numeric, num_classes=NUM_CLASSES)

# Reshape test images to match model input shape
test_images = test_images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 3)

# Loop over SNR range
for snr in snr_range:
    # Add noise to test images
    test_noisy_images = add_noise(test_images, snr=snr)
    
    # Loop over models
    for i, model_path in enumerate(model_paths):
        # Load model
        model = load_model(model_path)
        
        # Evaluate model on test images
        test_loss, test_acc = model.evaluate(test_noisy_images, test_labels, verbose=0)
        
        # Store accuracy
        accuracies[i].append(test_acc)

plt.figure(figsize=(10, 6))
for i, acc in enumerate(accuracies):
    if i == 0:
        plt.plot(snr_range, acc, label="Trained with Original", color='green')
    elif i == 1:
        plt.plot(snr_range, acc, label="Trained with 10 SNR", color='red')
    elif i == 2:
        plt.plot(snr_range, acc, label="Trained with 0-40 SNR", color='purple')

plt.title('Accuracy vs SNR')
plt.xlabel('SNR (db)')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.grid(True)
plt.show()
