import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Define constants
ROOT_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_train'
IMG_SIZE = (100, 100)
BATCH_SIZE = 8
NUM_CLASSES = 29
NUM_EPOCHS = 10
NUM_IMAGES = 300  # Number of images to load per class

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

# Load your saved model
model_path = "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\my_model.h5"
model = load_model(model_path)

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



# Add noise to test images
SNR = -10  # Adjust SNR value as needed
noisy_test_images = add_noise(test_images, SNR)

# Convert pixel values to float32 and scale to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Evaluate the model on noisy test images
loss, accuracy = model.evaluate(noisy_test_images, test_labels, verbose=1)
print("Test Accuracy with SNR {}: {:.2f}%".format(SNR, accuracy * 100))



# Function to plot images
def plot_images(original_images, noisy_images):
    plt.figure(figsize=(10, 5))
    num_images = min(len(original_images), 4)
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i])
        plt.title('Original')
        plt.axis('off')
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(noisy_images[i])
        plt.title('Noisy (SNR={})'.format(SNR))
        plt.axis('off')
    plt.show()

# Select some test images to visualize
num_visualize = 4
original_images = test_images[:num_visualize]
noisy_images = noisy_test_images[:num_visualize]

# Plot original and noisy images
plot_images(original_images, noisy_images)