import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Define constants
ROOT_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_train'
IMG_SIZE = (100, 100)
BATCH_SIZE = 8
NUM_CLASSES = 29
NUM_EPOCHS = 10
NUM_IMAGES = 100  # Number of images to load per class

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

# Add noise to training images with fixed SNR
train_noisy_images = add_noise(train_images, snr=10)


# Model Architecture for fixed SNR
model_fixed_snr = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile Model
model_fixed_snr.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# Train Model with fixed SNR
history_fixed_snr = model_fixed_snr.fit(
    train_noisy_images, train_labels,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(test_images, test_labels)
)

# Save the trained model with fixed SNR
model_fixed_snr.save('model_fixed_snr.h5')

# Add variable noise to training images
train_variable_noisy_images = add_variable_noise(train_images, snr_min=0, snr_max=40)

# Model Architecture for variable SNR
model_variable_snr = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile Model
model_variable_snr.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

# Train Model with variable SNR
history_variable_snr = model_variable_snr.fit(
    train_variable_noisy_images, train_labels,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(test_images, test_labels)
)

# Save the trained model with variable SNR
model_variable_snr.save('model_variable_snr.h5')
