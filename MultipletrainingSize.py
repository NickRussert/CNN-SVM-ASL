import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define constants
ROOT_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_train'
IMG_SIZE = (100, 100)
BATCH_SIZE = 8
NUM_CLASSES = 29
NUM_EPOCHS = 10
TRAINING_SIZES = [0, 10, 25, 50, 100, 250, 500, 1000, 2250]

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
images, labels = load_data(ROOT_DIR, max(TRAINING_SIZES))

# Convert labels to one-hot encoding
label_to_index = {label: idx for idx, label in enumerate(np.unique(labels))}
index_to_label = {idx: label for label, idx in label_to_index.items()}
labels = np.array([label_to_index[label] for label in labels])
labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Split data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.25, stratify=labels, random_state=42
)

# Convert pixel values to float32 and scale to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Model Architecture
model = models.Sequential([
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
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Iterate over different training sizes
train_accuracies = []
for size in TRAINING_SIZES:
    print(f"Training model with {size} images")
    
    # Select subset of data based on current training size
    train_images_subset = train_images[:size]
    train_labels_subset = train_labels[:size]
    
    # Train model
    history = model.fit(
        train_images_subset, train_labels_subset,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(test_images, test_labels),
        verbose=0
    )
    
    # Evaluate model on full test set
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    train_accuracies.append((size, test_acc))

# Plotting
sizes, accuracies = zip(*train_accuracies)
plt.plot(sizes, accuracies, marker='o')
plt.title('Accuracy vs Training Size')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.show()
