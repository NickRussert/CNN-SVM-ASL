import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



# Define constants
ROOT_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_train'
IMG_SIZE = (100, 100)
BATCH_SIZE = 8
NUM_CLASSES = 29
NUM_EPOCHS = 10
NUM_IMAGES = 3000  # Number of images to load per class

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

# Evaluate Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Train Model
history = model.fit(
    train_images, train_labels,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(test_images, test_labels)
)

# Save the trained model
model.save('my_model.h5')


# Evaluate Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Make predictions on test data
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Convert one-hot encoded labels back to categorical labels
true_labels = np.argmax(test_labels, axis=1)
true_labels_categorical = [index_to_label[idx] for idx in true_labels]
predicted_labels_categorical = [index_to_label[idx] for idx in predicted_labels]

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels_categorical, predicted_labels_categorical)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_to_index.keys(), yticklabels=label_to_index.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


#recent additions to visualize training vs validating 

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()