import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\my_model.h5"  # Replace with the path to your saved model
model = load_model(model_path)

# Define constants
ROOT_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_train'
IMG_SIZE = (100,100)
BATCH_SIZE = 8
NUM_CLASSES = 29
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

# Convert pixel values to float32 and scale to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Make predictions on the training data
predictions = model.predict(train_images)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Convert one-hot encoded train_labels back to class indices
train_labels_indices = np.argmax(train_labels, axis=1)

# Compare predicted labels with true labels
misclassified_indices = np.where(predicted_labels != train_labels_indices)[0]

for idx in misclassified_indices:
    misclassified_image = train_images[idx]
    true_label = train_labels[idx]
    predicted_label = predicted_labels[idx]
    
    # Print true and predicted labels
    true_label = index_to_label[true_label.argmax()]
    predicted_label = index_to_label[predicted_label]

    # Find an example image of the predicted class
    example_image_path = os.path.join(ROOT_DIR, predicted_label, os.listdir(os.path.join(ROOT_DIR, predicted_label))[0])
    example_image = cv2.imread(example_image_path)

    # Resize images for better visualization
    resized_misclassified_image = cv2.resize(misclassified_image, (200, 200))
    resized_example_image = cv2.resize(example_image, (200, 200))

    # Display the misclassified image
    cv2.imshow("Misclassified Image", resized_misclassified_image)

    # Create a blank canvas to display example image of predicted class
    canvas = np.zeros((300, 500, 3), dtype="uint8")

    # Place example image of predicted class on the canvas
    canvas[20:220, 220:420] = resized_example_image

    # Add true and predicted labels underneath the example image
    cv2.putText(canvas, f"True Label: {true_label}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(canvas, f"Predicted Label: {predicted_label}", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Display the canvas with example image and labels
    cv2.imshow("Example Image of Predicted Class", canvas)

    # Wait until a key is pressed to continue to the next image
    cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()