import os
import cv2
import CNN_Self
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model_path = "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\my_model.h5"  
model = load_model(model_path)

# Define the path to the directory containing images
#image_dir = "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_test\\asl_alphabet_test"

# Visualize the model architecture
print("Model Summary:")
model.summary()

'''
# Get the total number of trainable parameters
total_params = model.count_params()
print(f"Total Trainable Parameters: {total_params}")

# Function to preprocess an image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    # Resize to match model input shape
    img = cv2.resize(img, (100, 100))  # Assuming your model input shape is (50, 50, 3)
    # Preprocess the image (e.g., normalization)
    img = img / 255.0  # Normalize pixel values
    # Return preprocessed image
    return img

# Iterate through images in the directory
predictions = []
for filename in os.listdir(image_dir):
    # Construct the full path to the image
    img_path = os.path.join(image_dir, filename)
    # Preprocess the image
    img = preprocess_image(img_path)
    if img is None:
        continue  # Skip to the next image if preprocessing failed
    # Reshape the image to match model input shape (if needed)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    # Make prediction
    prediction = model.predict(img)
    # Append prediction to list of predictions
    predictions.append(prediction)

# Convert predictions to a numpy array
predictions = np.array(predictions)

# Print predictions
print(predictions)

# Function to get class labels
def get_class_labels():
    # Define your class labels here
    return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Iterate through images in the directory
for filename in os.listdir(image_dir):
    # Construct the full path to the image
    img_path = os.path.join(image_dir, filename)
    # Preprocess the image
    img = preprocess_image(img_path)
    if img is None:
        continue  # Skip to the next image if preprocessing failed
    # Reshape the image to match model input shape (if needed)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    # Make prediction
    prediction = model.predict(img)
    # Get class labels
    class_labels = get_class_labels()
    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(class_labels, prediction[0])
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    '''