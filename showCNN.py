from tensorflow import keras
import matplotlib.pyplot as plt

def visualize_filters_and_features(image_path, model_path):
  # Load the image
  img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))  # Adjust target size as needed
  image = keras.preprocessing.image.img_to_array(img)
  image = image / 255.0  # Normalize (assuming image has 0-255 values)
  image = image.reshape((1, *image.shape))  # Add batch dimension

  # Load the model
  model = keras.models.load_model(model_path)

  # Get the first convolutional layer (assuming first layer is convolutional)
  conv_layer = model.layers[0]

  # Extract filters (weights) from the layer
  filters = conv_layer.get_weights()[0]

  # Visualize filters
  num_filters = filters.shape[3]
  for i in range(num_filters):
    plt.imshow(filters[:, :, :, i], cmap="gray")
    plt.title(f"Filter {i+1}")
    plt.show()

  # Run the image through the first layer to get feature maps
  feature_maps = model.predict(image)

  # Visualize feature maps (assuming grayscale image)
  for i in range(feature_maps.shape[3]):
    plt.imshow(feature_maps[0, :, :, i], cmap="gray")
    plt.title(f"Feature Map Channel {i+1}")
    plt.show()

# Example usage
image_path = "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_test\\asl_alphabet_test\\F_test.jpg"
model_path = "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\my_model.h5"
visualize_filters_and_features(image_path, model_path)
