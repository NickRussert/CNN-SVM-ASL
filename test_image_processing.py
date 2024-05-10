import processing

num_training = 1
num_testing = 1
data_path = "C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\"


training_images, training_labels, testing_images, testing_labels = processing.import_images(num_training, num_testing, data_path)

print(training_images.shape)
print(testing_images.shape)

cluster_size = 100
cluster = processing.calculate_clusters(training_images, cluster_size)
print(cluster.shape)

training_features = processing.extract_features(cluster, training_images, training_labels)
print(training_features)