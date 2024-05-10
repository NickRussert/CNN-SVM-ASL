import numpy as np
import cv2

LABEL_DEFS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def import_images(num_training, num_testing, data_path):
    total_images = 3000
    num_labels = len(LABEL_DEFS)
    training_images = np.zeros((num_training*num_labels, 200, 200))
    training_labels = np.zeros((num_training*num_labels))
    testing_images = np.zeros((num_testing*num_labels, 200, 200))
    testing_labels = np.zeros((num_testing*num_labels))

    shuffled = np.random.permutation(total_images)

    training_i = shuffled[:num_training]
    testing_i = shuffled[num_training:num_training+num_testing]

    count = 0
    for p in training_i:
        filename = data_path+"asl_alphabet_train\\"
        for i in range(num_labels):
            filename_tmp = filename + LABEL_DEFS[i]+"\\"+LABEL_DEFS[i]+str(p)+".jpg"
            img = cv2.imread(filename_tmp, cv2.IMREAD_GRAYSCALE)
            training_images[count] = img
            training_labels[count] = i
            count += 1

    count = 0
    for p in testing_i:
        filename = data_path+"asl_alphabet_train\\" # use extra training data for testing
        for i in range(num_labels):
            filename_tmp = filename + LABEL_DEFS[i]+"\\"+LABEL_DEFS[i]+str(p)+".jpg"
            img = cv2.imread(filename_tmp, cv2.IMREAD_GRAYSCALE)
            testing_images[count] = img
            testing_labels[count] = i
            count += 1
    return training_images.astype('uint8'), training_labels, testing_images.astype('uint8'), testing_labels

def calculate_clusters(images, cluster_size):
    # calculate bow cluster to use as basis for feature extraction
    sift = cv2.SIFT_create()

    bow_trainer = cv2.BOWKMeansTrainer(cluster_size)

    count = 1
    for img in images:
        print('Clustering: '+str(np.round(count/len(images)*100))+"%", end='\r')
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            bow_trainer.add(des)
        count += 1
    print()
    return bow_trainer.cluster()

def extract_features(cluster, images, labels):
    # use bow cluster from training data to extract feature vector for each image
    sift = cv2.SIFT_create()
    bow_extract = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))

    bow_extract.setVocabulary(cluster)

    features = []
    new_labels = []
    count = 1
    for img in images:
        print('Extracting: '+str(np.round(count/len(images)*100))+"%", end='\r')
        histogram = bow_extract.compute(img, sift.detect(img))
        if histogram is not None:
            features.append(histogram[0])
            new_labels.append(labels[count-1])
        count += 1
    print()
    return np.array(features), np.array(new_labels)

