# Imports for Deep Learning
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from keras.models import Sequential, load_model
import os
import matplotlib.pyplot as plt
#from keras.preprocessing.image import ImageDataGenerator

# Ensure consistency across runs
from numpy.random import seed
import random
seed(2)
#from tensorflow import set_random_seed
#set_random_seed(2)

# Imports to view data
import cv2
from glob import glob

# Metrics
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
#from keras.utils import print_summary
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

# Utils
from pathlib import Path
import pandas as pd
import numpy as np
from os import getenv
import time
import itertools

# Image Preprocessing
from skimage.filters import sobel, scharr

#paths to training and testing directories
TRAIN_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_train'
TEST_DIR = 'C:\\Users\\russe\\OneDrive\\Desktop\\549Project\\asl_alphabet_test'

# Get the classes (folders) from the training directory
#this is A-Z + del, nothing, space
CLASSES = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]
CLASSES.sort()


def plot_one_sample_of_each(base_path):
    cols = 5
    rows = int(np.ceil(len(CLASSES) / cols))
    fig = plt.figure(figsize=(16, 20))
    
    for i in range(len(CLASSES)):
        cls = CLASSES[i]
        img_path = base_path + '/' + cls + '/**'
        path_contents = glob(img_path)
    
        imgs = random.sample(path_contents, 1)

        sp = plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.imread(imgs[0]))
        plt.title(cls)
        sp.axis('off')

    plt.show()
    return

plot_one_sample_of_each(TRAIN_DIR)

# Get the classes (folders) from the training directory
CLASSES = os.listdir(TRAIN_DIR)
CLASSES.sort()

# Count the number of samples in each class
samples_per_class = {}
for cls in CLASSES:
    class_dir = os.path.join(TRAIN_DIR, cls)
    num_samples = len(os.listdir(class_dir))
    samples_per_class[cls] = num_samples

# Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(samples_per_class.keys(), samples_per_class.values())
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Number of Samples per Class in Training Data')
plt.xticks(rotation=45)
plt.show()