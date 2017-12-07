import numpy as np
import scipy.misc
import glob
import sys
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import os
from os.path import basename
import collections
import tflearn

from collections import deque
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix


def get_img_array(path):
    """
    Given path of image, returns it's numpy array
    """
    return scipy.misc.imread(path)

def check_files(folder):
    """
    Given path to folder, returns whether it's empty or not
    """
    filenames = [file for file in glob.glob(folder+'*/*')]
    if(len(filenames)==0):
        return False
    return True

def get_files(folder):
    """
    Given path to folder, returns list of files in it
    """
    filenames = [file for file in glob.glob(folder+'*/*')]
    filenames.sort()
    return filenames

def get_label(label2id, label):
    """
    Returns label for a folder
    """
    if label in label2id:
        return label2id[label]
    else:
        sys.exit("Invalid label: " + label)


# Functions to load data, DO NOT change these

def get_labels(folder, label2id):
    """
    Returns vector of labels extracted from filenames of all files in folder
    :param folder: path to data folder
    :param label2id: mapping of text labels to numeric ids. (Eg: automobile -> 0)
    """
    files = get_files(folder)
    y = []
    for f in files:
        y.append(get_label(f,label2id))
    return np.array(y)

def one_hot(y, num_classes=10):
    """
    Converts each label index in y to vector with one_hot encoding
    """
    y_one_hot = np.zeros((y.shape[0], num_classes))
    y_one_hot[y] = 1
    return y_one_hot.T

def get_label_mapping(label_file):
    """
    Returns mappings of label to index and index to label
    The input file has list of labels, each on a separate line.
    """
    with open(label_file, 'r') as f:
        id2label = f.readlines()
        id2label = [l.strip() for l in id2label]
    label2id = {}
    count = 0
    for label in id2label:
        label2id[label] = count
        count += 1
    return id2label, label2id

def get_images(folder):
    """
    returns numpy array of all samples in folder
    each column is a sample resized to 30x30 and flattened
    """
    files = get_files(folder)
    images = []
    count = 0
    
    for f in files:
        count += 1
        if count % 10000 == 0:
            print("Loaded {}/{}".format(count,len(files)))
        img_arr = get_img_array(f)
        img_arr = img_arr.flatten() / 255.0
        images.append(img_arr)
    X = np.column_stack(images)
    return X

def get_image_map(folder, label):
    """
    returns numpy array of all samples in folder
    each column is a sample resized to 30x30 and flattened
    """
    image_map = {}
    files = get_files(folder)
    images = []
    count = 0
    
    for f in files:
        count += 1
        if count % 10000 == 0:
            print("Loaded {}/{}".format(count,len(files)))
        #img_arr = get_img_array(f) / 255.0
        #img_arr = get_img_array(f)
        #img_arr = img_arr.flatten() / 255.0
        img_arr = np.load(f)
        img_arr = img_arr.flatten()
        name = basename(f)
        #print(name)
        num = int(name.split("_")[1].split(".")[0])
        image_map[num] = [img_arr, label]
        #images.append(img_arr)
    #X = np.column_stack(images)
    return image_map

def get_train_data(data_root_path, label_mapping_path):
    """
    Return X and y
    """
    X = []
    y = []
    labels = os.listdir(data_root_path)
    id2label, label2id = get_label_mapping(label_mapping_path)
    print(label2id)
    image_map = {}
    for label in labels:
        train_data_path = data_root_path + label
        if(check_files(train_data_path)) :
            temp_map = get_image_map(train_data_path, label)
            image_map.update(temp_map)
    new_map = collections.OrderedDict(sorted(image_map.items()))
    for k, v in new_map.items():
        X.append(v[0])
        y.append(get_label(label2id, v[1]))
    X = np.array(X)
    return X, np.array(y)

def save_predictions(filename, y):
    """
    Dumps y into .npy file
    """
    np.save(filename, y)# Load the data



def get_data(X_from_image, y_from_image, num_frames, num_classes, input_length):
    X = []
    y = []
    #print(X_from_image.shape)
    #print("hi")
    image_seq = deque()
    #y_list = deque()
    for row in range(0,len(X_from_image)):
        image_seq.append(X_from_image[row])
        #y_list.append(y_from_image[row])
        if len(image_seq) == num_frames:
            X.append(np.array(list(image_seq)))
            y.append(y_from_image[row])
            #y.append(np.array(list(y_list)))
            image_seq.popleft()
            #y_list.popleft()
    
    X = np.array(X)
    y = np.array(y)
    print(X.shape)
    y = to_categorical(y, num_classes)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test
    