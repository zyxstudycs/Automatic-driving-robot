get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import sys
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import os


def get_img_array(path):
    """
    Given path of image, returns it's numpy array
    """
    return scipy.misc.imread(path)

def check_files(folder):
    """
    Given path to folder, returns list of files in it
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


# In[4]:


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

def get_train_data(data_root_path, label_mapping_path):
    """
    Return X and y
    """
    X = []
    y = []
    labels = os.listdir(data_root_path)
    id2label, label2id = get_label_mapping(label_mapping_path)
    print(label2id)
    for label in labels:
        train_data_path = data_root_path + label
        if(check_files(train_data_path)) :
            X_temp = get_images(train_data_path)
            size = X_temp[0].size
            y.extend(np.full((size), get_label(label2id, label)))
            if(len(X)==0):
                X = X_temp
            else:
                X = np.concatenate((X,X_temp), axis = 1)

    return X, np.array(y)


def save_predictions(filename, y):
    """
    Dumps y into .npy file
    """
    np.save(filename, y)# Load the data


# In[17]:


#bottle_neck_file = './bottle_neck3/'
#label_file = './labels.txt'


# In[114]:


class bottle_neck_process(object):
    def __init__(self, bottle_neck_file, label_file):
        self.bottle_neck_file = bottle_neck_file
        self.label_file = label_file
        self.id2label, self.label2id = self.get_label_mapping(self.label_file)
        self.val_inputs_each = None
        self.val_labels_each = None
        self.val_inputs = None
        self.val_labels = None
        self.train_inputs = None
        self.train_labels = None
        self.train_number = None
        self.split_data()
        self.get_val_each()
        
    # return: data_map: {label: matrix, label:matrix..}   label_map: {label: vector, label: vector...} 
    def get_data(self):
        data_map = {}
        label_map = {}
        data_number = {}
        for i in self.id2label:
            read_dir = self.bottle_neck_file + i
            files = get_files(read_dir)
            data_number[i] = len(files)
            print(read_dir)
            print(len(files))
            
            inputs = []
            labels = []
            for f in files:
                inputs.append(np.load(f))
                labels.append(self.label2id[i])
            print('the lengeh of inputs is ' + str(len(inputs)))
            data_map[i] = np.reshape(np.asarray(inputs), (len(files), -1))
            label_map[i] = np.reshape(np.asarray(labels), -1)

        return data_map, label_map, data_number
    
    # split the data to training set and validation set. 80% to train, 20% to test(val)
    # 
    def split_data(self):      
        data_map, label_map, data_number = self.get_data()

        train_inputs = {}
        train_labels = {}
        train_number = {}
        val_inputs = np.array([])
        val_labels = np.array([])

        dimen = None
        total_train_number = 0
        for i in self.id2label:
            init = data_map[i]
            label = label_map[i]
            number = int(data_number[i] * 0.2)

            index = np.random.choice(data_number[i], number, replace=False)
            val_input = init[index, :]
            val_label = label[index]
            train_input = np.delete(init, index, axis = 0)
            train_label = np.delete(label, index)

            dimen = train_input.shape[1]
            val_inputs = np.append(val_input, val_inputs.reshape(-1, dimen), axis = 0)
            val_labels = np.append(val_label, val_labels, axis = 0)
            train_inputs[i] = train_input
            train_labels[i] = train_label
            train_number[i] = train_input.shape[0]
            total_train_number = total_train_number + train_input.shape[0]

        val_inputs = np.reshape(np.asarray(val_inputs), (-1, dimen))
        val_labels = np.reshape(np.asarray(val_labels), -1)
        
        self.val_inputs = val_inputs
        self.val_labels = val_labels
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.train_number = train_number
        self.total_train_number = total_train_number

    def next_batch(self, mini_batch):
        inputs = np.array([])
        labels = np.array([])
        
        for i in self.id2label:
            input_ = self.train_inputs[i]
            label = self.train_labels[i]
            num = int(self.train_number[i] / self.total_train_number * mini_batch)
            dimen = input_.shape[1]
            
            index = np.random.choice(self.train_number[i], num, replace=True)
            inputs_temp = input_[index, :]
            labels_temp = label[index]
            
            inputs = np.append(inputs_temp, inputs.reshape(-1, dimen), axis = 0)
            labels = np.append(labels_temp, labels, axis = 0)
            
        inputs = np.reshape(np.asarray(inputs), (-1, dimen))
        labels = np.reshape(np.asarray(labels), -1)
        
        return inputs, labels
        
    
    def get_label_mapping(self, label_file):
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

    def get_val_each(self):
        val_inputs_each = {}
        val_labels_each = {}
        for i in self.id2label:
            num = self.label2id[i]
            mask = (self.val_labels == num)
            input_ = self.val_inputs[mask,:]
            output_ = self.val_labels[mask]
            val_inputs_each[i] = input_
            val_labels_each[i] = output_
            
        self.val_inputs_each = val_inputs_each
        self.val_labels_each = val_labels_each