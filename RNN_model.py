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
from RNN_data_process import get_train_data, get_data



def get_rnn(images, input_size, num_classes):
    
    rnn = tflearn.input_data(shape=[None, images, input_size])
    rnn = tflearn.lstm(rnn, 512, dropout=0.8, return_seq=True)
    rnn = tflearn.lstm(rnn, 512)
    rnn = tflearn.fully_connected(rnn, num_classes, activation='softmax')
    rnn = tflearn.regression(rnn, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return rnn


def train(X_train, y_train, X_test, y_test, seq_length, batch_size):
    
    
    input_length = X_train.shape[2]
    num_classes = len(y_train[0])
    
    rnn = get_rnn(seq_length, input_length, num_classes)

    model = tflearn.DNN(rnn, tensorboard_verbose=1)
    model.fit(X_train, y_train, validation_set=(X_test, y_test),
              show_metric=True, batch_size=batch_size, snapshot_step=100,
              n_epoch=3)

    model.save('model_file/checkpoints/rnn.tflearn')
    results =  model.predict_label(X_test)
    
    return results


if __name__ == '__main__':
    input_length = 128 #look at the data and figure this out!
    seq_length = 2
    batch_size = 32
    num_classes = 3
    
    data_root_path = './middle_data/last_layer3/' #will take all the folders in this directory to be used as labels
    label_mapping_path = './labels.txt' #labels.txt should NOT be in the data_root_path
    X_from_image, y_from_image = get_train_data(data_root_path, label_mapping_path)
    
    X_test = []
    y_test =[]
    X_train, X_test, y_train, y_test = get_data(X_from_image, y_from_image, seq_length, num_classes, input_length)
    results = train(X_train, y_train, X_test, y_test, seq_length, batch_size)
    
    # predict the accuracy for each class
    y_true = np.argmax(y_test, axis=1)
    y_pred = results[:,0]
    cmat = confusion_matrix(y_true, y_pred)
    print(cmat.diagonal()/cmat.sum(axis=1))

