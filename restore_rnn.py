import tflearn
import tensorflow as tf
from RNN_data_process import get_data, get_train_data


def get_rnn(images, input_size, num_classes):
    
    rnn = tflearn.input_data(shape=[None, images, input_size])
    rnn = tflearn.lstm(rnn, 512, dropout=0.8, return_seq=True)
    rnn = tflearn.lstm(rnn, 512)
    rnn = tflearn.fully_connected(rnn, num_classes, activation='softmax')
    rnn = tflearn.regression(rnn, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return rnn



def load_model(seq_length, input_length, num_classes, model_path):
    rnn = get_rnn(seq_length, input_length, num_classes)
    new_model = tflearn.DNN(rnn)
    new_model.load(model_path,weights_only=True)
    return new_model

