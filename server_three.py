import socket               # Import socket module
import time
import sys  #Used for closing the running program
import os, os.path
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import scipy.misc
from restore_rnn import load_model
from collections import deque

charpre='p'
accelerate=0

# set parameters for load CNN model
save_model_file = './model_file/inception_resnet_v2_for4'
softmax_tensor_name = "my_fine_tune/logits/BiasAdd:0"
input_name = "Placeholder:0"
bottle_neck_name = "Placeholder_2:0"
before_logit_name = "InceptionResnetV2/Logits/Dropout/Identity:0"
batch_norm2_name = "my_fine_tune/batch_norm2/batchnorm/add_1:0"

# set parameters for load RNN model
seq_length = 2
input_length = 128 #look at the data and figure this out!
num_classes = 3
model_path = './model_file/checkpoints/RNN_data4/RNN_data4'

data_queue = deque()

def run_inference(rnn_model,images, out_file, labels, model_file, sess, batch_norm2, softmax, before_logit, k=2):

    # Creates graph from saved GraphDef.
    if out_file:
        out_file = open(out_file, 'wb', 1)

    bottle_num = sess.run(before_logit,feed_dict = {input_name: images})
    predictions = sess.run(batch_norm2, feed_dict = {bottle_neck_name: bottle_num})
    
    if (len(data_queue)>=seq_length):
        data_queue.popleft()
    data_queue.append(predictions)
    input_for_rnn = np.array(list(data_queue))
    
    
    if (len(data_queue) < seq_length):
        predictions= sess.run(softmax, feed_dict = {bottle_neck_name: bottle_num})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-k:][::-1]  # Getting top k predictions
        result = labels[top_k[0]]
    else:
        result = rnn_model.predict_label(input_for_rnn.reshape(1,seq_length,128))
        print(result)
        result = labels[result[0][0]]
    
    return result

def get_image(imagename):
    return scipy.misc.imread(imagename).reshape(1,480, 720, 3)/255

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify Image(s)')
    parser.add_argument('-li','--list', help='List File having input image paths')
    parser.add_argument('-o','--out', help='Output file for storing the content')
    parser.add_argument('-m','--model', help='model file path (protobuf)', required=True)
    parser.add_argument('-l','--labels', help='labels text file', required=True)
    parser.add_argument('-r','--root', help='path to root directory of input data')
    args = vars(parser.parse_args())
    iteration=0
    s = socket.socket()         # Create a socket object
 # host = socket.gethostname() # Get local machine name
    port = 12344                 # Reserve a port for your service.
    s.bind(('', port))        # Bind to the port
    f = open('torecv.jpg','wb')
                        
    rnn_model = load_model(seq_length, input_length, num_classes,model_path)
    print('rnn_model is restored')
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], save_model_file)
        print('cnn model is restored')
        softmax_tensor = sess.graph.get_tensor_by_name(softmax_tensor_name)
        before_logit = sess.graph.get_tensor_by_name(before_logit_name)
        batch_norm2 = sess.graph.get_tensor_by_name(batch_norm2_name)
        
                    
        while True:
            # Read input
            f = open('torecv.jpg','wb')
            print('listening')
            s.listen(5)      
            c, addr = s.accept()
            print('Got connection from', addr)
            print("Receiving...")
            l = c.recv(1024)
            while (l):
                f.write(l)
                l = c.recv(1024)
            f.close()
            print("Done Receiving")


            imagename='./torecv.jpg'
            images = get_image(imagename)
            # if a separate root directory given then make a new path
            if args['root']:
                    print("Input data from  : %s" % args['root'])
                    images = map(lambda p: os.path.join(args['root'], p), images)

            with open(args['labels'], 'rb') as f:
                    labels = [str(w).replace("\n", "") for w in f.readlines()]


            predictedlabel=run_inference(rnn_model,images=images, out_file=args['out'],
                                         labels=labels,model_file=args['model'], 
                                         sess=sess, batch_norm2=batch_norm2, softmax=softmax_tensor, before_logit=before_logit)

            print(predictedlabel)
            char=predictedlabel 

            c.send(char.encode())
            c.close()                # Close the connection




