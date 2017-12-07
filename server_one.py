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

charpre='p'
accelerate=0
save_model_file = './model_file/inception_resnet_v2_for3'
softmax_tensor_name = "my_fine_tune/logits/BiasAdd:0"
input_name = "Placeholder:0"
bottle_neck_name = "Placeholder_2:0"
before_logit_name = "InceptionResnetV2/Logits/Dropout/Identity:0"

def run_inference(images, out_file, labels, model_file, sess, softmax_tensor, before_logit, k=2):
    answer = None

    # Creates graph from saved GraphDef.
    if out_file:
        out_file = open(out_file, 'wb', 1)

    bottle_num = sess.run(before_logit,feed_dict = {input_name: images})
    predictions = sess.run(softmax_tensor, feed_dict = {bottle_neck_name: bottle_num})
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-k:][::-1]  # Getting top k predictions

    return labels[top_k[0]]

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
                        
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], save_model_file)
        softmax_tensor = sess.graph.get_tensor_by_name(softmax_tensor_name)
        before_logit = sess.graph.get_tensor_by_name(before_logit_name)
                    
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


            predictedlabel=run_inference(images=images, out_file=args['out'],
                                         labels=labels,model_file=args['model'], 
                                         sess=sess, softmax_tensor=softmax_tensor, before_logit=before_logit)

            print(predictedlabel)
            char=predictedlabel 

            c.send(char.encode())
            c.close()                # Close the connection




