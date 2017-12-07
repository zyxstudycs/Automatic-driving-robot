import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import matplotlib.image as mpimg
import matplotlib
from CNN_data_preprocess import get_label_mapping, get_files, bottle_neck_process
slim = tf.contrib.slim



checkpoint_file = './model_file/inception_resnet_v2_2016_08_30.ckpt'
label_file = './labels.txt'
train_file = './Data3_4/'
bottle_neck_file = './middle_data/bottle_neck3_4/'
save_model_file = './model_file/inception_resnet_v2_for3_4'
last_layer_file = './middle_data/last_layer3_4/'



def build_graph():
    images = tf.placeholder(tf.float32, shape=(None, 480, 720, 3))
    labels = tf.placeholder(tf.int32, shape=(None,))

    # restore from the inception_resnet model
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2(images, num_classes = 1001, is_training = False) 

    # extract the before_logit tensor from the extracted model
    before_logit = end_points['PreLogitsFlatten']


    # define a placeholder for training from the bottle, not from the beginning.
    bottle_neck = tf.placeholder(tf.float32, shape=(None,1536))

    # now define the fine tune part of the graph
    with tf.name_scope('my_fine_tune') as scope:
    #     dropout = tf.layers.dropout(bottle_neck, rate=0.3, name='dropout')
        batch_norm1 = tf.layers.batch_normalization(bottle_neck, name='batch_norm1')
        dropout1 = tf.layers.dropout(batch_norm1, rate=0.3, name='dropout1')
        dense1 = tf.layers.dense(dropout1, 128, activation=tf.nn.relu, name='dense1')
        batch_norm2 = tf.layers.batch_normalization(dense1, name='batch_norm2')
        dropout2 = tf.layers.dropout(batch_norm2, rate = 0.3, name='dropout2')
        logits = tf.layers.dense(dropout2, 4, name='logits')

        x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(x_entropy, name='loss')
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # extract variables that need to be trained
        weight1, bias1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dense1')
        weight2, bias2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'logits')
        variables_to_train = [weight1, weight2, bias1, bias2]

        optimizer = tf.train.AdamOptimizer()
        train_op = slim.learning.create_train_op(loss, optimizer, variables_to_train=variables_to_train)

    # defiine the name scope we don't need to restore from inception_resnet
    exclude = ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits',
               'my_fine_tune/','dense1','logits','batch_norm']

    variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
    saver = tf.train.Saver(variables_to_restore)
    init = tf.global_variables_initializer()
    builder = tf.saved_model.builder.SavedModelBuilder(save_model_file)

    return images, labels, before_logit, bottle_neck,batch_norm2, loss, accuracy, train_op, saver, init, builder



# extract the data before the fine tune layer
def get_bottle_neck_data(sess, before_logit, images, train_file, bottle_neck_file, label_file, hsv = False):
    id2label, label2id = get_label_mapping(label_file)
    for i in id2label:
        read_dir = train_file + i
        des_dir = bottle_neck_file + i + '/'
        os.mkdir(des_dir)
        
        files = get_files(read_dir)
        for f in files:
            img=mpimg.imread(f).reshape(1,480,720,3) / 255
            if(hsv):
                img = matplotlib.colors.rgb_to_hsv(img)
            bottle = sess.run(before_logit, feed_dict={images:img})
            np.save(des_dir + f[len(read_dir):],  bottle)
            print('save the bottle neck file ' + f)
            

def get_last_layer(sess, bottle_neck_file, last_layer_file, label_file, bottle_neck, batch_norm2):
    id2label, label2id = get_label_mapping(label_file)
    os.mkdir(last_layer_file)
    print(last_layer_file)
    for i in id2label:
        read_dir = bottle_neck_file + i
        des_dir = last_layer_file + i + '/'
        print(des_dir)
        os.mkdir(des_dir)
        
        files = get_files(read_dir)
        for f in files:
            img=np.load(f)
            print(f)
            bottle = sess.run(batch_norm2, feed_dict={bottle_neck:img})
            np.save(des_dir + f[len(read_dir):],  bottle)
            print('save the bottle neck file ' + f)


def train_graph(sess, model_list):
    images, labels, before_logit, bottle_neck, batch_norm2, loss, accuracy, train_op, saver, init, builder = model_list
    
    sess.run(init)
    saver.restore(sess, checkpoint_file)

    # if there is no dir for the bottle data, then get the data first.
    if(os.path.isdir(bottle_neck_file) is not True):
        os.mkdir(bottle_neck_file)
        get_bottle_neck_data(sess, before_logit, images, train_file, bottle_neck_file, label_file, hsv = False)

    # get training data from bottle_neck file, and train.
    bottle_neck_processor = bottle_neck_process(bottle_neck_file, label_file) 
    for i in range(2000):
        inputs, groud_truth = bottle_neck_processor.next_batch(512)
        train_loss, train_accuracy = sess.run([train_op, accuracy], feed_dict={bottle_neck: inputs, 
                                                                             labels: groud_truth})

        if(i % 50 == 0):
            val_loss, val_accuracy = sess.run([loss, accuracy],
                                              feed_dict={bottle_neck: bottle_neck_processor.val_inputs,
                                                         labels: bottle_neck_processor.val_labels})
        
            print('####################################')
            print('the training loss after ' + str(i) + ' iterations is: ' + str(train_loss))
            print('the training accuracy after ' + str(i) + ' iterations is: ' + str(train_accuracy))
            print('the val loss after ' + str(i) + ' iterations is: ' + str(val_loss))
            print('the val accuracy after ' + str(i) + ' iterations is: ' + str(val_accuracy))
            print('-------------')
            
            for i in bottle_neck_processor.id2label:
                val_accuracy = sess.run([accuracy],
                                      feed_dict={bottle_neck: bottle_neck_processor.val_inputs_each[i],
                                                 labels: bottle_neck_processor.val_labels_each[i]})
                print(i + ': the val accuracy is: '  + str(val_accuracy))
        
#         after the training process finished, store the model to a '.pb' version file.
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING],
                                         signature_def_map=None, assets_collection=None)
    
    builder.save()



if __name__ == '__main__':
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        model_list = build_graph()
        train_graph(sess, model_list)

