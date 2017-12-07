
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import matplotlib.image as mpimg
from dlproject import get_label_mapping, get_files, bottle_neck_process
slim = tf.contrib.slim


# In[ ]:


save_model_file = './model_file/inception_resnet_v2_for3'
last_layer_file = './middle_data/last_layer3/'
bottle_neck_file = './middle_data/bottle_neck3/'
label_file = './labels.txt'


# In[10]:


def get_last_layer(sess, bottle_neck_file, last_layer_file, label_file, bottle_neck, batch_norm2):
    id2label, label2id = get_label_mapping(label_file)
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
            np.save(des_dir + f[17:],  bottle)
            print('save the bottle neck file ' + f)


# In[11]:


if __name__ == '__main__':
    with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], save_model_file)
            input_ = sess.graph.get_tensor_by_name("Placeholder_2:0")
            output = sess.graph.get_tensor_by_name("my_fine_tune/batch_norm2/batchnorm/add_1:0")
            get_last_layer(sess, bottle_neck_file, last_layer_file, label_file, input_, output)

