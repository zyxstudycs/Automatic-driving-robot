import tensorflow as tf
from CNN_model import get_last_layer


bottle_neck_file = './middle_data/bottle_neck3_4_hsv/'
last_layer_file = './middle_data/last_layer3_4_hsv/'
save_model_file = './model_file/inception_resnet_v2_for3_4_hsv'
label_file = './labels.txt'


bottle_neck_name = "Placeholder_2:0"
batch_norm2_name = "my_fine_tune/batch_norm2/batchnorm/add_1:0"


if __name__ == '__main__':
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], save_model_file)
        bottle_neck = sess.graph.get_tensor_by_name(bottle_neck_name)
        batch_norm2 = sess.graph.get_tensor_by_name(batch_norm2_name)
        get_last_layer(sess, bottle_neck_file, last_layer_file, label_file, bottle_neck, batch_norm2)
    

