import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
# from tensorflow.contrib import layers as contrib_layers  # for furture release
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, BatchNormalization
import tensorflow_addons as tfa
# variables
def var_conv_kernel(ch_in, ch_out, k_conv=None, initialiser=None, name='W'):
    with tf.variable_scope(name):
        if k_conv is None:
            k_conv = [3, 3, 3]
        if initialiser is None:
            #initialiser = tf.contrib.layers.xavier_initializer()
            initialiser = tf.truncated_normal_initializer()
        return tf.get_variable(name, shape=k_conv+[ch_in]+[ch_out], initializer=initialiser)

# blocks
def conv3_block(input_, ch_in, ch_out, k_conv=None, strides=None, name='conv3_block'):
    if strides is None:
        strides = [1, 1, 1, 1, 1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        w = var_conv_kernel(ch_in, ch_out, k_conv)
        #return tf.nn.relu(tfa.layers.InstanceNormalization()(tf.nn.conv3d(input_, w, strides, "SAME"),True))
        return tfa.layers.InstanceNormalization()(tf.nn.conv3d(input_, w, strides, "SAME"))




def downsample_resnet_block(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, k_pool=[1, 2, 2, 2, 1], name='down_resnet_block'):
    if k_conv0 is None:
        k_conv0 = [1, 1, 1]
    strides1 = [1, 1, 1, 1, 1]
    #strides2 = [1, 2, 2, 2, 1]
    strides2 = [1, 2, 1, 1, 1]
    with tf.variable_scope(name):
        h0 = conv3_block(input_, ch_in, ch_out, k_conv0, name='W0')
        r1 = conv3_block(h0, ch_out, ch_out, name='WR1')
        wr2 = var_conv_kernel(ch_out, ch_out)

        #r2 = tf.nn.relu(BatchNormalization()(tf.nn.conv3d(r1, wr2, strides1, "SAME"),True) + h0)
        r2 = tf.nn.relu(tfa.layers.InstanceNormalization()(tf.nn.conv3d(r1, wr2, strides1, "SAME")) + h0)
        if use_pooling:
            #k_pool = [1, 2, 2, 2, 1]
 
            h1 = tf.nn.max_pool3d(r2, k_pool, strides2, padding="SAME")
        else:
            w1 = var_conv_kernel(ch_out, ch_out)
            h1 = conv3_block(r2, w1, strides2, name='W1')
        return h1, h0
    
    
    
def downsample_block(input_, ch_in, ch_out, k_conv0=None, use_pooling=True, k_pool=[1, 2, 2, 2, 1], name='down_resnet_block'):
    if k_conv0 is None:
        k_conv0 = [3, 3, 3]
    strides2 = [1, 2, 1, 1, 1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        h = conv3_block(input_, ch_in, ch_out, k_conv0, name='W0')
        h =  tf.nn.relu(h)
        h = tf.nn.max_pool3d(h, k_pool, strides2, padding="SAME")
    return h



def resize_volume(image, size, method=0, name='resize_volume'):
    # size is [depth, height width]
    # image is Tensor with shape [batch, depth, height, width, channels]
    shape = image.get_shape().as_list()
    with tf.variable_scope(name):
        reshaped2d = tf.reshape(image, [-1, shape[2], shape[3], shape[4]])
        resized2d = tf.image.resize_images(reshaped2d, [size[1], size[2]], method)
        reshaped2d = tf.reshape(resized2d, [shape[0], shape[1], size[1], size[2], shape[4]])
        permuted = tf.transpose(reshaped2d, [0, 3, 2, 1, 4])
        reshaped2db = tf.reshape(permuted, [-1, size[1], shape[1], shape[4]])
        resized2db = tf.image.resize_images(reshaped2db, [size[1], size[0]], method)
        reshaped2db = tf.reshape(resized2db, [shape[0], size[2], size[1], size[0], shape[4]])
        return tf.transpose(reshaped2db, [0, 3, 2, 1, 4])

def fully_connected(input_, length_out, initial_bias_global=0.0, name='fully_connected'):
    initial_std_global = 0.0
    input_size = input_.shape.as_list()
    with tf.variable_scope(name):
        w = var_projection([input_size[1]*input_size[2]*input_size[3]*input_size[4], length_out],
                           initialiser=tf.random_normal_initializer(0, initial_std_global))
        b = var_bias([1, length_out], initialiser=tf.constant_initializer(initial_bias_global))
        return tf.matmul(tf.reshape(input_, [input_size[0], -1]), w) + b

def var_projection(shape_, initialiser=None, name='P'):
    with tf.variable_scope(name):
        if initialiser is None:
            initialiser = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape=shape_, initializer=initialiser)

def var_bias(b_shape, initialiser=None, name='b'):
    with tf.variable_scope(name):
        if initialiser is None:
            initialiser = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name, shape=b_shape, initializer=initialiser)
    
