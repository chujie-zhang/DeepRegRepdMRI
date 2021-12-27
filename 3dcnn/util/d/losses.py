import tensorflow as tf
import numpy as np

def similarity(fixed,moving):
    moving_mean2 = tf.reduce_mean(moving,axis=2,keep_dims=True)
    fixed_mean2 = tf.reduce_mean(fixed,axis=2,keep_dims=True)
    moving_mean3 = tf.reduce_mean(moving,axis=3,keep_dims=True)
    fixed_mean3 = tf.reduce_mean(fixed,axis=3,keep_dims=True)

    theta = fixed_mean2 - moving_mean2
    phi = fixed_mean3 - moving_mean3
    return theta, phi


def mse_loss(label_fixed, label_moving,label_flag_moving,label_flag_fixed):


    one_batch = tf.reduce_mean(tf.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4])
    label_loss_batch = tf.reduce_mean(one_batch * label_flag_moving * label_flag_fixed)

    return label_loss_batch
