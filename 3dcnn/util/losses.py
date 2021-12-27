#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf

def similarity(fixed,moving):
    moving_mean2 = tf.reduce_mean(moving,axis=2,keep_dims=True)
    fixed_mean2 = tf.reduce_mean(fixed,axis=2,keep_dims=True)
    moving_mean3 = tf.reduce_mean(moving,axis=3,keep_dims=True)
    fixed_mean3 = tf.reduce_mean(fixed,axis=3,keep_dims=True)

    theta = fixed_mean2 - moving_mean2
    phi = fixed_mean3 - moving_mean3
    return theta, phi



def mse_loss(label_fixed, label_moving,label_flag_moving):
    one_batch = tf.reduce_mean(tf.abs(label_fixed - label_moving), axis=[1, 2, 3, 4])
    all_batch = tf.reduce_mean(one_batch * label_flag_moving)
    return all_batch

def new_loss(label_fixed, label_moving,theta,phi):
    one_batch = tf.reduce_mean(tf.abs(label_fixed - tf.roll(label_moving,shift=[2,3],axis=[2,3])), axis=[1, 2, 3, 4])
    return one_batch



def dice_loss(label_fixed, label_moving,eps_vol=1e-6):
    pos_label_fixed = tf.gather(label_fixed,indices=[0,1],axis=0)
    pos_label_moving = tf.gather(label_moving,indices=[0,1],axis=0)
    neg_label_fixed = tf.gather(label_fixed,indices=[2,3],axis=0)
    neg_label_moving = tf.gather(label_moving,indices=[2,3],axis=0)
    
    
    pos_numerator = tf.reduce_sum(pos_label_fixed*pos_label_moving, axis=[0, 1, 2, 3, 4]) * 2
    pos_denominator = tf.reduce_sum(pos_label_fixed, axis=[0, 1, 2, 3, 4]) + tf.reduce_sum(pos_label_moving, axis=[0, 1, 2, 3, 4])+eps_vol
    
    neg_numerator = tf.reduce_sum(neg_label_fixed*neg_label_moving, axis=[0, 1, 2, 3, 4]) * 2
    neg_denominator = tf.reduce_sum(neg_label_fixed, axis=[0, 1, 2, 3, 4]) + tf.reduce_sum(neg_label_moving, axis=[0, 1, 2, 3, 4])+eps_vol
    
    pos_similarity = 1 - pos_numerator/pos_denominator
    neg_similarity = 1 - neg_numerator/neg_denominator
    #return tf.reduce_mean(pos_similarity), tf.reduce_mean(neg_similarity)
    return pos_similarity, neg_similarity
    
