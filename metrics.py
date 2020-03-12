import tensorflow as tf
import math
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape
import numpy as np

TARGET_HEIGHT = 55
TARGET_WIDTH = 74
N = K.cast(TARGET_HEIGHT * TARGET_WIDTH, dtype='float32')

#reshape to batch of 1
def reshape(y_true, y_pred):
    y_true = Reshape((55 * 74 ,1))(y_true)
    y_pred = Reshape((55 * 74, 1))(y_pred)
    return y_true, y_pred

# sets y true and pred to the same value if y true is invalid (0)
def clean_y(y_true, y_pred):
    # y_pred = tf.where(y_true < np.finfo(np.float32).eps, tf.ones_like(y_true), y_pred)
    # y_true = tf.where(y_true < np.finfo(np.float32).eps, tf.ones_like(y_true), y_true)
    return y_true, y_pred


def clean_x(y_true, y_pred):
    # y_true = tf.where(y_pred < np.finfo(np.float32).eps, tf.ones_like(y_pred), y_true)
    # y_pred = tf.where(y_pred < np.finfo(np.float32).eps, tf.ones_like(y_pred), y_pred)
    return y_true, y_pred


def abs_relative_diff(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    y_true, y_pred = clean_y(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.abs(d) / y_true
    return K.mean(K.sum(single_diff, axis=1) / N)

def squared_relative_diff(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    y_true, y_pred = clean_y(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.square(K.abs(d)) / y_true
    return K.mean(K.sum(single_diff, axis=1) / N)

def rmse(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    d = K.cast(y_pred - y_true, dtype='float32')
    a = K.square(d)
    return K.sqrt(K.sum(a, axis=1)/N)

def rmse_log(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    y_true, y_pred = clean_y(y_true, y_pred)
    y_true, y_pred = clean_x(y_true, y_pred)

    # discard negative, zero and infinity
    # y_true = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    # y_pred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)
    # invalid_depths = tf.where(y_true <= 0, 0.0, 1.0)
    # y_true = tf.multiply(y_true, invalid_depths)
    # y_pred = tf.multiply(y_pred, invalid_depths)

    # if (y_true <= 0 || y_pred <=0) {
    #     y_true = 1.0
    #     y_pred = 1.0
    # }
    
    d = K.cast(K.log(y_pred) - K.log(y_true), dtype='float32')
    a = K.square(K.abs(d))
    return K.sqrt(K.mean(K.sum(a, axis=1)/N))

#a = 1/n (sum(log ytrue - log ypred))
#loss = sum(sq(log ypred - log ytrue + a)) / n
def rmse_scale_invariance_log(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    y_true, y_pred = clean_y(y_true, y_pred)
    y_true, y_pred = clean_x(y_true, y_pred)

    d = K.cast(K.log(y_pred) - K.log(y_true), dtype='float32')
    a = K.sum(K.log(y_true) - K.log(y_pred), axis=1) / N
    loss = K.sum(K.square(d + K.repeat(a, 4070)), axis=1) / N
    return K.mean(loss)

#a = 1/n (sum(log ytrue - log ypred))
#loss = sum(sq(log ypred - log ytrue + a)) / n
# def rmse_scale_invariance(y_true, y_pred):
#     y_true, y_pred = reshape(y_true, y_pred)
#     y_true, y_pred = clean_y(y_true, y_pred)
#     y_true, y_pred = clean_x(y_true, y_pred)

#     d = K.cast(y_pred - y_true, dtype='float32')
#     a = K.sum(y_true - y_pred) / N
#     loss = K.sum(K.square(d + a), axis=1)/ N
#     return K.mean(loss)


