import tensorflow as tf
import math
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape

TARGET_HEIGHT = 55
TARGET_WIDTH = 74
N = K.cast(TARGET_HEIGHT * TARGET_WIDTH, dtype='float32')

# set y true to small non zero number if it's zero to avoid division error
# set y true to 1 if infinity
def clean_y_true(y_true):
    y_true = tf.where(y_true == 0, 0.001, y_true)
    y_true = tf.where(tf.math.is_inf(y_true), 1.0, y_true)
    return y_true

def abs_relative_diff(y_true, y_pred):
    y_true = clean_y_true(K.cast(y_true, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.abs(d) / y_true
    mean_diff = K.sum(single_diff) / N
    return mean_diff

def squared_relative_diff(y_true, y_pred):
    y_true = clean_y_true(K.cast(y_true, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.square(K.abs(d)) / y_true
    mean_diff = K.sum(single_diff) / N
    return mean_diff

def rmse(y_true, y_pred):
    d = K.cast(y_pred - y_true, dtype='float32')
    return K.sqrt(K.sum(K.square(d))/N)

def rmse_scale_invariant(y_true, y_pred):
    y_true = Reshape((55*74, 1))(y_true)
    y_pred = Reshape((55*74, 1))(y_pred)
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # without discarding infinity pixels, the loss will quickly gets to nan.
    lnYTrue = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    lnYPred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)

    invalid_depths = tf.where(y_true <= 0, 0.0, 1.0)
    lnYTrue = tf.multiply(lnYTrue, invalid_depths)
    lnYPred = tf.multiply(lnYPred, invalid_depths)

    d_arr = K.cast(lnYTrue - lnYPred, dtype='float32')

    log_diff = K.cast(K.sum(K.square(d_arr), axis=1) / N, dtype='float32')
    penalty = K.square(K.sum(d_arr, axis=1)) / K.cast(K.square(N), dtype='float32')

    diff = log_diff - penalty
    rmse = K.sqrt(K.mean(diff))

    return rmse
