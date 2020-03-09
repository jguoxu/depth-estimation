import tensorflow as tf
import math
import tensorflow.keras.backend as K

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
