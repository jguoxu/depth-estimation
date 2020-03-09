import tensorflow as tf
import math
import tensorflow.keras.backend as K

TARGET_HEIGHT = 55
TARGET_WIDTH = 74
N = K.cast(TARGET_HEIGHT * TARGET_WIDTH, dtype='float32')

# without discarding infinity pixels, the loss will quickly gets to nan.
def discardInfinityAndInvalid(y_true, y_pred):
    y_true = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    y_pred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)

    invalid_depths = tf.where(y_true < 0, 0.0, 1.0)
    y_true = tf.multiply(y_true, invalid_depths)
    y_pred = tf.multiply(y_pred, invalid_depths)
    return y_true, y_pred

def abs_relative_diff(y_true, y_pred):
    y_true, y_pred = discardInfinityAndInvalid(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_true - y_pred, dtype='float32')
    single_diff = K.abs(d) / y_true
    mean_diff = K.sum(single_diff) / N
    return mean_diff

def squared_relative_diff(y_true, y_pred):
    y_true, y_pred = discardInfinityAndInvalid(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_true - y_pred, dtype='float32')
    single_diff = K.square(K.abs(d)) / y_true
    mean_diff = K.sum(single_diff) / N
    return mean_diff

def rmse(y_true, y_pred):
    y_true, y_pred = discardInfinityAndInvalid(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_true - y_pred, dtype='float32')
    return K.sqrt(K.sum(K.square(d))/N)
