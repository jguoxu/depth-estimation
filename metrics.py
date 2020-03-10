import tensorflow as tf
import math
import tensorflow.keras.backend as K

TARGET_HEIGHT = 55
TARGET_WIDTH = 74
N = K.cast(TARGET_HEIGHT * TARGET_WIDTH, dtype='float32')

# set y true to 1 if it's zero to avoid NaN division error
# set y pred to 1 if y true was 0 (y true becomes 1 after this block) so
# that the metric does not take it into account (difference = 0)
def clean_y_true(y_true, y_pred):
    y_pred = tf.where(y_true == 0, 1.0, y_pred)
    y_true = tf.where(y_true == 0, 1.0, y_true)
    return y_true, y_pred

def abs_relative_diff(y_true, y_pred):
    y_true, y_pred = clean_y_true(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.abs(d) / y_true
    mean_diff = K.sum(single_diff) / N
    return mean_diff

def squared_relative_diff(y_true, y_pred):
    y_true, y_pred = clean_y_true(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.square(K.abs(d)) / y_true
    mean_diff = K.sum(single_diff) / N
    return mean_diff

def rmse(y_true, y_pred):
    d = K.cast(y_pred - y_true, dtype='float32')
    return K.sqrt(K.sum(K.square(d))/N)
