import tensorflow as tf
import math
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape

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
    y_pred = tf.where(y_true < (1.0 * math.pow(10, -6)), tf.ones_like(y_true), y_pred)
    y_true = tf.where(y_true < (1.0 * math.pow(10, -6)), tf.ones_like(y_true), y_true)
    return y_true, y_pred

def abs_relative_diff(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    y_true, y_pred = clean_y(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.abs(d) / y_true
    return K.mean(K.sum(single_diff) / N)

def squared_relative_diff(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    y_true, y_pred = clean_y(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(y_pred - y_true, dtype='float32')
    single_diff = K.square(K.abs(d)) / y_true
    return K.mean(K.sum(single_diff) / N)

def rmse(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    d = K.cast(y_pred - y_true, dtype='float32')
    a = K.square(K.abs(d))
    return K.mean(K.sqrt(K.sum(a)/N))

def rmse_log(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)

    # discard negative, zero and infinity
    y_true = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    y_pred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)
    invalid_depths = tf.where(y_true <= 0, 0.0, 1.0)
    y_true = tf.multiply(y_true, invalid_depths)
    y_pred = tf.multiply(y_pred, invalid_depths)
    
    d = K.cast(K.log(y_pred) - K.log(y_true), dtype='float32')
    a = K.square(K.abs(d))
    return K.mean(K.sqrt(K.sum(a)/N))

#a = 1/n (sum(log ytrue - log ypred))
#loss = sum(sq(log ypred - log ytrue + a)) / n
def rmse_scale_invariance(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    d = K.cast(y_pred - y_true, dtype='float32')
    a = K.sum(y_true - y_pred) / N
    loss = K.sum(K.square(d + a))/ N
    return K.mean(loss)

#a = 1/n (sum(log ytrue - log ypred))
#loss = sum(sq(log ypred - log ytrue + a)) / n
def rmse_scale_invariance_log(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    d = K.cast(K.log(y_pred) - K.log(y_true), dtype='float32')
    a = K.sum(K.log(y_true) - K.log(y_pred)) / N
    loss = K.sum(K.square(d + a))/ N
    return K.mean(loss)
