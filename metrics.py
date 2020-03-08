import tensorflow as tf
import math
import tensorflow.keras.backend as K

TARGET_HEIGHT = 55
TARGET_WIDTH = 74
N = TARGET_HEIGHT * TARGET_WIDTH
FLOAT32_N = K.cast(N, dtype='float32')

# without discarding infinity pixels, the loss will quickly gets to nan.
def discardInfinityPixels(y_true, y_pred):
    ln_y_true = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    ln_y_pred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)
    return ln_y_true, ln_y_pred

# "Training loss" or equation 4 from https://arxiv.org/pdf/1406.2283.pdf
# a = mean(sum(sq(d)))
# b = (l/sq(n)) * sum(sq(d)))
# loss = a - b
def scale_invariant_loss(y_true, y_pred):
    ln_y_true, ln_y_pred = discardInfinityPixels(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    d = K.cast(ln_y_true - ln_y_pred, dtype='float32')
    
    #a
    sum_squared_d = K.sum(K.square(d))
    mean_sum_sq_d = sum_squared_d / FLOAT32_N

    #b
    l = K.cast(0.5, dtype='float32') #lambda
    constant = (l / math.pow(FLOAT32_N, 2))
    penalty = constant * K.square(K.sum(d)) 

    # a - b
    loss = mean_sum_sq_d + penalty # huh?? why does minus give a negative value
    return loss

# Equation 1 from https://arxiv.org/pdf/1406.2283.pdf
# a = d
# b = mean(sum(d))
# loss = mean(sq((a + b))
def scale_invariant_mse(y_true, y_pred):
    ln_y_true, ln_y_pred = discardInfinityPixels(K.cast(y_true, dtype='float32'), K.cast(y_pred, dtype='float32'))
    
    #a
    d = K.cast(ln_y_true - ln_y_pred, dtype='float32')

    #b
    mean_sum_d = K.sum(d) / FLOAT32_N

    # sq(a + b)
    single_loss = K.square(d + mean_sum_d) 
    loss = K.sum(single_loss) / FLOAT32_N

    return loss

def depth_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # without discarding infinity pixels, the loss will quickly gets to nan.
    lnYTrue = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    lnYPred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)

#    invalid_depths = tf.where(y_true < 0, 0.0, 1.0)
#    lnYTrue = tf.multiply(lnYTrue, invalid_depths)
#    lnYPred = tf.multiply(lnYPred, invalid_depths)

    d_arr = K.cast(lnYTrue - lnYPred, dtype='float32')

    log_diff = K.cast(K.sum(K.square(d_arr)) / 4070.0, dtype='float32')
    penalty = K.square(K.sum(d_arr)) / K.cast(K.square(4070.0), dtype='float32')
    
    loss = log_diff+penalty

    return loss