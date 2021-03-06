import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization, concatenate
import tensorflow.keras.backend as K

from train import IMAGE_HEIGHT, IMAGE_WIDTH
from train import TARGET_WIDTH, TARGET_HEIGHT

#reshape to batch of 1
def reshape(y_true, y_pred):
    y_true = Reshape((55 * 74 ,1))(y_true)
    y_pred = Reshape((55 * 74, 1))(y_pred)
    return y_true, y_pred

# sets y true and pred to the same value if y true is invalid (0)
def clean_y(y_true, y_pred):
    y_pred = tf.where(y_true < np.finfo(np.float32).eps, tf.ones_like(y_true), y_pred)
    y_true = tf.where(y_true < np.finfo(np.float32).eps, tf.ones_like(y_true), y_true)
    return y_true, y_pred


def clean_x(y_true, y_pred):
    y_true = tf.where(y_pred < np.finfo(np.float32).eps, tf.ones_like(y_pred), y_true)
    y_pred = tf.where(y_pred < np.finfo(np.float32).eps, tf.ones_like(y_pred), y_pred)
    return y_true, y_pred

# refered from: https://github.com/jahab/Depth-estimation/blob/master/Depth_Estimation_GD.ipynb
def depth_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # without discarding infinity pixels, the loss will quickly gets to nan.
    lnYTrue = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    lnYPred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)

    invalid_depths = tf.where(y_true <= 0, 0.0, 1.0)
    lnYTrue = tf.multiply(lnYTrue, invalid_depths)
    lnYPred = tf.multiply(lnYPred, invalid_depths)

    d_arr = K.cast(lnYTrue - lnYPred, dtype='float32')

    log_diff = K.cast(K.sum(K.square(d_arr)) / 4070.0, dtype='float32')
    penalty = K.square(K.sum(d_arr)) / K.cast(K.square(4070.0), dtype='float32')
    
    loss = log_diff + penalty

    return loss


def rmse_scale_invariance_log_loss(y_true, y_pred):
    y_true, y_pred = reshape(y_true, y_pred)
    y_true, y_pred = clean_y(y_true, y_pred)
    y_true, y_pred = clean_x(y_true, y_pred)

    d = K.cast(K.log(y_pred) - K.log(y_true), dtype='float32')
    a = K.sum(K.log(y_true) - K.log(y_pred), axis=1) / N
    loss = K.sum(K.square(d + K.repeat(a, 4070)), axis=1) / N
    return K.mean(loss)


def depth_loss_2(y_true, y_pred):
    y_true = Reshape((55*74 ,1))(y_true)
    y_pred = Reshape((55 * 74, 1))(y_pred)
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # without discarding infinity pixels, the loss will quickly gets to nan.
    lnYTrue = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    lnYPred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)

    invalid_depths = tf.where(y_true <= 0, 0.0, 1.0)
    lnYTrue = tf.multiply(lnYTrue, invalid_depths)
    lnYPred = tf.multiply(lnYPred, invalid_depths)

    d_arr = K.cast(lnYTrue - lnYPred, dtype='float32')

    log_diff = K.cast(K.sum(K.square(d_arr), axis = 1) / 4070.0, dtype='float32')
    penalty = K.square(K.sum(d_arr, axis = 1)) / K.cast(K.square(4070.0), dtype='float32')

    loss = log_diff + penalty
    loss = K.mean(loss)

    return loss

def coarse_network_model():
    first_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    conv1 = Conv2D(96, (11, 11), strides=(4, 4), padding='same')(first_layer)
    b1 = BatchNormalization()(conv1)
    a1 = Activation("relu")(b1)
    p1 = MaxPooling2D(pool_size=(2, 2))(a1)

    conv2 = Conv2D(256, (5, 5), padding='same')(p1)
    b2 = BatchNormalization()(conv2)
    a2 = Activation("relu")(b2)
    p2 = MaxPooling2D(pool_size=(2, 2))(a2)

    conv3 = Conv2D(384, (3, 3), padding='same')(p2)
    b3 = BatchNormalization()(conv3)
    a3 = Activation("relu")(b3)

    conv4 = Conv2D(384, (3, 3), padding='same')(a3)
    b4 = BatchNormalization()(conv4)
    a4 = Activation("relu")(b4)

    Dlayer1 = Dense(256)(a4)
    b5 = BatchNormalization()(Dlayer1)
    a5 = Activation("relu")(b5)
    p5 = MaxPooling2D(pool_size=(2, 2))(a5)

    flat = Flatten()(p5)
    flat = Dense(4096)(flat)
    flat = BatchNormalization()(flat)
    flat = Activation("linear")(flat)
    flat = Dropout(0.4)(flat)

    flat = Dense(4070, activation='relu')(flat)
    flat = BatchNormalization()(flat)
    coarse_output = Reshape((TARGET_HEIGHT, TARGET_WIDTH, 1))(flat)

    coarse_model = Model(inputs=first_layer, outputs=coarse_output)
    coarse_model.summary()
    return coarse_model, coarse_output, first_layer

def refined_network_model():
    coarse_model, coarse_output, first_layer = coarse_network_model()
    conv21 = Conv2D(63, (9, 9), strides=(2, 2), padding='valid')(first_layer)
    b21 = BatchNormalization()(conv21)
    p21 = MaxPooling2D(pool_size=(2, 2))(b21)

    Concat = concatenate([coarse_output, p21])
    conv22 = Conv2D(64, (5, 5), padding='same')(Concat)
    b22 = BatchNormalization()(conv22)

    out = Conv2D(1, (5, 5), padding='same')(b22)
    out = BatchNormalization()(out)

    refine_model = Model(inputs=first_layer, outputs=out)
    refine_model.summary()
    return refine_model, coarse_model