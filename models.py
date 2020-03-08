from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization, concatenate
from train import IMAGE_HEIGHT, IMAGE_WIDTH
from train import TARGET_WIDTH, TARGET_HEIGHT

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