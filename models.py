from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization, concatenate
from train import IMAGE_HEIGHT, IMAGE_WIDTH

def coarse_network_model():
    first_layer=Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    conv1 = Conv2D(96,(11,11),strides=(4,4),activation="relu",padding="same")(first_layer)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(batch1)
    
    conv2 = Conv2D(256, (5, 5), activation="relu", padding="same")(pool1)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(batch2)

    conv3 = Conv2D(384, (3, 3), activation="relu", padding="same")(pool2)
    batch3 = BatchNormalization()(conv3)

    conv4 = Conv2D(384, (3, 3), activation="relu", padding="same")(batch3)
    batch4 = BatchNormalization()(conv4)

    dense5 = Dense(256, activation="relu")(batch4)
    batch5 = BatchNormalization()(dense5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(batch5)

    flatten = Flatten()(pool5)
    flatten = Dense(4096, activation="linear")(flatten)
    flatten = BatchNormalization()(flatten)
    flatten = Dropout(0.4)(flatten) 

    reshaped=Reshape((64,64,1))(flatten)
    upsampled=UpSampling2D(size=(2,2))(reshaped)

    out1 = Conv2D(1, (74, 55), padding="valid")(upsampled)
    out1 = BatchNormalization()(out1)
    model = Model(inputs=first_layer, outputs=out1)
    model.summary
    return model, out1, first_layer

def refined_network_model():
    coarse_model, coarse_training_results, layer1 = coarse_network_model()

    conv1 = (Conv2D(filters = 63, kernel_size= (9, 9),
                    strides=(2, 2),
                    padding="valid"))(layer1)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(batch1)

    merged = concatenate([coarse_training_results, pool1]) 
    conv2 = Conv2D(filters = 64, kernel_size=(5, 5), padding="same")(merged)
    batch2 = BatchNormalization()(conv2)

    out = Conv2D(filters = 1, kernel_size=(5, 5), padding="same")(batch2)
    out = BatchNormalization()(out)

    refined_model = Model(inputs=layer1, outputs=out)
    refined_model.summary()
    return refined_model, coarse_model