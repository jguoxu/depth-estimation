from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization, concatenate
from train import IMAGE_HEIGHT, IMAGE_WIDTH
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

def coarse_network_model():
    model=Sequential()
    model.add(Conv2D(96,(11,11),strides=(4,4), input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256,(5,5),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(384,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(384,(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation("linear"))
    model.add(Dropout(0.4))

    model.add(Dense(4070, activation='relu'))
    model.add(BatchNormalization())
    model.add(Reshape((TARGET_HEIGHT, TARGET_WIDTH)))
    model.summary()
    return model

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