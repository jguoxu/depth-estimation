from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization
from train import IMAGE_HEIGHT, IMAGE_WIDTH

def model2():
    model=Sequential()

    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 3],padding='same'))
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

    model.add(Reshape((64, 64,1)))

    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(1,(55,74),padding='valid'))
    model.add(BatchNormalization())
    model.summary()
    return model

def model1():
    model = Sequential()

    # When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
    # https://keras.io/layers/convolutional/
    model.add(Conv2D(filters = 96, kernel_size= (11, 11),
                    strides=(4, 4),
                    padding='same',
                    activation='relu',
                    input_shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 3]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters = 256, kernel_size= (5, 5),
                    strides=(1, 1),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(filters = 384, kernel_size= (3, 3),
                    padding='same'))
    model.add(Conv2D(filters = 384, kernel_size= (3, 3),
                    padding='same'))
    model.add(Conv2D(filters = 256, kernel_size= (3, 3),
                    padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4070, activation='relu'))
    model.summary()
    return model