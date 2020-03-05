from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization, concatenate
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

def refinedNetworkModel(coarse_training_results):
    layer1 = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)) 

    conv1 = (Conv2D(filters = 63, kernel_size= (9, 9),
                    strides=(2, 2),
                    padding="valid",
                    input_shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 3]))(layer1)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(batch1)

    #convert sequential model to functional so that we can concatenate
    coarse_input = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    # verify how to convert input 
    # https://stackoverflow.com/questions/50837715/conversion-of-a-sequential-model-to-a-functional-model-with-keras-2-2-0/50937113#50937113
    coarse_layer = coarse_input
    for layer in coarse_training_results.layers:
        coarse_layer = layer(coarse_layer)

    merged = concatenate([coarse_layer, pool1]) 
    conv2 = Conv2D(filters = 64, kernel_size=(5, 5), padding="same")(merged)
    batch2 = BatchNormalization()(conv2)

    out = Conv2D(filters = 1, kernel_size=(5, 5), padding="same")(batch2)
    out = BatchNormalization()(out)

    model = Model(inputs=[layer1, coarse_layer], outputs=out)
    model.summary()
    return model

# def refinedNetworkModelSequential(coarse_training_results):
#     first_layer = Sequential()
#     first_layer.add(Conv2D(filters = 63, kernel_size= (9, 9),
#                     strides=(2, 2),
#                     padding="valid",
#                     input_shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 3]))
#     first_layer.add(Activation("relu"))
#     first_layer.add(BatchNormalization())
#     first_layer.add(MaxPooling2D(pool_size=(2,2)))


#     merged_model = Concatenate([first_layer, coarse_training_results]) 
#     merged_model.add(Conv2D(filters = 64, kernel_size=(5, 5), padding="same"))
#     merged_model.add(BatchNormalization())
#     merged_model.add(Conv2D(filters = 1, kernel_size=(5, 5), padding="same"))
#     merged_model.add(BatchNormalization())
#     merged_model.summary()
#     return model