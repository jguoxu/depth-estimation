from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import multi_gpu_model

# from scipy import imageio
from PIL import Image
import numpy as np

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74


class NyuDepthGenerator(keras.utils.Sequence) :

    def __init__(self, batch_size, csv_path='data/train.csv') :
        self.batch_size = batch_size

        self.csv_file = open(csv_path, mode='r')
        self.csv_lines = self.csv_file.readlines()


    def __len__(self) :
        return int(np.floor(len(self.csv_lines) / self.batch_size))


    def __getitem__(self, idx) :
        x_train = []
        y_train = []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            line = self.csv_lines[i]
            line = line.replace('\n', '')
            pairs = line.split(',')

            example = Image.open(pairs[0])
            label = Image.open(pairs[1])

            example = example.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
            label = label.resize((TARGET_HEIGHT, TARGET_WIDTH))

            x_train.append(np.array(example))
            # flatten is needed because of the dense layer output is 1d
            y_train.append(np.array(label))

        return np.array(x_train) / 255.0, np.array(y_train) / 255.0


# refered from: https://github.com/jahab/Depth-estimation/blob/master/Depth_Estimation_GD.ipynb
def depth_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # without discarding infinity pixels, the loss will quickly gets to nan.
    lnYTrue = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    lnYPred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)

    d_arr = K.cast(lnYTrue - lnYPred, dtype='float32')

    log_diff = K.cast(K.sum(K.square(d_arr)) / 4070.0, dtype='float32')
    penalty = K.square(K.sum(d_arr)) / K.cast(K.square(4070.0), dtype='float32')
    
    loss = log_diff+penalty

    return loss



def msr_loss(y_true, y_pred):
    # print(y_true)

    flatten_true = K.flatten(y_true)
    flatten_pred = K.flatten(y_pred)
    loss=K.mean(K.sum(K.square(flatten_true-flatten_pred)))

    # K.print_tensor(y_true)

    # print ("predict:")
    # K.print_tensor(y_pred)

    # d = K.log(y_true) - K.log(y_pred)
    # log_diff = K.sum(K.square(d)) / 4070.0 # 4070 is number of pixels (74, 55)
    # penalty = K.square(K.sum(d)) / K.square(4070.0)
    # loss = log_diff - penalty

    return loss


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


def main():
    print(tf.__version__)

    model = model2()
    nyu_data_generator = NyuDepthGenerator(batch_size=10)

    parallel_model = multi_gpu_model(model, gpus=4)
    parallel_model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize
              loss=depth_loss,
              # List of metrics to monitor
              metrics=None)

    # model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
    #           # Loss function to minimize
    #           loss=depth_loss,
    #           # List of metrics to monitor
    #           metrics=None)

    print("model metric names: " + str(model.metrics_names))

    print('# Fit model on training data')
    # when using data generate, x contains both X and Y. 
    # batch size is define in the generator thus passing None to batch_size
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    history = parallel_model.fit(x=nyu_data_generator,
                        epochs=10)#(x_val, y_val))

    # history = model.fit_generator(nyu_data_generator, steps_per_epoch=5, epochs=1)

    print('\nhistory dict:', history.history)

    # Evaluate the model on the test data using `evaluate`
    # print('\n# Evaluate on test data')
    # results = model.evaluate(x_test, y_test, batch_size=128)
    # print('test loss, test acc:', results)

    # # Generate predictions (probabilities -- the output of the last layer)
    # # on new data using `predict`
    # print('\n# Generate predictions for 3 samples')
    # predictions = model.predict(x_test[:3])
    # print('predictions shape:', predictions.shape)


def debug_display_rgbd_pair(rgb, d):
    img = Image.fromarray(rgb, 'RGB')
    img.show()

    # mode 'P' is 8bit pixel:
    # https://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes
    img = Image.fromarray(d, 'P')
    img.show()


if __name__ == '__main__':
    main()
    # x_train, y_train = csv_inputs()
    # print("x_train shape: " + str(x_train.shape))
    # print("y_train shape: " + str(y_train.shape))

    # debug_display_rgbd_pair(x_train[0], y_train[0])
