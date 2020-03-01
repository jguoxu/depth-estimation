from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten

# from scipy import imageio
from PIL import Image
import numpy as np

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

# def csv_inputs(csv_file_path='data/train.csv'):
#     x_train = []
#     y_train = []
#     with open(csv_file_path, mode='r') as csv_file:
#         lines = csv_file.readlines()
#         for line in lines:
#             line = line.replace('\n', '')
#             pairs = line.split(',')

#             example = Image.open(pairs[0])
#             label = Image.open(pairs[1])

#             example = example.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
#             label = label.resize((TARGET_HEIGHT, TARGET_WIDTH))

#             x_train.append(np.array(example))
#             y_train.append(np.array(label))

#     return np.array(x_train), np.array(y_train)

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
            y_train.append(np.array(label).flatten())

        return np.array(x_train) / 255.0, np.array(y_train) / 255.0


def msr_loss(y_true, y_pred):
    flatten_true = K.flatten(y_true)
    flatten_pred = K.flatten(y_pred)
    loss=K.mean(K.sum(K.square(flatten_true-flatten_pred)))

    return loss


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

    # inputs = keras.Input(shape=(784,), name='digits')
    # x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    # x = layers.Dense(64, activation='relu', name='dense_2')(x)
    # outputs = layers.Dense(10, name='predictions')(x)

    # model = keras.Model(inputs=inputs, outputs=outputs)

    # model = model1

    # x_train, y_train = csv_inputs()
    # x_train = x_train / 255.0
    # y_train = y_train / 255.0

    # # Reserve 1 samples for validation
    # x_val = x_train[-1:]
    # y_val = y_train[-1:]
    # x_train = x_train[:-1]
    # y_train = y_train[:-1]

    # print(x_val.shape)
    # print(y_val.shape)
    # print(x_train.shape)
    # print(y_train.shape)

    model = model1()
    nyu_data_generator = NyuDepthGenerator(batch_size=10)

    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize
              loss=msr_loss,
              # List of metrics to monitor
              metrics=None)

    print("model metric names: " + str(model.metrics_names))

    print('# Fit model on training data')
    # when using data generate, x contains both X and Y. 
    # batch size is define in the generator thus passing None to batch_size
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
    history = model.fit(x=nyu_data_generator,
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
