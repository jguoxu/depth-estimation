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

def csv_inputs(csv_file_path='data/train.csv'):
    x_train = []
    y_train = []
    with open(csv_file_path, mode='r') as csv_file:
        lines = csv_file.readlines()
        for line in lines:
            line = line.replace('\n', '')
            pairs = line.split(',')

            example = Image.open(pairs[0])
            label = Image.open(pairs[1])

            example = example.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
            label = label.resize((TARGET_HEIGHT, TARGET_WIDTH))

            x_train.append(np.array(example))
            y_train.append(np.array(label))

    return np.array(x_train), np.array(y_train)

class NyuDepthGenerator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
               for file_name in batch_x])/255.0, np.array(batch_y)


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

    model = model1

    x_train, y_train = csv_inputs()
    x_train = x_train / 255.0
    y_train = y_train / 255.0

    # Reserve 1 samples for validation
    x_val = x_train[-1:]
    y_val = y_train[-1:]
    x_train = x_train[:-1]
    y_train = y_train[:-1]

    print(x_val.shape)
    print(y_val.shape)
    print(x_train.shape)
    print(y_train.shape)

    model = model1()

    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=msr_loss,
              # List of metrics to monitor
              metrics=[msr_loss])

    print("model metric names: " + str(model.metrics_names))

    print('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=10,
                        epochs=10,
                        validation_data=(x_val, y_val))

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
