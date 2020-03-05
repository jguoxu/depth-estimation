from __future__ import absolute_import, division, print_function, unicode_literals

import os.path

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, InputLayer,UpSampling2D, Reshape,UpSampling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import multi_gpu_model

# from scipy import imageio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

COARSE_CHECKPOINT_PATH = 'checkpoints/coarse/coarse_ckpt'
COARSE_CHECKPOINT_DIR = os.path.dirname(COARSE_CHECKPOINT_PATH)
PREDICT_FILE_PATH = 'data/predict'

class NyuDepthGenerator(keras.utils.Sequence) :

    def __init__(self, batch_size, csv_path='data/train.csv') :
        tf.keras.backend.clear_session() #Reset notebook state
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

            example = example.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            label = label.resize((TARGET_WIDTH, TARGET_HEIGHT))

            x_train.append(np.array(example))
            # flatten is needed because of the dense layer output is 1d
            y_train.append(np.array(label))
        # (isinghal): Not sure if we should divide input by 255
        return np.array(x_train) / 1.0, np.array(y_train) / 255.0


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
    
    loss = log_diff + penalty

    return loss



def msr_loss(y_true, y_pred):
    print(y_true)

    flatten_true = K.flatten(y_true)
    flatten_pred = K.flatten(y_pred)
    loss=K.mean(K.sum(K.square(flatten_true-flatten_pred)))

    print ("y_true:")
    K.print_tensor(y_true)

    # print ("predict:")
    # K.print_tensor(y_pred)

    # d = K.log(y_true) - K.log(y_pred)
    # log_diff = K.sum(K.square(d)) / 4070.0 # 4070 is number of pixels (74, 55)
    # penalty = K.square(K.sum(d)) / K.square(4070.0)
    # loss = log_diff - penalty

    return loss


def model2():
    model=Sequential()

    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3],padding='same'))
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
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    model.add(Dense(4070))
    model.add(BatchNormalization())
    model.add(Activation("linear"))
    model.add(Reshape((TARGET_HEIGHT, TARGET_WIDTH,1)))

    # model.add(Reshape((64, 64, 1)))
    #
    # model.add(UpSampling2D(size=(2,2)))
    # model.add(Conv2D(1,(55,74),padding='valid'))
    # model.add(BatchNormalization())
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

    # Create a callback that saves the model's weights with every epoch (save_freq=1)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=COARSE_CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
    model = model2()

    latest_checkpoint = tf.train.latest_checkpoint(COARSE_CHECKPOINT_DIR)

    if latest_checkpoint:
        print("\nRestored model from checkpoint")
        model.load_weights(latest_checkpoint)
    else: 
        print("\nTraining model from scratch")
        
    nyu_data_generator = NyuDepthGenerator(batch_size=10)

    # parallel_model = multi_gpu_model(model, gpus=2)
    # parallel_model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
    #           # Loss function to minimize
    #           loss=msr_loss,
    #           # List of metrics to monitor
    #           metrics=None)

    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
              # Loss function to minimize
              loss=depth_loss,
              # List of metrics to monitor
              metrics=None)

    print("model metric names: " + str(model.metrics_names))

    print('# Fit model on training data')
    history = model.fit(x=nyu_data_generator,
                       epochs=300, callbacks=[cp_callback])

    print('\nhistory dict:', history.history)
    np.savetxt("loss_history.txt", history.history["loss"], delimiter=",")

    result = model.evaluate(x=nyu_data_generator, steps=1)
    print("test loss: ", result)

    if not os.path.isdir(PREDICT_FILE_PATH):
        os.mkdir(PREDICT_FILE_PATH)
    predictions = model.predict(x=nyu_data_generator, steps=1)
    print('predictions shape:', predictions.shape)
    for i in range(predictions.shape[0]):
        predictions[i] = (predictions[i] / np.max(predictions[i])) * 255.0
        print("prediction shape:" + str(predictions[i].shape))
        # return
        # print(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH))
        image_name = os.path.join(PREDICT_FILE_PATH, '%05d_d.png' % i)
        image_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)), mode="L")
        image_im.save(image_name)
    # predictions[0] = (predictions[0] / np.max(predictions[0])) * 255.0
    # print(predictions[0].reshape(TARGET_HEIGHT, TARGET_WIDTH))
    # image_name = os.path.join(PREDICT_FILE_PATH, '%05d_d.png' % (1))
    # image_im = Image.fromarray(np.uint8(predictions[0].reshape(TARGET_HEIGHT, TARGET_WIDTH)), mode="L")
    # image_im.save(image_name)


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
