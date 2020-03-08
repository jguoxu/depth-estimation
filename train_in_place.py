from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
import models

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import multi_gpu_model

# from scipy import imageio
from PIL import Image
import numpy as np

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

COARSE_CHECKPOINT_PATH = 'checkpoints/coarse/coarse_ckpt'
COARSE_CHECKPOINT_DIR = os.path.dirname(COARSE_CHECKPOINT_PATH)
REFINED_CHECKPOINT_PATH = 'checkpoints/refined/refined_ckpt'
REFINED_CHECKPOINT_DIR = os.path.dirname(REFINED_CHECKPOINT_PATH)
PREDICT_FILE_PATH = 'data/predict'

RUN_REFINE = False

# class NyuDepthGenerator(keras.utils.Sequence):

#     def __init__(self, batch_size, csv_path='data/train.csv') :
#         tf.keras.backend.clear_session() #Reset notebook state
#         self.batch_size = batch_size

#         self.csv_file = open(csv_path, mode='r')
#         self.csv_lines = self.csv_file.readlines()


#     def __len__(self) :
#         return int(np.floor(len(self.csv_lines) / self.batch_size))


#     def __getitem__(self, idx) :
#         x_train = []
#         y_train = []
#         for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
#             line = self.csv_lines[i]
#             line = line.replace('\n', '')
#             pairs = line.split(',')

#             example = Image.open(pairs[0])
#             label = Image.open(pairs[1])

#             example = example.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
#             label = label.resize((TARGET_WIDTH, TARGET_HEIGHT))

#             x_train.append(np.array(example))
#             # flatten is needed because of the dense layer output is 1d
#             y_train.append(np.array(label))

#         return np.array(x_train) / 255.0, np.array(y_train) / 255.0

def csv_inputs(csv_file_path='data/train.csv'):	
    x_train = []
    y_train = []

    count = 0
    with open(csv_file_path, mode='r') as csv_file:	
        lines = csv_file.readlines()
        for line in lines:
            line = line.replace('\n', '')
            pairs = line.split(',')

            example = Image.open(pairs[0])
            label = Image.open(pairs[1])

            example = example.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            label = label.resize((TARGET_WIDTH, TARGET_HEIGHT))

            x_train.append(np.array(example))
            y_train.append(np.array(label))

            if count % 10 == 0:
                print("processed img: " + str(count))
            count = count + 1

    return np.array(x_train) / 255.0, np.array(y_train) / 255.0


# refered from: https://github.com/jahab/Depth-estimation/blob/master/Depth_Estimation_GD.ipynb
def depth_loss(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')

    # without discarding infinity pixels, the loss will quickly gets to nan.
    lnYTrue = tf.where(tf.math.is_inf(y_true), tf.ones_like(y_true), y_true)
    lnYPred = tf.where(tf.math.is_inf(y_pred), tf.ones_like(y_pred), y_pred)

#    invalid_depths = tf.where(y_true < 0, 0.0, 1.0)
#    lnYTrue = tf.multiply(lnYTrue, invalid_depths)
#    lnYPred = tf.multiply(lnYPred, invalid_depths)

    d_arr = K.cast(lnYTrue - lnYPred, dtype='float32')

    log_diff = K.cast(K.sum(K.square(d_arr)) / 4070.0, dtype='float32')
    penalty = K.square(K.sum(d_arr)) / K.cast(K.square(4070.0), dtype='float32')
    
    loss = log_diff+penalty

    return loss


def main():
    print(tf.__version__)

    cp_callback_coarse = tf.keras.callbacks.ModelCheckpoint(filepath=COARSE_CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)
    cp_callback_refine = tf.keras.callbacks.ModelCheckpoint(filepath=REFINED_CHECKPOINT_PATH,
                                                            save_weights_only=True,
                                                            verbose=1)

    csv_logger = CSVLogger('log.csv', append=False, separator=',')

    x_train, y_train = csv_inputs()

    latest_checkpoint_refine = tf.train.latest_checkpoint(REFINED_CHECKPOINT_DIR)
    latest_checkpoint_coarse = tf.train.latest_checkpoint(COARSE_CHECKPOINT_DIR)
    if RUN_REFINE:
        refine_model, coarse_model = models.refined_network_model()
        model = refine_model
        if latest_checkpoint_refine:
            print("\nRestored refine model from checkpoint")
            refine_model.load_weights(latest_checkpoint_refine)
        elif latest_checkpoint_coarse:
            print("\nRestored coarse model from checkpoint")
            coarse_model.load_weights(latest_checkpoint_coarse)
        else:
            print("\nCoarse model not restored. Please exit and run coarse model first")
            print("\nStarting one pass training")
    else:
        coarse_model, _, _ = models.coarse_network_model()
        model = coarse_model
        if latest_checkpoint_coarse:
            print("\nRestored coarse model from checkpoint")
            coarse_model.load_weights(latest_checkpoint_coarse)
        else:
            print("\nNo coarse checkpoint saved")

    model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                  # Loss function to minimize
                  loss=depth_loss,
                  # List of metrics to monitor
                  metrics=None)

    print('Fit model on training data')
    if RUN_REFINE:
        history = model.fit(x=x_train, y = y_train,
                            epochs=30, callbacks=[cp_callback_refine, csv_logger])
    else:
        history = model.fit(x=x_train, y = y_train,
                            epochs=30, callbacks=[cp_callback_coarse, csv_logger])

    print('\nHistory dict:', history.history)


    result = model.evaluate(x=x_train, steps=10)
    print("Final eval loss: ", result)

    if not os.path.isdir(PREDICT_FILE_PATH):
        os.mkdir(PREDICT_FILE_PATH)

    predictions = model.predict(x=x_train, steps=10)
    print("Prediction dim: " + str(predictions.shape))

    for i in range(predictions.shape[0]):
        predictions[i] = (predictions[i] / np.max(predictions[i])) * 255.0
        image_name = os.path.join(PREDICT_FILE_PATH, '%05d_d.png' % i)
        image_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)), mode="L")
        image_im.save(image_name)


if __name__ == '__main__':
    main()
