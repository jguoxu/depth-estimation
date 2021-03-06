from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
import models
import metrics

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.metrics import RootMeanSquaredError

# from scipy import imageio
from PIL import Image
import numpy as np
# import cv2

import h5py

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74

COARSE_CHECKPOINT_PATH = 'checkpoints/coarse/coarse_ckpt'
COARSE_CHECKPOINT_DIR = os.path.dirname(COARSE_CHECKPOINT_PATH)
REFINED_CHECKPOINT_PATH = 'checkpoints/refined/refined_ckpt'
REFINED_CHECKPOINT_DIR = os.path.dirname(REFINED_CHECKPOINT_PATH)
PREDICT_FILE_PATH = 'data/predict'
TRAIN_PREDICT_FILE_PATH = 'data/predict_train'

RUN_REFINE = False
NYU_FILE_PATH = 'data/nyu_depth_v2_labeled.mat'

class PredictWhileTrain(keras.callbacks.Callback):
    def __init__(self, x_train):
        self.x_train = x_train

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 50 != 0:
            return
        folder_name = os.path.join(TRAIN_PREDICT_FILE_PATH, 'predict_train_%d' % epoch)
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        predictions = self.model.predict(x=self.x_train)

        for i in range(predictions.shape[0]):
            predictions[i] = (predictions[i] / np.max(predictions[i])) * 255.0
            image_name = os.path.join(folder_name, '%05d_d.png' % i)
            image_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)), mode="L")
            image_im.save(image_name)

        print("Saved output predictions for epoch number " + str(epoch))


def main():
    print(tf.__version__)

    x_train = []
    y_train = []
    x_eval = []
    y_eval = []
    h5file = h5py.File(NYU_FILE_PATH, 'r')
    file_count = h5file['images'].shape[0]

    dev_split = 0.9
    train_count = file_count * dev_split
    for i in range(file_count):
        if i % 10 == 0:
            print("processing file " + str(i))

        image = np.transpose(h5file['images'][i], (2, 1, 0))
        depth = np.transpose(h5file['depths'][i], (1, 0))

        image_im = Image.fromarray(np.uint8(image))
        image_im = image_im.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image_np_arr = np.array(image_im)
        # print ("image_np_arr shape: " + str(image_np_arr.shape))

        depth_scaled = (depth / 10.0) * 255.0
        depth_im = Image.fromarray(np.uint8(depth_scaled))
        depth_im = depth_im.resize((TARGET_WIDTH, TARGET_HEIGHT))
        depth_np_arr = np.array(depth_im)
        depth_np_arr = depth_np_arr / 255.0 * 10.0
        # print ("depth_np_arr shape: " + str(depth_np_arr.shape))
        # print ("depth_np_arr: " + str(depth_np_arr))
        if i < train_count:
            x_train.append(image_np_arr)
            y_train.append(depth_np_arr)
        else:
            x_eval.append(image_np_arr)
            y_eval.append(depth_np_arr)

    
    x_train = np.array(x_train) / 255.0
    y_train = np.array(y_train) 
    x_eval = np.array(x_eval) / 255.0
    y_eval = np.array(y_eval)
    print(x_train.shape)
    print(y_train.shape)
    print(x_eval.shape)
    print(y_eval.shape)


    cp_callback_coarse = tf.keras.callbacks.ModelCheckpoint(filepath=COARSE_CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1, period=10)
    cp_callback_refine = tf.keras.callbacks.ModelCheckpoint(filepath=REFINED_CHECKPOINT_PATH,
                                                            save_weights_only=True,
                                                            verbose=1, period=10)

    csv_logger = CSVLogger('log.csv', append=False, separator=',')

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
                  loss=models.depth_loss_2,
                  metrics= [metrics.abs_relative_diff, metrics.squared_relative_diff, metrics.rmse, metrics.rmse_log, metrics.rmse_scale_invariance_log])

    predict_while_train = PredictWhileTrain(x_train)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    if not os.path.isdir(TRAIN_PREDICT_FILE_PATH):
        os.mkdir(TRAIN_PREDICT_FILE_PATH)
    print('Fit model on training data')
    if RUN_REFINE:
        history = model.fit(x=x_train, y = y_train, 
                            validation_data=(x_eval, y_eval),
                            epochs=300, callbacks=[cp_callback_refine, csv_logger])
    else:
        history = model.fit(x=x_train, y = y_train, validation_data=(x_eval, y_eval),
                            epochs=300, callbacks=[cp_callback_coarse, csv_logger])

    print('\nHistory dict:', history.history)

    result = model.evaluate(x=x_eval, y=y_eval)
    print("Final eval loss on validation: ", result)


if __name__ == '__main__':
    main()
