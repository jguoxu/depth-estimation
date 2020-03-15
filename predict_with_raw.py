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

# from scipy import imageio
from PIL import Image
import numpy as np

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

RUN_REFINE = False
NYU_FILE_PATH = 'data/nyu_depth_v2_labeled.mat'

def main():
    print(tf.__version__)

    x_eval = []
    y_eval = []
    h5file = h5py.File(NYU_FILE_PATH, 'r')
    file_count = h5file['images'].shape[0]

    # file_count = 100
    dev_split = 0.9
    train_count = file_count * dev_split
    for i in range(file_count):
        if i < train_count:
            continue
        else:
            print("processing image " + str(i))
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
            x_eval.append(image_np_arr)
            y_eval.append(depth_np_arr)

    x_eval = np.array(x_eval) / 255.0
    y_eval = np.array(y_eval)

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
                  # List of metrics to monitor
                #   metrics= [metrics.rmse])
                  metrics= [metrics.abs_relative_diff, metrics.squared_relative_diff, metrics.rmse])


    result = model.evaluate(x=x_eval, y=y_eval, batch_size=32)
    print("Final eval loss on validation: ", result)

    # print("truth: " + str(y_eval))

    # return
    if not os.path.isdir(PREDICT_FILE_PATH):
        os.mkdir(PREDICT_FILE_PATH)

    predictions = model.predict(x=x_eval)
    print("Prediction dim: " + str(predictions.shape))

    for i in range(predictions.shape[0]):
        # print("prediction before: " + str(predictions[i]))

        predictions[i] = (predictions[i] / 10.0) * 255.0

        # print("prediction after: " + str(predictions[i]))
        prediction_name = os.path.join(PREDICT_FILE_PATH, '%05d_predict.png' % i)
        prediction_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)))
        prediction_im.save(prediction_name)

        color_name = os.path.join(PREDICT_FILE_PATH, '%05d_c.png' % i)
        color_im = Image.fromarray(np.uint8(x_eval[i] * 255.0))
        color_im.save(color_name)

        depth_name = os.path.join(PREDICT_FILE_PATH, '%05d_d.png' % i)
        depth_im = Image.fromarray(np.uint8((y_eval[i] / 10.0) * 255.0))
        depth_im.save(depth_name)


if __name__ == '__main__':
    main()
