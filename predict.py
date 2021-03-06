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

RUN_REFINE = True

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


def main():
    print(tf.__version__)

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
                  loss=models.depth_loss,
                  # List of metrics to monitor
                  metrics=None)


    x_eval, y_eval = csv_inputs(csv_file_path='data/dev.csv')
    result = model.evaluate(x=x_eval, y=y_eval)
    print("Final eval loss on validation: ", result)

    if not os.path.isdir(PREDICT_FILE_PATH):
        os.mkdir(PREDICT_FILE_PATH)

    predictions = model.predict(x=x_eval)
    print("Prediction dim: " + str(predictions.shape))

    for i in range(predictions.shape[0]):
        predictions[i] = (predictions[i] / np.max(predictions[i])) * 255.0
        prediction_name = os.path.join(PREDICT_FILE_PATH, '%05d_predict.png' % i)
        prediction_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)))
        prediction_im.save(prediction_name)

        color_name = os.path.join(PREDICT_FILE_PATH, '%05d_c.png' % i)
        color_im = Image.fromarray(np.uint8(x_eval[i] * 255.0))
        color_im.save(color_name)

        depth_name = os.path.join(PREDICT_FILE_PATH, '%05d_d.png' % i)
        depth_im = Image.fromarray(np.uint8(y_eval[i] * 255.0))
        depth_im.save(depth_name)


if __name__ == '__main__':
    main()
