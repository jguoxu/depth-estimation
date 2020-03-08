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

class PredictWhileTrain(keras.callbacks.Callback):
    def __init__(self, input_generator):
        self.input_generator = input_generator

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 30 != 0:
            return
        folder_name = os.path.join(TRAIN_PREDICT_FILE_PATH, 'predict_train_%d' % epoch)
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        predictions = self.model.predict(x=self.input_generator, steps=1)

        for i in range(predictions.shape[0]):
            predictions[i] = (predictions[i] / np.max(predictions[i])) * 255.0
            image_name = os.path.join(folder_name, '%05d_d.png' % i)
            image_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)), mode="L")
            image_im.save(image_name)

        print("Saved output predictions for epoch number " + str(epoch))

class NyuDepthGenerator(keras.utils.Sequence):

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

        return np.array(x_train) / 255.0, np.array(y_train) / 255.0


def main():
    print(tf.__version__)

    cp_callback_coarse = tf.keras.callbacks.ModelCheckpoint(filepath=COARSE_CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)
    cp_callback_refine = tf.keras.callbacks.ModelCheckpoint(filepath=REFINED_CHECKPOINT_PATH,
                                                            save_weights_only=True,
                                                            verbose=1)

    csv_logger = CSVLogger('log.csv', append=False, separator=',')

    nyu_data_generator = NyuDepthGenerator(batch_size=10, csv_path='data/train.csv')
    # eval_data_generator = NyuDepthGenerator(batch_size=1, csv_path='data/dev.csv')

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
                  metrics=[RootMeanSquaredError(name='keras_default_RMSE'), 
                  metrics.scale_invariant_loss, metrics.abs_relative_diff, metrics.squared_relative_diff])

    predict_while_train = PredictWhileTrain(nyu_data_generator)
    if not os.path.isdir(PREDICT_FILE_PATH):
        os.mkdir(TRAIN_PREDICT_FILE_PATH)
    print('Fit model on training data')
    if RUN_REFINE:
        history = model.fit(x=nyu_data_generator,
                            epochs=10, callbacks=[cp_callback_refine, csv_logger, predict_while_train])
    else:
        history = model.fit(x=nyu_data_generator,
                            epochs=10, callbacks=[cp_callback_coarse, csv_logger, predict_while_train])

    print('\nHistory dict:', history.history)


    result = model.evaluate(x=nyu_data_generator, steps=1)
    print("Final eval loss: ", result)

    if not os.path.isdir(PREDICT_FILE_PATH):
        os.mkdir(PREDICT_FILE_PATH)

    predictions = model.predict(x=nyu_data_generator, steps=1)
    print("Prediction dim: " + str(predictions.shape))

    for i in range(predictions.shape[0]):
        predictions[i] = (predictions[i] / np.max(predictions[i])) * 255.0
        image_name = os.path.join(PREDICT_FILE_PATH, '%05d_d.png' % i)
        image_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)), mode="L")
        image_im.save(image_name)


if __name__ == '__main__':
    main()
