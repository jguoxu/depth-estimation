from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
import models

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
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


# def main():
#     print(tf.__version__)
#
#     # Pick a model (coarse or refined)
#     if (is_refine_training):
#         current_filepath = REFINED_CHECKPOINT_PATH,
#         coarse_training_results = models.model2()
#         model = models.refinedNetworkModel(coarse_training_results)
#     else:
#         current_filepath = COARSE_CHECKPOINT_PATH
#         model = models.model2()
#
#     # Create a callback that saves the model's weights with every epoch
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=current_filepath, save_weights_only=True,verbose=1, save_freq='epoch')
#
#     latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(current_filepath))
#
#     # If training refined for the first time, no checkpoint will be found in refined dir.
#     # Restore from coarse directory.
#     if is_refined_training and not latest_checkpoint:
#         latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(COARSE_CHECKPOINT_PATH))
#         print("\nGetting coarse checkpoint for refined training")
#         model.load_weights(latest_checkpoint)
#     elif (latest_checkpoint):
#         print("\nRestoring form checkpoint")
#         model.load_weights(latest_checkpoint)
#     else:
#         print("\nTraining model from scratch")
#
#     nyu_data_generator = NyuDepthGenerator(batch_size=10)
#
#     # parallel_model = multi_gpu_model(model, gpus=2)
#     # parallel_model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
#     #           # Loss function to minimize
#     #           loss=msr_loss,
#     #           # List of metrics to monitor
#     #           metrics=None)
#
#     model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
#               # Loss function to minimize
#               loss=depth_loss,
#               # List of metrics to monitor
#               metrics=None)
#
#     print("model metric names: " + str(model.metrics_names))
#
#     print('# Fit model on training data')
#     # when using data generate, x contains both X and Y.
#     # batch size is define in the generator thus passing None to batch_size
#     # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
#     history = model.fit(x=nyu_data_generator,
#                         epochs=10, callbacks=[cp_callback])#(x_val, y_val))
#
#     # history = model.fit_generator(nyu_data_generator, steps_per_epoch=5, epochs=1)
#
#     print('\nhistory dict:', history.history)
#
#     # Evaluate the model on the test data using `evaluate`
#     # print('\n# Evaluate on test data')
#     # results = model.evaluate(x_test, y_test, batch_size=128)
#     # print('test loss, test acc:', results)
#
#     # # Generate predictions (probabilities -- the output of the last layer)
#     # # on new data using `predict`
#     # print('\n# Generate predictions for 3 samples')
#     # predictions = model.predict(x_test[:3])
#     # print('predictions shape:', predictions.shape)

def main():
    print(tf.__version__)

    cp_callback_coarse = tf.keras.callbacks.ModelCheckpoint(filepath=COARSE_CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)
    cp_callback_refine = tf.keras.callbacks.ModelCheckpoint(filepath=REFINED_CHECKPOINT_PATH,
                                                            save_weights_only=True,
                                                            verbose=1)

    latest_checkpoint_refine = tf.train.latest_checkpoint(REFINED_CHECKPOINT_DIR)
    latest_checkpoint_coarse = tf.train.latest_checkpoint(COARSE_CHECKPOINT_DIR)
    refine_model, coarse_model = models.refine_model_def()
    # if RUN_REFINE:
    #     refine_model, coarse_model = models.refine_model_def()
    #     # model = refine_model
    #     # if latest_checkpoint_refine:
    #     #     print("\nRestored refine model from checkpoint")
    #     #     refine_model.load_weights(latest_checkpoint_refine)
    #     # elif latest_checkpoint_coarse:
    #     #     print("\nRestored coarse model from checkpoint")
    #     #     coarse_model.load_weights(latest_checkpoint_coarse)
    #     # else:
    #     #     print("\nCoarse model not restored. Please run coarse model first")
    #     #     return
    # else:
    #     coarse_model, _, _ = models.coarse_model_def()
    #     model = coarse_model
    #     if latest_checkpoint_coarse:
    #         print("\nRestored coarse model from checkpoint")
    #         coarse_model.load_weights(latest_checkpoint_coarse)
    #     else:
    #         print("\nNo coarse checkpoint saved")

    nyu_data_generator = NyuDepthGenerator(batch_size=10)

    refine_model.compile(optimizer=keras.optimizers.Adam(),  # Optimizer
                  # Loss function to minimize
                  loss=depth_loss,
                  # List of metrics to monitor
                  metrics=None)

    print('# Fit model on training data')
    if RUN_REFINE:
        history = refine_model.fit(x=nyu_data_generator,
                            epochs=300, callbacks=[cp_callback_refine])
    else:
        history = coarse_model.fit(x=nyu_data_generator,
                            epochs=300, callbacks=[cp_callback_coarse])
    print('\nhistory dict:', history.history)
    np.savetxt("loss_history.txt", history.history["loss"], delimiter=",")

    result = refine_model.evaluate(x=nyu_data_generator, steps=1)
    print("test loss: ", result)

    if not os.path.isdir(PREDICT_FILE_PATH):
        os.mkdir(PREDICT_FILE_PATH)
    predictions = refine_model.predict(x=nyu_data_generator, steps=1)
    print('predictions shape:', predictions.shape)
    for i in range(predictions.shape[0]):
        predictions[i] = (predictions[i] / np.max(predictions[i])) * 255.0
        image_name = os.path.join(PREDICT_FILE_PATH, '%05d_d.png' % i)
        image_im = Image.fromarray(np.uint8(predictions[i].reshape(TARGET_HEIGHT, TARGET_WIDTH)), mode="L")
        image_im.save(image_name)


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
