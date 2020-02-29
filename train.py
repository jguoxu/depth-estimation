from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# from scipy import imageio
from PIL import Image
import numpy as np
# import tf.io.decode_png

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

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

            x_train.append(np.array(example))
            y_train.append(np.array(label))

    return x_train, y_train

    # # input
    # jpg = tf.read_file(filename)
    # image = tf.image.decode_jpeg(jpg, channels=3)
    # image = tf.cast(image, tf.float32)       
    # # target
    # depth_png = tf.read_file(depth_filename)
    # depth = tf.image.decode_png(depth_png, channels=1)
    # depth = tf.cast(depth, tf.float32)
    # depth = tf.div(depth, [255.0])
    # #depth = tf.cast(depth, tf.int64)
    # # resize
    # image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    # depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
    # invalid_depth = tf.sign(depth)
    # # generate batch
    # images, depths, invalid_depths = tf.train.batch(
    #     [image, depth, invalid_depth],
    #     batch_size=self.batch_size,
    #     num_threads=4,
    #     capacity= 50 + 3 * self.batch_size,
    # )
    # return images, depths, invalid_depths

def main():
    print(tf.__version__)

    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              # List of metrics to monitor
              metrics=['sparse_categorical_accuracy'])

    print('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=3,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(x_val, y_val))

    print('\nhistory dict:', history.history)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=128)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print('\n# Generate predictions for 3 samples')
    predictions = model.predict(x_test[:3])
    print('predictions shape:', predictions.shape)


def debug_display_rgbd_pair(rgb, d):
    img = Image.fromarray(rgb, 'RGB')
    img.show()

    # mode 'P' is 8bit pixel:
    # https://pillow.readthedocs.io/en/4.2.x/handbook/concepts.html#concept-modes
    img = Image.fromarray(d, 'P')
    img.show()

if __name__ == '__main__':
    # main()
    x_train, y_train = csv_inputs()

    # debug_display_rgbd_pair(x_train[0], y_train[0])
