import os
import numpy as np
import h5py
import random
import wget
from PIL import Image
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NYU_FILE_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
NYU_FILE_PATH = 'data/nyu_depth_v2_labeled.mat'
TRAIN_FILE_PATH = 'data/train'

DEV_PERCENT = 0.1 # split 10% data to dev examples
AUGMENTATION_COUNT = 4 # number of augmentations per image

def convert_nyu(path):
    if not os.path.isfile(path):
        print('File not exist: %s, starting download NYU dataset' % NYU_FILE_PATH)
        filename = wget.download(NYU_FILE_URL, out="data")
        print('\nDownloaded: ', filename)

    print('Loading dataset: %s' % (path))
    h5file = h5py.File(path, 'r')

    # # print all mat variabls:
    # # (u'depths', <HDF5 dataset "depths": shape (1449, 640, 480), type "<f4">)
    # # (u'images', <HDF5 dataset "images": shape (1449, 3, 640, 480), type "|u1">)
    # for item in h5file.items():
    #     print(str(item))
    # image = h5file['images'][0]

    # training example contain a list of rgb depth pairs.
    train_examples = []
    dev_examples = []

    if not os.path.isdir(TRAIN_FILE_PATH):
        os.mkdir(TRAIN_FILE_PATH)

    file_count = h5file['images'].shape[0]
    train_file_count = file_count * (1.0 - DEV_PERCENT)
    datagen = ImageDataGenerator()
    for i in range(file_count):
        image = np.transpose(h5file['images'][i], (2, 1, 0))
        depth = np.transpose(h5file['depths'][i], (1, 0))

        image_name = os.path.join(TRAIN_FILE_PATH, '%05d_c.png' % (i))
        depth_name = os.path.join(TRAIN_FILE_PATH, '%05d_d.png' % (i))

        # save to local png file.
        if not os.path.isfile(image_name):
            image_im = Image.fromarray(np.uint8(image))
            image_im.save(image_name)

        if not os.path.isfile(depth_name):
            scaled_depth = (depth / np.max(depth)) * 255.0
            depth_im = Image.fromarray(np.uint8(scaled_depth))
            depth_im.save(depth_name)

        for augment_count in range(AUGMENTATION_COUNT):
            brightness = random.uniform(0.7, 1.0)
            zoom_scale = random.uniform(0.7, 1.0)
            flip_horizontal = bool(random.getrandbits(1))

            augmented_im_bytes = datagen.apply_transform(x=image, transform_parameters={'brightness':brightness, 'zx':zoom_scale, 'zy':zoom_scale, 'flip_horizontal':flip_horizontal})
            augmented_im = Image.fromarray(np.uint8(augmented_im_bytes))
            agumented_image_name = os.path.join(TRAIN_FILE_PATH, '%05d_c_aug_%d.png' % (i, augment_count))
            augmented_im.save(agumented_image_name)

            # expand depth to 3 channels, keras apply_transform can only tranform 3 channel images.
            depth_multi_channel = np.array([depth, depth, depth])
            # tranpose depth image to (height, width, channel)
            depth_multi_channel = np.transpose(depth_multi_channel, (1, 2, 0))

            augmented_depth_bytes = datagen.apply_transform(x=depth_multi_channel, transform_parameters={'zx':zoom_scale, 'zy':zoom_scale, 'flip_horizontal':flip_horizontal})

            # get back single channel depth
            single_channel_aug_d = augmented_depth_bytes[:, :, 0]
            single_channel_aug_d = (single_channel_aug_d / np.max(single_channel_aug_d)) * 255.0
            augmented_depth = Image.fromarray(np.uint8(single_channel_aug_d))
            agumented_depth_name = os.path.join(TRAIN_FILE_PATH, '%05d_d_aug_%d.png' % (i, augment_count))
            augmented_depth.save(agumented_depth_name)
            train_examples.append((agumented_image_name, agumented_depth_name))

        if i < train_file_count:
            train_examples.append((image_name, depth_name))
        else:
            dev_examples.append((image_name, depth_name))

        print('Processed file: %i out of %d' % (i, file_count))

    random.shuffle(train_examples)
    # write train_examples to csv
    with open('data/train.csv', 'w') as output:
        for (image_name, depth_name) in train_examples:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

    with open('data/dev.csv', 'w') as output:
        for (image_name, depth_name) in dev_examples:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

def main(argv=None):
    convert_nyu(NYU_FILE_PATH)

if __name__ == '__main__':
    main()
