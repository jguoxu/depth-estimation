import os
import numpy as np
import h5py
import random
import wget
from PIL import Image

NYU_FILE_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
NYU_FILE_PATH = 'data/nyu_depth_v2_labeled.mat'
TRAIN_FILE_PATH = 'data/train'
MAX_DEPTH_METER = 6.0
DEV_PERCENT = 0.1 # split 10% data to dev examples

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
    # # transpose to w, h, channel layout
    # image = np.transpose(image, (2, 1, 0))

    # depth = h5file['depths'][0]
    # # transpose to w, h layout
    # depth = np.transpose(depth, (1, 0))
    
    # _, axarr = plt.subplots(2, sharex=True)
    # axarr[0].imshow(image)
    # axarr[1].imshow(depth)
    # plt.show()

    # training example contain a list of rgb depth pairs.
    train_examples = []
    dev_examples = []

    if not os.path.isdir(TRAIN_FILE_PATH):
        os.mkdir(TRAIN_FILE_PATH)

    file_count = h5file['images'].shape[0]
    train_file_count = file_count * (1.0 - DEV_PERCENT)
    for i in range(file_count):
        image = np.transpose(h5file['images'][i], (2, 1, 0))
        depth = np.transpose(h5file['depths'][i], (1, 0))

        # Do not use max depth to scale, instead use constant MAX_DEPTH_METER
        # to make sure all depth image are distance is scaled proportionally.
        depth = (depth / np.max(depth)) * 255.0

        image_name = os.path.join(TRAIN_FILE_PATH, '%05d_c.png' % (i))
        depth_name = os.path.join(TRAIN_FILE_PATH, '%05d_d.png' % (i))

        # save to local png file.
        if not os.path.isfile(image_name):
            image_im = Image.fromarray(np.uint8(image))
            image_im.save(image_name)

        if not os.path.isfile(depth_name):
            depth_im = Image.fromarray(np.uint8(depth))
            depth_im.save(depth_name)

        if i < train_file_count:
            train_examples.append((image_name, depth_name))
        else:
            dev_examples.append((image_name, depth_name))

        print('Processed file: %i out of %d' % (i, file_count))

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
