import os
import numpy as np
import h5py
import random
import wget
from PIL import Image
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


CSV_PATH = 'log.csv'

def plot_csv(path):
    loss = []
    val_loss = []
    t = []
    with open(CSV_PATH,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            if i == 0:
                i = i+1
                continue
            
            t.append(int(row[0]))
            loss.append(float(row[1]))
            # val_loss.append(float(row[2]))

    plt.plot(t, loss,  '-b', label='loss')
    # plt.plot(t, val_loss,  '-r', label='val loss')
    plt.xlabel("n iteration")
    plt.ylim(top=1)
    plt.ylim(bottom=0.3)
    plt.legend(loc='upper left')
    plt.title("Loss")

    plt.show()


def main(argv=None):
    plot_csv(CSV_PATH)


if __name__ == '__main__':
    main()
