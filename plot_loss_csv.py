import os
import numpy as np
import h5py
import random
import wget
from PIL import Image
import csv
import matplotlib.pyplot as plt


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
            val_loss.append(float(row[2]))


    print(loss)
    print(t)
    # return
    # axs.set_ylim(20, 0)
    # lst_iter = range(100)
    # lst_loss = [0.01 * i - 0.01 * i ** 2 for i in range(100)]
    # print(lst_loss)
    plt.plot(t, loss,  '-b', label='loss')
    plt.plot(t, val_loss,  '-r', label='val loss')
    plt.xlabel("n iteration")
    plt.ylim(top=0.8)
    plt.ylim(bottom=0.49)
    plt.legend(loc='upper left')
    plt.title("Loss")

    plt.show()


def main(argv=None):
    plot_csv(CSV_PATH)


if __name__ == '__main__':
    main()
