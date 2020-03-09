import os
import csv
import matplotlib.pyplot as plt

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float('bottom', 0.3, 'plot lower bound')
flags.DEFINE_float('top', 30.0, 'plot upper bound')
flags.DEFINE_string('file_path', 'log.csv', 'plot upper bound')

# Plot output csv.
# 
# Example content in the csv:
# epoch,loss,val_loss
# 0,335.93163042908884,710.1894700792101
# 1,286.35812078600185,135.08468543158637
# 2,244.79143238286863,39.02669249640571
# .....
#
# Require install absl https://github.com/abseil/abseil-py
# pip install absl-py
#
# Example command to run: python plot_loss_csv.py --top 30.0 --bottom=0.3 --file_path=log.csv
def plot_csv():
    labels = []
    columns = []
    with open(FLAGS.file_path , 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')

        i = 0
        for row in plots:
            # First row is labels
            if i == 0:
                for label in row:
                    labels.append(label)
                    columns.append([])
            else:
                for col_count in range(len(row)):
                    if col_count == 0:
                        # first colmn is always epoch count.
                        val = int(row[col_count])
                    else:
                        val = float(row[col_count])

                    columns[col_count].append(val)
            i = i + 1

    for i in range(1, len(labels)):
        plt.plot(columns[0], columns[i], label=labels[i])

    plt.xlabel("n epoch")
    plt.ylim(top=FLAGS.top)
    plt.ylim(bottom=FLAGS.bottom)

    plt.legend(loc='upper left')
    plt.title("Loss Metrics")

    plt.show()


def main(argv):
    plot_csv()


if __name__ == '__main__':
    app.run(main)
