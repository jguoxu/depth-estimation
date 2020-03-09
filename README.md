# Depth Estimation
cs230 final project


# installation
```
pip install h5py
pip install Pillow
pip install matplotlib
pip install wget
pip install absl-py
```

https://matplotlib.org/2.0.0/users/installing.html
https://pillow.readthedocs.io/en/stable/installation.html


# Run Training

```
python prepare_data.py
python train_in_place.py # or train.py
```

# Plot Metrics
```
python plot_loss_csv.py --top 30.0 --bottom=0.3 --file_path=log.csv
```

# Files
data_helper.py: prepare data converts NYU dataset from .mat file to a list of png pairs in data/train folder.

train.py: take the output from prepare_data.py and perform training.

train_in_place.py: take the output from prepare_data.py and perform training all in memory. Training is much faster all inputs are in memory, but this method doesn't work if train data is larger than memory size.

plot_loss_csv.py: plot log.csv output from after training finished.
