# Depth Estimation
cs230 final project


# installation
```
pip install h5py
pip install Pillow
pip install matplotlib
```

https://matplotlib.org/2.0.0/users/installing.html
https://pillow.readthedocs.io/en/stable/installation.html


# Run Training

```
python prepare_data.py
python train.py
```


# Files
data_helper.py: prepare data converts NYU dataset from .mat file to a list of png pairs in data/train folder.
train.py: take the output from prepare_data.py and perform training.