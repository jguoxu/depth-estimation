
import metrics
import numpy as np
import tensorflow as tf
import math
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape

y_true = np.random.rand(1, 55, 74)
# y_pred = np.random.rand(1, 55, 74)

y_true_tensor = tf.constant(y_true, dtype='float32')
y_pred_tensor = tf.constant(y_true, dtype='float32')

print ("y_true_tensor: " + str(y_true_tensor))
print ("y_pred_tensor: " + str(y_pred_tensor))

abs_relative_diff  = metrics.abs_relative_diff(y_true_tensor, y_pred_tensor)
squared_relative_diff  = metrics.squared_relative_diff(y_true_tensor, y_pred_tensor)
rmse  = metrics.rmse(y_true_tensor, y_pred_tensor)
rmse_log  = metrics.rmse_log(y_true_tensor, y_pred_tensor)
rmse_scale_invariance_log  = metrics.rmse_scale_invariance_log(y_true_tensor, y_pred_tensor)

print ("abs_relative_diff: " + str(abs_relative_diff))
print ("squared_relative_diff: " + str(squared_relative_diff))
print ("rmse: " + str(rmse))
print ("rmse_log: " + str(rmse_log))
print ("rmse_scale_invariance_log: " + str(rmse_scale_invariance_log))