import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import data
import numpy as np
import sklearn
import scipy
import h5py

data_path = "/scratch/datasets/MPNN/hdf5/other-regressors"
filename = "MPNN-uniform-25-45-ML-regressor-test"
model_name = "/ANN-regressor"

x_test, y_test = data.read_data(data_path, filename, normalise=True, remove_nan=True)
x_test, y_test = tf.convert_to_tensor(x_test, dtype=np.float32), tf.convert_to_tensor(y_test, dtype=np.float32)

model = tf.keras.models.load_model(data_path + model_name)

y_pred = model.predict(x_test).numpy()
y_test = y_test.numpy()
R_2 = sklearn.metrics.r2_score(y_test, y_pred)
p = scipy.stats.pearsonr(y_test, y_pred)


with open('{}{}-accuracy.txt'.format(data_path, model_name), 'w') as f:
    f.write("R2 \t p")
    f.write('\n')
    f.write('{} \t {}'.format(R_2, p))

with h5py.File('{}/{}.hdf5'.format(data_path, "ANN-raw"), 'w') as f:
    dset = f.create_dataset("raw", data=np.hstack((y_pred, y_test)))
