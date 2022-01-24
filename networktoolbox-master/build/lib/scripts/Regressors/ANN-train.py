import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import data
import numpy as np

data_path = "/scratch/datasets/MPNN/hdf5/other-regressors"
filename = "MPNN-uniform-25-45-ML-regressor"

x_train, y_train = data.read_data(data_path, filename, normalise=True, remove_nan=True)
x_train, y_train = tf.convert_to_tensor(x_train, dtype=np.float32), tf.convert_to_tensor(y_train, dtype=np.float32)

model = keras.Sequential([
    layers.Dense(32, input_shape=(x_train.shape[1],), activation="tanh"),
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear"),
])

model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.MeanAbsoluteError(),
    # List of metrics to monitor
    metrics=[keras.metrics.MeanAbsoluteError()],
)

history = model.fit(x_train, y_train, epochs=2000, batch_size=64, shuffle=True, validation_split=0.1, verbose=1)
model.save(data_path+"/ANN-regressor")
