from sklearn.linear_model import ElasticNet
import data
import sklearn
import scipy
import numpy as np
import h5py

data_path = "/scratch/datasets/MPNN/hdf5/other-regressors"

filename="MPNN-uniform-ML-regressor"
filename_test="MPNN-uniform-ML-regressor-test"

x_train, y_train, mean, std = data.read_data(data_path, filename, normalise=True, remove_nan=True, return_stats=True)

Elasticnet = ElasticNet(alpha=0.0001, l1_ratio=0.8,normalize = False, max_iter=100000)

Elasticnet.fit(x_train, y_train)

x_test, y_test= data.read_data(data_path, filename_test, normalise=True, remove_nan=True)
y_pred = Elasticnet.predict(x_test)
print(np.shape(np.squeeze(y_test)))
print(np.shape(x_test))
print(np.shape(y_pred))
R_2 = sklearn.metrics.r2_score(np.squeeze(y_test), y_pred)
# p = scipy.stats.pearsonr(y_test, y_pred)
print(R_2)
print(data.denormalise_data(y_test, mean, std)[:5])
print(data.denormalise_data(y_pred, mean, std)[:5])
print(mean)
print(std)
# print(p)

# with open('{}{}-accuracy.txt'.format(data_path, "/ElasticNet"), 'w') as f:
#     f.write("R2 \t p")
#     f.write('\n')
#     f.write('{} \t {}'.format(R_2, p))
#
with h5py.File('{}/{}.hdf5'.format(data_path, "ElasticNet-raw"), 'w') as f:
    dset = f.create_dataset("raw", data=np.hstack((y_pred, y_test)))
