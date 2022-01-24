import h5py
import numpy as np

def read_data(path, filename, normalise=False, remove_nan=False, return_stats=False):
    with h5py.File('{}/{}.hdf5'.format(path, filename), 'r') as f:
        data = f['features']
        print(data)
        x = data[:,:-1]
        y = data[:,-1]
        mean = y.mean()
        std = y.std()
        if normalise:
            y = np.reshape((y - y.mean()) / y.std(), (len(y), 1))
        if remove_nan:
            x_delete_ind = []

            for ind, item in enumerate(x):
                if np.any(np.isnan(item)) == True:
                    x_delete_ind.append(ind)
            print(x_delete_ind)
            x = np.delete(x, x_delete_ind, 0)
            y = np.delete(y, x_delete_ind, 0)
    if return_stats:
        return x,y, mean, std
    else:
        return x,y

def denormalise_data(data, mean, std):
    data = (data*std)+mean
    return data


