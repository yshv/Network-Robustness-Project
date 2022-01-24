import h5py
import NetworkToolkit as nt
import numpy as np

filename="MPNN-uniform-ML-regressor-test"
path = "/scratch/datasets/MPNN/hdf5/other-regressors"
collection = "MPNN-uniform-test"
max_count = 120000
find_dic = {"variance_sp_cost":{"$exists":True}, "ILP-connections Capacity":{"$exists":True}, "test data":{"$exists":False},
            "nodes":{"$lte":15}}

if __name__ == "__main__":
    features = ["m", "S", "algebraic connectivity", "degree variance", "mean internodal distance",
                "communicability distance", "communicability traffic index","average_sp_cost",
                "variance_sp_cost", "ILP-connections Capacity"]
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", collection, *features, find_dic=find_dic, max_count=max_count)
    eigenvalues = []
    data = np.squeeze(np.array([[graph[2:]] for graph in graph_list]))


    with h5py.File('{}/{}.hdf5'.format(path, filename), 'w') as f:
        dset = f.create_dataset("features", data=data)

    with h5py.File('{}/{}.hdf5'.format(path, filename), 'r') as f:
        data = f['features']
        print(data[:1])


