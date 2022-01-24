import NetworkToolkit as nt
from NetworkToolkit import Data
from tqdm import tqdm
import ray
import networkx as nx
import numpy as np

hostname = "128.40.41.23"
port = 7112
collection = "MPNN-uniform"
max_count = 1000
find_dic = {"Weighted Spectrum Distribution":{"$exists":False}, "test data":{"$exists":False}, "ILP-connections":{"$exists":True}}
workers=100

@ray.remote
def update_graph_features(graph_list):

    funcs = [GP.update_m, GP.update_spanning_tree,
             GP.update_algebraic_connectivity, GP.update_node_variance, GP.update_mean_internodal_distance,
             GP.update_communicability_index, GP.update_comm_traff_ind, GP.update_graph_spectrum,
             GP.update_shortest_path_cost, GP.update_clustering_coefficient, GP.update_weighted_spectrum_distribution]
    for func in funcs:
        for graph in graph_list:
            func(graph, collection)



if __name__ == "__main__":
    ind = 0
    graph_list = [1]
    while len(graph_list) != 0:
        ind+=1

        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", collection, max_count=max_count, find_dic=find_dic)
        ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)
        GP = Data.GraphProperties()
        # workers=len(graph_list)
        indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
        tasks = [update_graph_features.remote(graph_list[indeces[ind]:indeces[ind+1]]) for ind in range(workers)]
        ray.get(tasks)

        # ray.get([func.remote(graph, collection) for func in funcs for graph in tqdm(graph_list)])
        print("graphs processed: {}".format(ind * max_count))
        ray.shutdown()
