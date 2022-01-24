import ray
import NetworkToolkit as nt
import networkx as nx
import numpy as np
from tqdm import tqdm

def generate_SNR_embeddings(graph):
    network = nt.Network.OpticalNetwork(graph)
    SNR_matrix = network.get_SNR_matrix()
    np.fill_diagonal(SNR_matrix, 0)
    SNR_embedding = {node: SNR_matrix[node-1,:].tolist() for node in graph.nodes}
    nx.set_node_attributes(graph, SNR_embedding, "SNR")
    return graph

def distribute_func(func, graph_list, workers=1):
    # get indeces [0,1,2] for example for data 0-2
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    # Run all the ray instances
    results = ray.get([func.remote(graph_list[indeces[i]:indeces[i+1]]) for i in range(workers)])

@ray.remote
def generate_SNR_data(graph_list):
    for graph, _id in tqdm(graph_list):
        graph = generate_SNR_embeddings(graph)
        nt.Database.update_data_with_id("Topology_Data", "ILP-Regression", _id,
                                        {"$set":{"topology data":nt.Tools.graph_to_database_topology(graph),
                                                 "node data":nt.Tools.node_data_to_database(dict(graph.nodes.data())), "SNR written":True}})


if __name__ == "__main__":
    ray.init(address='auto', _redis_password='5241590000000000')
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ILP-Regression", find_dic={"SNR written": {
        "$exists": False}})
    distribute_func(generate_SNR_data, graph_list, workers=10)