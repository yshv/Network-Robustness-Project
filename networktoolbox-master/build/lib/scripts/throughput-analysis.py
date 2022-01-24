import ray
import NetworkToolkit as nt
import networkx as nx
import numpy as np

@ray.remote
def recalculate_throughput(data, db="Topology_Data", collection=None):
    graph = nt.Tools.read_database_topology(data["topology data"], node_data=data["node data"], use_pickle=True)
    network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
    network.physical_layer.add_uniform_launch_power_to_links(network.channels)
    network.physical_layer.add_wavelengths_to_links(nt.Tools.read_database_dict(data["ILP-connections RWA"]))
    network.physical_layer.add_non_linear_NSR_to_links()
    max_capacity = network.physical_layer.get_lightpath_capacities_PLI(nt.Tools.read_database_dict(data["ILP-connections RWA"]))
    node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
    print("Capacity: {}".format(max_capacity[0]*1e-12))
    nt.Database.update_data_with_id(db, collection, data['_id'], newvals={"$set": {"ILP-connections node pair capacities": node_pair_capacities,
                                                                            "ILP-connections Capacity":max_capacity[0]}})

@ray.remote
def update_T_c(data, node_graph, db="Topology_Data", collection=None):
    top = nt.Topology.Topology()
    graph = nt.Tools.read_database_topology(data["topology data"], node_data=data["node data"], use_pickle=True)
    nx.set_node_attributes(graph, dict(node_graph.nodes.data()))
    graph = top.assign_distances_grid(graph, harvesine=True)
    for s,d in graph.edges:
        graph[s][d]["weight"] = np.ceil(graph[s][d]["weight"]*data["distance_scale"])
    topology_data = nt.Tools.graph_to_database_topology(graph, use_pickle=True)
    node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
    network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
    SNR_matrix = network.get_SNR_matrix()
    T_c = network.demand.bandwidth_2_connections(SNR_matrix, data["T_b"])
    nt.Database.update_data_with_id(db, collection, data["_id"], newvals={"$set": {"topology data":topology_data,
                                                                                         "node data":node_data,
                                                                                   # "T_b":T_b.tolist(),
                                                                                   "T_c":T_c.tolist()}})

def update_T_c_parralel(dataframe, node_graph, collection=None):
    ray.init(address='128.40.41.48:6379', _redis_password='5241590000000000', dashboard_port=8265)
    ray.get([update_T_c.remote(data, node_graph, collection=collection) for ind, data in dataframe.iterrows()])

def recalculate_throughput_parralel(dataframe, collection=None):
    ray.init(address='128.40.41.48:6379', _redis_password='5241590000000000', dashboard_port=8265)
    ray.get([recalculate_throughput.remote(data, collection=collection) for ind, data in dataframe.iterrows()])

if __name__ == "__main__":
    dataframe = nt.Database.read_data_into_pandas("Topology_Data",
                                                     "ta",
                                                     find_dic={"type" :"ga_lc","flag" :'feed tb','gamma':0})
    nsfnet = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "NSFNET"},
                                                    node_data=True)[0][0]
    print(len(dataframe))
    # recalculate_throughput_parralel(dataframe, collection="ta")
    update_T_c_parralel(dataframe, nsfnet, collection="ta")