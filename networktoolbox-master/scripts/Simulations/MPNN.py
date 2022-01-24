import NetworkToolkit as nt
from tqdm import tqdm
import numpy as np
import ray
import ast
import copy
import networkx as nx

@ray.remote
def create_SBAG_top(amount, grid_graph, alpha, db_name="Topology_Data",
                    collection_name="MPNN"):
    top = nt.Topology.Topology()
    for i in tqdm(range(amount)):
        graph = top.create_real_based_grid_graph(grid_graph, len(list(grid_graph.edges())),
                                                 database_name=db_name,
                                                 collection_name=collection_name,
                                                 sequential_adding=True,
                                                 undershoot=True,
                                                 remove_C1_C2_edges=True,
                                                 SBAG=True,
                                                 alpha=alpha
                                                 )
        nt.Database.insert_graph(graph, db_name=db_name, collection_name=collection_name,
                                 node_data=True, alpha=alpha, type="SBAG")

def create_graphs():
    grid_graph = nt.Database.read_topology_dataset_list("Topology_Data", "ECOC", find_dic={"name":
                                                                                               "NSFNET"},
                                                        node_data=True)[0][0]

    ray.init(address='auto', redis_password='5241590000000000')
    amount = 1000
    workers = 100
    data_len = int(amount/workers)
    rest = amount-data_len*workers
    amount_workers = [data_len for i in range(workers)]
    amount_workers[-1] = amount_workers[-1]+rest
    alpha = 5


    results = ray.get([create_SBAG_top.remote(_amount, grid_graph, alpha) for _amount in amount_workers])

def scale_graphs():
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN", find_dic={})
    print(len(graph_list))
    for graph, _id in tqdm(graph_list):
        for scale in [0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2.0]:
            graph_copy = copy.deepcopy(graph)
            for s,d in graph_copy.edges:
                graph_copy[s][d]["weight"] = np.ceil(scale*graph_copy[s][d]["weight"])

            nt.Database.insert_graph(graph_copy, db_name="Topology_Data", collection_name="MPNN",
                                     node_data=True, scale=scale, identifier=_id)

def create_throughput_data(workers=100):
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN", find_dic={"regression data "
                                                                                           "written":{"$exists":False}},
    max_count=1000)
    traffic_skew = np.arange(0.0, 1.1, 0.1)
    ray.init(address='auto', redis_password='5241590000000000')
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([route_graphs_skewed.remote(graph_list[indeces[i]:indeces[i+1]], traffic_skew) for i in range(
        workers)])


def scatter_nodes(N, radius=400, y=[31.687, 48.1261], x=[-124.8621, -67.3952]):
    """
    Method to scatter the nodes
    :param N:
    :param radius:
    :param y:
    :param x:
    :return:
    """
    nodes = []
    for n in range(N):
        distance_flag = True
        while distance_flag == True:
            y_pos = y[0] + np.random.random([1, ]) * (y[1] - y[0])
            x_pos = x[0] + np.random.random([1, ]) * (x[1] - x[0])
            _distances = np.array([nt.Topology.calculate_harvesine_distance(y_pos, y, x_pos, x) for y,x in nodes])
            _distance_filter = np.where(_distances < radius)

            distance_flag = np.any(_distance_filter)
        nodes.append((x_pos, y_pos))
    nodes_data = {ind + 1: {"Longitude": node[0][0], "Latitude": node[1][0]} for ind, node in enumerate(nodes)}
    return nodes_data

def create_graphs_MPNN(nodes=[10,15,20,25,30], graph_num=10, db_name="Topology_Data", collection="MPNN", scale_range=[0.6,0.8,1.0,1.2,1.4],
                       alpha_t_range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    edges = lambda N, i: np.ceil(N+0.2*N*i)
    alpha=5

    tasks=[]

    index = iter([i for i in range(graph_num*10*len(nodes)*len(scale_range)*len(alpha_t_range))])
    pbar=tqdm(total=graph_num*10*len(nodes)*len(scale_range)*len(alpha_t_range))
    # for N in nodes:
    #     for i in range(1,11):
    #         L = int(edges(N, i))
    #         for j in range(graph_num):
    #             print("N: {}".format(N))
    #             print("L: {}".format(L))
    #             # network = nt.Network.OpticalNetwork(new_graph, channel_bandwidth=16e9)
    #             # T_b = network.demand.create_uniform_bandwidth_normalised()
    #             # SNR_matrix = network.get_SNR_matrix()
    #             # T_c = network.demand.bandwidth_2_connections(SNR_matrix, T_b)
    #             #
    #             # nt.Database.insert_graph(graph, db_name=db_name, collection_name=collection,
    #             #                          node_data=True, alpha=5, type="SBAG", scale=1.0, T_b=T_b,T_c=T_c, alpha_t=0.0,
    #             #                          use_pickle=True)
    #             print("created graph - creating associated data")
    ray.init(address='auto', _redis_password='5241590000000000')
    ray.get([write_data.remote(N,int(edges(N,i)),scale, alpha=alpha, collection=collection, alpha_t=alpha_t, index=index.__next__())
             for N in nodes for i in range (1,11) for j in tqdm(range(graph_num)) for scale in scale_range for alpha_t in alpha_t_range])
                # for scale in scale_range:
                #     for alpha_t in alpha_t_range:

                        # tasks.append(write_data.remote(N,L,scale, alpha=alpha, collection=collection, alpha_t=alpha_t, index=index))
                        # index+=1



@ray.remote(num_cpus=1)
def write_data(N, L, scale, db_name="Topology_Data", collection="MPNN-scratch",
               alpha=5, alpha_t=0, index=1, pbar=None):
    # print("strating graph number {}".format(index))
    top = nt.Topology.Topology()
    node_attr = scatter_nodes(N, radius=100)
    graph = nx.Graph()
    graph.add_nodes_from(range(1, N + 1))
    nx.set_node_attributes(graph, node_attr)
    new_graph = top.create_real_based_grid_graph(graph, L,
                                                 database_name="Topology_Data",
                                                 collection_name="real",
                                                 sequential_adding=True,
                                                 undershoot=True,
                                                 remove_C1_C2_edges=True,
                                                 SBAG=True,
                                                 alpha=alpha)

    graph_copy = copy.deepcopy(new_graph)
    for s, d in graph_copy.edges:
        graph_copy[s][d]["weight"] = np.ceil(scale * graph_copy[s][d]["weight"])
    network = nt.Network.OpticalNetwork(new_graph, channel_bandwidth=16e9)
    SNR_matrix = network.get_SNR_matrix()

    T_skew = network.demand.create_skewed_demand(network.graph, alpha_t)
    T_c = network.demand.bandwidth_2_connections(SNR_matrix, T_skew)
    # degree data
    node_data = dict(new_graph.nodes.data())
    for node in node_data.keys(): node_data[node]["degree"] = dict(nx.degree(new_graph))[node]
    nx.set_node_attributes(new_graph, node_data)
    # traffic data
    T_b = np.array(T_skew).sum(axis=0)
    node_data = dict(new_graph.nodes.data())
    for node in node_data.keys(): node_data[node]["traffic"] = T_b[node - 1]
    nx.set_node_attributes(new_graph, node_data)
    # NSR data
    network = nt.Network.OpticalNetwork(new_graph, channel_bandwidth=16e9)
    network.physical_layer.add_wavelengths_full_occupation(network.channels)
    network.physical_layer.add_uniform_launch_power_to_links(network.channels)
    network.physical_layer.add_non_linear_NSR_to_links(channels_full=network.channels,
                                                       channel_bandwidth=network.channel_bandwidth)
    for s, d in network.graph.edges:
        new_graph[s][d]["worst case NSR"] = np.array(network.graph[s][d]["NSR"]).max()
    nt.Database.insert_graph(new_graph, db_name=db_name, collection_name=collection,
                             node_data=True, alpha=alpha, type="SBAG", scale=scale, T_b=T_skew.tolist(), T_c=T_c.tolist(),
                             alpha_t=alpha_t, new_data=1,NSR_written=1,
                             use_pickle=True)

    # print("graph {} done - N: {} L: {}".format(index, N, L))

@ray.remote
def route_graphs_skewed(graph_list, traffic_skew):

    data = []
    for graph, _id in tqdm(graph_list):
        for alpha in traffic_skew:
            graph_copy = copy.deepcopy(graph)
            traffic = nt.Demand.Demand.create_skewed_demand(graph_copy, alpha=alpha)
            data.append((graph_copy, traffic, alpha, _id))
    network_simulator = nt.NetworkSimulator.NetworkSimulator()
    # print(data)
    for graph, traffic, skew, _id in data:
        routing_data = network_simulator.incremental_non_uniform_demand_simulation(graph, traffic,
                                                                      routing_func="ILP-min-wave",
                                                                      channel_bandwidth=32e9,
                                                                      start_bandwidth=300e12,
                                                                      incremental_bandwidth=100e12,
                                                                      connections=False,
                                                                      e=5, max_count=50)
        rwa_assignment = {str(key): value for key, value in routing_data["rwa"].items()}
        nt.Database.insert_data("Topology_Data", "MPNN",{"topology data":nt.Tools.graph_to_database_topology(graph),
                                                         "node data":nt.Tools.node_data_to_database(dict(
                                                             graph.nodes.data())),
                                                         "throughput data":routing_data["capacity"][0],
                                                         "data type":"regression",
                                                         "traffic":traffic.tolist(),
                                                         "RWA assignment": rwa_assignment,
                                                         "traffic skew": skew})
    nt.Database.update_data_with_id("Topology_Data", "MPNN", _id, newvals={"$set":{"regression data written":True}})
        # print(routing_data)

if __name__ == "__main__":
    # create_graphs()
    # scale_graphs()
   # create_throughput_data(workers=50)
   #  create_graphs_MPNN(nodes=list(range(35,105,5)), graph_num=200, collection="MPNN-uniform", alpha_t_range=[0], scale_range=[1])
   #  create_graphs_MPNN(nodes=list(range(35, 105, 5)), graph_num=50, collection="MPNN-uniform-test", alpha_t_range=[0],
   #                     scale_range=[1])
    create_graphs_MPNN(nodes=list(range(11,15)), graph_num=100, collection="MPNN-uniform", alpha_t_range=[0], scale_range=[1])
