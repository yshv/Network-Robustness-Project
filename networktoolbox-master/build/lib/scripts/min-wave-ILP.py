import ray
import NetworkToolkit as nt
import networkx as nx
import numpy as np
from tqdm import tqdm
import socket

def generate_SNR_embeddings(graph):
    network = nt.Network.OpticalNetwork(graph)
    SNR_matrix = network.get_SNR_matrix()
    np.fill_diagonal(SNR_matrix, 0)
    SNR_embedding = {node: SNR_matrix[node-1,:].tolist() for node in graph.nodes}
    nx.set_node_attributes(graph, SNR_embedding, "SNR")
#     print(graph.nodes.data())
    return graph

def distribute_func(func, graph_list, workers=1):
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    # ray.init()
    results = ray.get([func.remote(graph_list[indeces[i]:indeces[i+1]]) for i in range(workers)])
    # ray.shutdown()

@ray.remote
def generate_SNR_data(graph_list):
    for graph, _id in tqdm(graph_list):
        graph = generate_SNR_embeddings(graph)
        nt.Database.update_data_with_id("Topology_Data", "ILP-Regression", _id,
                                        {"$set":{"topology data":nt.Tools.graph_to_database_topology(graph),
                                                 "node data":nt.Tools.node_data_to_database(dict(graph.nodes.data())), "SNR written":True}})

@ray.remote
def generate_eigenvalue_data(graph_list):
    for graph, _id in tqdm(graph_list):
        node_data = dict(graph.nodes.data())
        eigenvalues = nx.normalized_laplacian_spectrum(graph)
        for node, eigenvalue in zip(graph.nodes, eigenvalues): node_data[node]["eigenvalue"] = eigenvalue
        nx.set_node_attributes(graph, node_data)
        nt.Database.update_data_with_id("Topology_Data", args.c, _id,
                                        {"$set": {"topology data": nt.Tools.graph_to_database_topology(graph),
                                                  "node data": nt.Tools.node_data_to_database(dict(graph.nodes.data())),
                                                  "eigenvalues written": 1}})

@ray.remote
def generate_eccentricity_data(graph_list):
    for graph, _id in tqdm(graph_list):
        node_data = dict(graph.nodes.data())
        for key, item in nx.eccentricity(graph).items(): node_data[key]["eccentricity"] = item
        nx.set_node_attributes(graph, node_data)
        nt.Database.update_data_with_id("Topology_Data", args.c, _id,
                                        {"$set": {"topology data": nt.Tools.graph_to_database_topology(graph),
                                                  "node data": nt.Tools.node_data_to_database(
                                                      dict(graph.nodes.data())),
                                                  "eccentricity written": 1}})
@ray.remote
def generate_NSR_data(graph_list):
    for graph, _id in tqdm(graph_list):
        try:
            print("calculating NSR values on : {}".format(socket.gethostname()))
            network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
            network.physical_layer.add_wavelengths_full_occupation(network.channels)
            network.physical_layer.add_uniform_launch_power_to_links(network.channels)
            network.physical_layer.add_non_linear_NSR_to_links(channels_full=network.channels,
                                                            channel_bandwidth=network.channel_bandwidth)
            for s, d in network.graph.edges:
                graph[s][d]["worst case NSR"] = np.array(network.graph[s][d]["NSR"]).max()
            topology_data=nt.Tools.graph_to_database_topology(graph, use_pickle=True)
            nt.Database.update_data_with_id("Topology_Data", args.c, _id,
                                            {"$set":{"topology data":topology_data,
                                                     "NSR written":1}})
        except Exception as err:
            print(err)

@ray.remote
def generate_degree_data(graph_list):
    for graph, _id in tqdm(graph_list):

        node_data = dict(graph.nodes.data())
        for node in node_data.keys(): node_data[node]["degree"] = dict(nx.degree(graph))[node]
        nx.set_node_attributes(graph, node_data)
        topology_data = nt.Tools.graph_to_database_topology(graph, use_pickle=True)
        node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
        nt.Database.update_data_with_id("Topology_Data", args.c, _id,
                                        {"$set":{"topology data":topology_data,
                                                 "node data":node_data,
                                                 "degree written":1}})

@ray.remote
def generate_traffic_data(graph_list):
    for graph,_id, T_b in tqdm(graph_list):
        T_b = np.array(T_b).sum(axis=0)
        node_data = dict(graph.nodes.data())
        for node in node_data.keys(): node_data[node]["traffic"]= T_b[node-1]
        nx.set_node_attributes(graph, node_data)
        topology_data = nt.Tools.graph_to_database_topology(graph, use_pickle=True)
        node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
        nt.Database.update_data_with_id("Topology_Data", args.c, _id,
                                        {"$set": {"topology data": topology_data,
                                                  "node data": node_data,
                                                  "traffic written": 1}})


@ray.remote
def re_write_topology_data(graph_list):
    for graph, _id in tqdm(graph_list):
        topology_data = nt.Tools.graph_to_database_topology(graph, use_pickle=True)
        node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
        nt.Database.update_data_with_id("Topology_Data", args.c, _id,
                                        {"$set": {"topology data": topology_data,
                                                  "node data": node_data,
                                                  "topology data written":1}})

def parralel_sim(graph_list, start_bandwidth, incremental_bandwidth, workers=8, collection=None):
    NetworkSimulator = ray.remote(nt.NetworkSimulator.NetworkSimulator)
    simulators = [NetworkSimulator.remote() for i in range(workers)]
    indices = nt.Tools.create_start_stop_list(len(graph_list), workers)
    results = ray.get([s.incremental_uniform_demand_simulation_graph_list.remote(graph_list,
                                                                                 start=indices[ind],
                                                                                 stop=indices[ind+1],
                                                                                 start_bandwidth=start_bandwidth,
                                                                                 incremental_bandwidth=incremental_bandwidth,
                                                                                 routing_func="ILP-min-wave",
                                                                                 collection_name=collection,
                                                                                 db_name="Topology_Data",
                                                                                 e=200,k=10) for ind,s in enumerate(simulators)])


# data_uni_band = run_ILP(graph_list,
#                         incremental_bandwidth=100000e9,
#                         start_bandwidth=100e12)
# ray.shutdown()
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-fd', nargs="+", default=None)
parser.add_argument('-w', action='store', type=int)
parser.add_argument('-mc', action='store', type=int, default=100000000)
parser.add_argument('-c', action='store', type=str,default=None)
parser.add_argument('-p', action='store', type=str,default=None)
args = parser.parse_args()
print(args.w)
print(args.fd)
find_dic = nt.Tools.create_find_dic(args.fd)
print(find_dic)

if args.p == "SNR":
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ILP-Regression", find_dic={"SNR written":{
    "$exists":False}}, max_count=args.mc, node_data=True)
    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000')
    distribute_func(generate_SNR_data, graph_list, workers=args.w)
if args.p == "NSR":
    print(args.c)
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c,
                                                        find_dic={"nodes": {"$gte": 10, "$lte": 15},
                                                                      "NSR written": {"$exists": False}},
                                                        max_count=args.mc, node_data=True, use_pickle=True)
    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000')
    # ray.init()
    if args.w > len(graph_list):
        workers = len(graph_list)
    else:
        workers = args.w

    while len(graph_list) !=0:
        distribute_func(generate_NSR_data, graph_list, workers=workers)
        # graph_list = [(item[0], item[1]) for item in tqdm(graph_list)]
        # distribute_func(generate_degree_data, graph_list, workers=args.w)
        # distribute_func(generate_NSR_data, graph_list, workers=args.w)

        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c,
                                                            find_dic={
                                                                      "nodes": {"$gte": 10, "$lte": 15},
                                                                      "NSR written": {"$exists": False}},
                                                            max_count=args.mc, node_data=True, use_pickle=True)
    ray.shutdown()
if args.p=="degree":
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c, find_dic={"degree written": 0},
                                                        max_count=args.mc, node_data=True, use_pickle=True)
    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000')

    while len(graph_list) !=0:
        distribute_func(generate_degree_data, graph_list, workers=args.w)
        # graph_list = [(item[0], item[1]) for item in tqdm(graph_list)]
        # distribute_func(generate_degree_data, graph_list, workers=args.w)
        # distribute_func(generate_NSR_data, graph_list, workers=args.w)

        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c, find_dic={"degree written": 0},
                                                            max_count=args.mc, node_data=True, use_pickle=True)
    # distribute_func(generate_NSR_data, graph_list, workers=args.w)
elif args.p == "ILP":
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ILP-Regression", find_dic={"ILP-min-wave Capacity":{"$exists":False}}, max_count=args.mc)
    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000')
    results = parralel_sim(graph_list, 100000e9, 100e12, workers=args.w, collection=args.c)
elif args.p == "topology-rewrite":
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c,
                                                        find_dic={},
                                                        max_count=args.mc, node_data=False, use_pickle=True)

    ray.init(address='auto', _redis_password='5241590000000000')
    distribute_func(re_write_topology_data,graph_list, workers=args.w)
elif args.p == "traffic":
    ray.init(address='auto', _redis_password='5241590000000000')
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c,
                                                        "T_b",
                                                        find_dic={"NSR written": 0},
                                                        max_count=args.mc, node_data=True, use_pickle=True)
    print(len(graph_list))
    while len(graph_list) !=0:
        distribute_func(generate_traffic_data, graph_list, workers=args.w)
        # graph_list = [(item[0], item[1]) for item in tqdm(graph_list)]
        # distribute_func(generate_degree_data, graph_list, workers=args.w)
        # distribute_func(generate_NSR_data, graph_list, workers=args.w)

        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c,
                                                            "T_b",
                                                            find_dic={"NSR written": 0},
                                                            max_count=args.mc, node_data=True, use_pickle=True)
elif args.p == "eigenvalues":
    ray.init(address='auto', _redis_password='5241590000000000')
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c, find_dic={"eigenvalues written":{"$exists":False}}, max_count=args.mc, node_data=True, use_pickle=True)
    while len(graph_list) != 0:
        distribute_func(generate_eigenvalue_data, graph_list, workers=args.w)
        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c,find_dic={"eigenvalues written":{"$exists":False}}, max_count=args.mc, node_data=True, use_pickle=True)
elif args.p == "eccentricity":
    ray.init(address='auto', _redis_password='5241590000000000')
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c, find_dic={"eccentricity written":{"$exists":False}}, max_count=args.mc, node_data=True, use_pickle=True)
    while len(graph_list) != 0:
        distribute_func(generate_eccentricity_data, graph_list, workers=args.w)
        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", args.c,find_dic={"eccentricity written":{"$exists":False}}, max_count=args.mc, node_data=True, use_pickle=True)
# "ILP-min-wave Capacity":{"$exists":False}
# "SNR written":{"$exists":False}
# graph_list = list(map(lambda x: x[0], graph_list))
# print(len(graph_list))
# ray.init(address='auto', redis_password='5241590000000000')

