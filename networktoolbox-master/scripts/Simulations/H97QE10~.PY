import NetworkToolkit as nt
from tqdm import tqdm
import numpy as np
import ray
import ast
import socket
import os
import networkx as nx
import time
import datetime
import argparse
import MPNN
from simulation_base import *

parser = argparse.ArgumentParser(description='Optical Network Simulator Script')
parser.add_argument('-m', action='store', type=int, default=3000, help="How much memory to allocate to processes")
parser.add_argument('-cpu', action='store', type=int, default=1, help="How many cpus to use")
parser.add_argument('-mc', action='store', type=int, default=1000, help="How many samples to read")
parser.add_argument('--sleep', action='store', type=float, default=1000, help="How long to wait")
args = parser.parse_args()



if __name__ == "__main__":
    hostname="128.40.41.48"
    # scale_throughput()
    # graph, _id = nt.Database.read_topology_dataset_list("Topology_Data","real", find_dic={"name":
    #                                                                                  "30-Node-ONDPBook-Topology_nodes"})[0]
    # print('read graph, calculating chromatic number')
    # rwa, chrom_num = ILP_chromatic_number(graph,max_time=48*3600)
    # nt.Database.update_data_with_id("Topology_Data", "real",_id, newvals={"$set":{"wavelength requirement":chrom_num}})
    #     graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"BA",
    #                                                                                                      "nodes":14,
    #                                                                                                      "ILP Capacity":{
    #                                                                                                          "$exists":True}},
    # node_data=True, max_count=200)
    #     graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"ER",
    #                                                                                                      "nodes":14,
    #                                                                                                      "ILP Capacity":{
    #                                                                                                          "$exists":True}},
    #                                                         node_data=True, max_count=200)
    #     graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"SBAG",
    #                                                                                                      "nodes":14,
    #                                                                                                      "ILP Capacity":{
    #                                                                                                          "$exists":True}},
    #                                                         node_data=True,max_count=200)
    #     real_graph = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "NSFNET"},
    #                                                         node_data=True)[0][0]
    #     for gamma in np.around(np.arange(0, 1.1, 0.2), decimals=1):
    #         dataset_er = nt.Database.read_data_into_pandas("Topology_Data", "ta", find_dic={"type":"ER","purpose":"structural analysis","gamma":gamma})
    #         print(dataset_er[:1])
    #         T_c = dataset_er["traffic_matrix"][0]
    #         alpha = [0.66576701, 0.15450554, 0.05896479, 0.03678006, 0.02237619, 0.01477571,
    #                  0.01055498, 0.0077231,  0.00602809, 0.00448774, 0.00366258, 0.00290902,
    #                  0.00248457, 0.00198062, 0.00163766, 0.00139676, 0.0012504,  0.00101622,
    #                  0.00085918, 0.00083979]
    #         graphs = parralel_graph_generation_DWC(500000, 21, real_graph, workers=1000, T_c=T_c, port=6380, hostname=hostname, _alpha=0, alpha=alpha)
    #         print(len(graphs))
    #         # print(graph.number_of_edges())
    #         # print(DWC)
    #         for graph, DWC in graphs:
    #             nt.Database.insert_graph(graph, "Topology_Data", "ta", node_data=dict(graph.nodes.data()), use_pickle=True, type="BA", purpose="ga-analysis", gamma=gamma, T_c=T_c, DWC=DWC, data="new alpha")
    #     MPNN.create_graphs_MPNN(nodes=list(range(25, 50, 5)), graph_num=100, collection="MPNN-uniform-25-45-test",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname=hostname, amount=10, local=False)
    #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=100, collection="MPNN-uniform-55-100-test",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname=hostname, amount=10, local=False)
    #     ray.shutdown()
    #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=300, collection="MPNN-uniform-55-100",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname=hostname, amount=10, local=False)
    #     ray.shutdown()
    print("hello")
    # main(route_function="FF-kSP", collection="prufer-select-ga",
    #      query={"FF-kSP Capacity": {"$exists": True}}, hostname=hostname, port=7111,
    #      heuristic=True)
    # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
    #                                                     find_dic={"FF-kSP Capacity": {'$exists': True},
    #                                                               'nodes': 100},
    #                                                     max_count=1, use_pickle=True)

    # main(collection="routing-fibres-test-1",
    #      query={"nodes":10}, hostname=hostname, port=7112,
    #      ILP=True, fibre_num=1, ILP_threads=40)
    main(collection="routing-fibres-test-2",
         query={"nodes":10}, hostname=hostname, port=7112,
         ILP=True, fibre_num=2, ILP_threads=40)
