import numpy as np
# from geneticalgorithm import geneticalgorithm as ga
import NetworkToolkit as nt
import networkx as nx
import matplotlib.pyplot as plt
import socket

graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ta","T_c","DWC",
                                                    find_dic={"ga_topology":1},
                                                    use_pickle=True)

import os
# os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
            socket.gethostname().split('.')[0])
node_file_dir = "/scratch/datasets/gurobi/nodefiles"
# node_file_dir ="/Users/robin/OneDrive/OneDrive - University College London/Code/PycharmProjects/networktoolbox/scripts/nodefiles"
graph = graph_list[0][0]
network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
data = network.rwa.maximise_connection_demand(T_c=graph_list[0][2], max_time=24*3600,
                                              e=20,
                                              k=1,
                                              threads=80,
                                              node_file_dir=node_file_dir,
                                              node_file_start=0.01,
                                              emphasis=0,
                                              verbose=1,
                                              max_solutions=100)