import NetworkToolkit as nt
import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn
import time
from operator import itemgetter
from geneticalgorithm import geneticalgorithm as ga
import ray
import pandas as pd
import dask
from scipy.stats import spearmanr as spearman
from scipy.stats import kendalltau
import math


def prufer_sequence_ptd(N, E, grid_graph):
    '''
    Method to design graph with prufer sequence
    :param N: node number
    :param E: edge number
    :param T_C: traffic matrix
    :param grid_graph: node positions
    :return graph: The designed graph
    '''

    #     print(len(Prufer_sq))

    graph = nx.Graph()
    graph.add_nodes_from(list(np.arange(1, N + 1)))
    All_node = np.arange(1, N + 1)

    while True:
        Prufer_sq = list(np.random.randint(1, N + 1, size=N - 2))
        eligible_nodes = list(set(All_node).difference(set(Prufer_sq)))
        if len(eligible_nodes) == E - (N - 1):
            break

    sorted(eligible_nodes)

    leaf_nodes = eligible_nodes.copy()

    temp_sq = Prufer_sq.copy()

    for i in range(len(Prufer_sq)):

        graph.add_edge(eligible_nodes[0], temp_sq[0])
        eligible_nodes.pop(0)

        if len(temp_sq) >= 2:
            if temp_sq[0] not in temp_sq[1:]:
                eligible_nodes.append(temp_sq[0])
                sorted(eligible_nodes)
        elif len(temp_sq) == 1:
            eligible_nodes.append(temp_sq[0])

        temp_sq.pop(0)

    #     print(temp_sq)
    #     print(eligible_nodes)

    graph.add_edge(eligible_nodes[0], eligible_nodes[1])

    #     print(leaf_nodes)
    for i in range(len(leaf_nodes) - 1):
        graph.add_edge(leaf_nodes[i], leaf_nodes[i + 1])

    graph.add_edge(leaf_nodes[0], leaf_nodes[-1])

    #     print(len(graph.edges))

    nx.set_node_attributes(graph, dict(grid_graph.nodes.data()))
    top = nt.Topology.Topology()
    graph = top.assign_distances_grid(graph, pythagorus=False,
                                      harvesine=True)

    #     print(top.check_bridges(graph))

    #     DWC = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], [T_C], alpha)[0]

    return graph


def distribute_func(func, N, E, T_C, alpha, grid_graph, topology_num, generate_num, write=False, workers=10):
    '''
    Method of distributing the topology selection function on mutiple servers
    :param func: the running function
    :param N: number of nodes
    :param E: number of edges
    :param T_c: normalized traffic matrix
    :param alpha: weight list for kth shortest paths
    :param grid_graph: the graph used to decide node locations
    :param topology_num: the topology number we want to select from in each topology design
    :param generate_num: the number of topologies we want to generate
    :param write: whether write the graph into the database
    :param workers: the number of workers in running
    :return list: graph_result and DWC_result
    '''
    # get indeces [0,1,2] for example for data 0-2
    indeces = nt.Tools.create_start_stop_list(generate_num, workers)
    #     print(indeces)
    # Run all the ray instances
    results = ray.get(
        [func.remote(topology_num, N, E, T_C, grid_graph, indeces[i + 1] - indeces[i], write, alpha) for i in
         range(workers)])
    return results

@ray.remote
def topology_selection_random(topology_number, N, E, grid_graph, generate_num, write=False):
    """
    Method to generate prufer seueqnce topologies at random.
    :param topology_number: number of topologies to generate
    :param N:
    :param E:
    :param grid_graph:
    :param generate_num:
    :param write:
    :return:
    """


@ray.remote
def topology_selection(topology_number, N, E, T_C, grid_graph, generate_num, write=False, alpha=[1]):
    '''
    Method to select the topology with minimum DWC
    :param topology_number: the topology number we want to select from in each topology design
    :param E: edge number
    :param T_C: traffic matrix
    :param grid_graph: node positions
    :param generate_num: the number of topologies we want to generate
    :param write: whether write the topologies into the database
    :return graph_result: The selected graphs
    :return DWC_result: The DWCs of the selected graphs
    '''

    graph_result = []
    DWC_result = []
    for j in range(generate_num):
        DWC_min = 10000
        for i in range(topology_number):

            graph = prufer_sequence_ptd(N, E, grid_graph)
            DWC = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], [T_C], alpha)[0]

            if DWC < DWC_min:
                DWC_min = DWC
                final_graph = graph.copy()

        DWC_result.append(DWC_min)
        graph_result.append(final_graph)

        if write == True:
            topology_data = nt.Tools.graph_to_database_topology(final_graph)
            node_data = nt.Tools.node_data_to_database(dict(final_graph.nodes.data()), use_pickle=True)
            nt.Database.insert_graph(final_graph, "Topology_Data", "dwc-select-test", node_data=True, use_pickle=True,
                                     type="prufer-select", T_c=T_C,
                                     DWC=DWC_min, alpha=alpha, topology_data=topology_data, topology_num=topology_num, generate_num=generate_num)

    return graph_result, DWC_result


if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
    #                                                 find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
    #                                                 max_count=1, use_pickle=True)
    # dataset = nt.Database.read_data_into_pandas("Topology_Data", "HTD-test",
    #                                             find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
    #                                             max_count=1)
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "NSFNET"},
                                                        node_data=True, max_count=1)
    dataset = nt.Database.read_data_into_pandas("Topology_Data", "topology-paper", find_dic={"name": "NSFNET"},
                                                max_count=1)
    T_C = dataset["T_c"][0]

    N = dataset["nodes"][0]
    E = 140

    grid_graph = graph_list[0][0]

    topology_num = 10000
    generate_num = 200
    alpha = [1]
    #     graph, DWC = topology_selection(topology_number,N,E,T_C,grid_graph)

    results = distribute_func(topology_selection, N, E, T_C, alpha, grid_graph, topology_num, generate_num, write=True,
                              workers=50)
