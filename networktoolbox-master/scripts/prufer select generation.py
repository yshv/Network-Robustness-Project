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

def prufer_sequence_length(N,length_limit,grid_graph):
    '''
    Method to design graph with prufer sequence
    :param N: node number
    :param length_limit: total_length
    :param T_C: traffic matrix
    :param grid_graph: node positions
    :return graph: The designed graph
    '''   
        
#     print(len(Prufer_sq))
    
    

    while True:
        graph = nx.Graph()
        graph.add_nodes_from(list(np.arange(1,N+1)))
        All_node = np.arange(1,N+1)
        
        Prufer_sq = list(np.random.randint(1,N+1,size = N-2))        
        eligible_nodes = list(set(All_node).difference(set(Prufer_sq)))


        sorted(eligible_nodes)

        leaf_nodes = eligible_nodes.copy()


        temp_sq = Prufer_sq.copy()

        for i in range(len(Prufer_sq)):

            graph.add_edge(eligible_nodes[0],temp_sq[0])
            eligible_nodes.pop(0)

            if len(temp_sq)>=2:
                if temp_sq[0] not in temp_sq[1:]:
                    eligible_nodes.append(temp_sq[0])
                    sorted(eligible_nodes)
            elif len(temp_sq) == 1:
                eligible_nodes.append(temp_sq[0])

            temp_sq.pop(0)

    #     print(temp_sq)
    #     print(eligible_nodes)

        graph.add_edge(eligible_nodes[0],eligible_nodes[1])

    #     print(leaf_nodes)
        for i in range(len(leaf_nodes)-1):
            graph.add_edge(leaf_nodes[i],leaf_nodes[i+1])

        graph.add_edge(leaf_nodes[0],leaf_nodes[-1])

    #     print(len(graph.edges))

        nx.set_node_attributes(graph, dict(grid_graph.nodes.data()))
        top = nt.Topology.Topology()
        graph = top.assign_distances_grid(graph, pythagorus=False,
                                           harvesine=True)
        total_length = np.array([graph[s][d]["weight"]*80 for s,d in graph.edges]).sum()
        
        if total_length <= length_limit:
            break
    
#     print(top.check_bridges(graph))
    
#     DWC = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], [T_C], alpha)[0]
    
    return graph,total_length


def prufer_select_generation(N, length_limit, T_c, gamma, grid_graph, topology_num, generate_num,write = False, Alpha = [1]):
    '''
    Method to select the topology with minimum DWC
    :param topology_number: the topology number we want to select from in each topology design
    :param generate_num: the number of topologies we want to generate    
    :param E: edge number
    :param T_C: traffic matrix
    :param grid_graph: node positions
    :param write: whether write the topologies into the database
    :return graph_result: The selected graphs and their DWCs

    '''
    
    graph_result = []
#     DWC_result = []
#     for j in range(generate_num):
#     DWC_max = 10000
    for i in range(topology_num): 
        

        graph,total_length = prufer_sequence_length(N,length_limit,grid_graph)
#         graph_result.append(graph)
        E = len(graph.edges())
        alpha = E/N/(N-1)*2
        DWC_structure = nt.Tools.get_demand_weighted_cost([[graph,0]], [T_c], Alpha,penalty_num=1000)[0]
        DWC_distance = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], [T_c], Alpha)[0]
        DWC = alpha*DWC_distance + (1-alpha)* DWC_structure
        
        if len(graph_result) < generate_num:
            graph_result.append((graph, DWC))
        else:
#             print(max(graph_result, key=lambda item: item[1]))
            (graph_worst, DWC_worst) = max(graph_result, key=lambda item: item[1])
            
            # index = graphs
            index = graph_result.index((graph_worst, DWC_worst))
            if DWC < DWC_worst:
                graph_result[index] = (graph, DWC)
                
        
    if write == True:
        for graph, DWC in graph_result:
            topology_data = nt.Tools.graph_to_database_topology(graph)
            node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
            nt.Database.insert_graph(graph, "Topology_Data", "prufer-select-distance", node_data=True, use_pickle=True,
                                     type = "prufer-select-new", T_c=T_c,
                                     DWC = DWC,alpha=Alpha, topology_data = topology_data,gamma=gamma,length_limit = length_limit)
            
    
    return graph_result

if __name__ == "__main__":
    gamma_value = np.around(np.arange(0, 1.1, 0.2),decimals=1)

    # gamma_value = [1]

    results = []
    for gamma in gamma_value:

        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "prufer-select-ga", 'T_c',node_data=True,
                                                        find_dic={'gamma': gamma},
                                                        max_count=1, use_pickle=True)
    #     dataset = nt.Database.read_data_into_pandas("Topology_Data", "prufer-select-ga",
    #                                                 find_dic={'gamma': gamma},
    #                                                 max_count=1)
        T_C = [x[2] for x in graph_list][0]

        N = 100
        E = 140
        topology_num = 10000
        avg_length = 2000
        grid_graph = graph_list[0][0]
        alpha = [1]
        length_limit = avg_length*E

        generate_num = 200

    #     results = distribute_func(prufer_random_generation, N,length_limit, T_C,gamma,grid_graph,topology_num,alpha,write=True,workers=50)
        results = prufer_select_generation(N, length_limit, T_C, gamma, grid_graph, topology_num, generate_num,write = True, Alpha = [1])