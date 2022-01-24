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
from networkx.algorithms import bipartite
import math
import random


def decide_gate_nodes_set(subgraph_i, subgraph_j, T_c, num_nodes):
    """
    Method to decide the gate nodes between two subgraphs in the HTD process.
    :param subgraph_i: subgraph for which the gate nodes are to be chosen
    :param subgraph_j: subgraph for which the target traffic is for
    :param T_c: normalised traffic matrix in terms of connections between the two subgraphs
    :param num_nodes: number of nodes to choose for - has to be less or the same as len(subgraph_i)
    :return: set of gate nodes from subgraph_i - list
    """

    # assert that the number of nodes to be
    # chosen is not larger than the subgraph
    # print(num_nodes)
    # print(subgraph_i)
    subgraph_i_len = len(subgraph_i)
    # assert num_nodes <= len(subgraph_i)
    traffic_sum = []
    for node_i in subgraph_i.nodes:
        traffic = 0
        for node_j in subgraph_j.nodes:
            # print(node_i)
            # print(node_j)
            traffic += T_c[node_i - 1][node_j - 1]
        traffic_sum.append((node_i, traffic))
    traffic_sum.sort(reverse=True, key=lambda x: x[1])
    gate_nodes = list(map(lambda x: x[0], traffic_sum))[:int(num_nodes)]
    return gate_nodes


def decide_gate_nodes_all(graph, subgraphs, T_c, gate_node_num):
    """
    Method to decide gate nodes of all subgraphs.
    :param graph: original graph (set of nodes) to design the network for
    :param subgraphs: set of individual subgraphs for which to do the inter and intra network designs
    :param T_c: normalised traffic matrix in terms of connections for the design process
    :return: list of gate nodes for the different subnetworks - list of list
    """
    gate_node_num = gate_node_num
    gate_nodes = []
    for i, subgraph_i in enumerate(subgraphs):
        for j, subgraph_j in enumerate(subgraphs):
            if i == j:
                pass
            else:
                gate_nodes.append(
                    decide_gate_nodes_set(subgraph_i, subgraph_j, T_c, gate_node_num[i, j]))
    return gate_nodes


def determine_inter_subnetwork_gate_edges_set(design_graph, subgraph_i, subgraph_j, gatenodes_i, gatenodes_j, edge_num):
    """
    Method to determine which gate nodes are connected to each other between two seperate subgraphs.
    :param edge_num: number of edges to add to the set
    :param subgraph_i: one of the subnetworks
    :param subgraph_j: the other subnetwork
    :param gatenodes_i: gate nodes to be used in subgraph_i and to be connected to some gate nodes in subgraph_j
    :param gatenodes_j: gate nodes to be used in subgraph_j and to be connected to some gate nodes in subgraph_j
    :return: edges of original nodes to be connected - list(tuples)
    """
    graph = nx.Graph()

    # Add nodes with the node attribute "bipartite"
    print(dict(subgraph_i.nodes.data()))
    for node in subgraph_i:
        graph.add_node(node, bipartite=0, Longitude=dict(subgraph_i.nodes.data())[node]["Longitude"],
                       Latitude=dict(subgraph_i.nodes.data())[node]["Latitude"])
    for node in subgraph_j:
        graph.add_node(node, bipartite=1, Longitude=dict(subgraph_j.nodes.data())[node]["Longitude"],
                       Latitude=dict(subgraph_j.nodes.data())[node]["Latitude"])
    # graph.add_nodes_from(list(subgraph_i.nodes()), bipartite=0)
    # graph.add_nodes_from(list(subgraph_j.nodes()), bipartite=1)
    graph.add_edges_from([(node_i, node_j) for node_i in subgraph_i.nodes for node_j in subgraph_j.nodes])
    assert len(graph.edges) >= edge_num
    # for node_i in subgraph_i.nodes:
    #     for node_j in subgraph_j.nodes
    b1, b2 = bipartite.sets(graph)
    graph.add_edges_from([(s, d) for s in b1 for d in b2])

    top_tools = nt.Topology.Topology()
    top_tools.assign_distances_grid(graph, harvesine=True)
    edges = [((s, d), graph[s][d]["weight"]) for s, d in graph.edges]
    edges.sort(key=lambda x: x[1])
    edges = list(map(lambda x: x[0], edges))
    add_edges = []
    for edge in edges:
        if edge not in list(design_graph.edges) and (edge[1],edge[0]) not in list(design_graph.edges):
            add_edges.append(edge)

        #         print(len(add_edges))
        #         print(int(edge_num))
        if len(add_edges) == int(edge_num):
            break
    return add_edges


def determine_inter_subnetwork_gate_edges_all(graph, subgraphs, gatenodes, edge_matrix):
    """
    Method to determine all the edges to connect the different gate nodes between the different subnetworks.
    :param gatenodes: gate nodes from all the subnetworks (list of list)
    :param graph: original graph (set of nodes) to design the network for
    :param subgraphs: set of individual subgraphs for which to do the inter and intra network designs
    :return: original graph with all edges added between the inter subnetworks
    """
    for i, subgraph_i in enumerate(subgraphs):
        for j, subgraph_j in enumerate(subgraphs):
            if subgraph_i == subgraph_j:
                pass
            else:
                edges = determine_inter_subnetwork_gate_edges_set(graph, subgraph_i, subgraph_j, gatenodes[i],
                                                                  gatenodes[j],
                                                                  edge_matrix[i, j])
                assert len(edges) == int(edge_matrix[i, j])
                assert edges not in list(graph.edges)
                graph.add_edges_from(edges)
    return graph


def subcenter_generation(N_sub):
    '''
    Method to calculate the center coordinate of the subgraphs
    :param N_sub: number of subnetworks
    :return sub_center: list of subnetwork center coordinates
    '''
    alpha = 2 * math.pi / N_sub
    sub_center = [[] for i in range(N_sub)]
    for i in range(N_sub):
        sub_center[i] = [math.cos(alpha * (i)), math.sin(alpha * (i))]

    return sub_center


def subnetwork_partition(grid_graph, N_sub, sub_center):
    '''
    Method to split nodes into subnetworks according to locations
    :param grid_graph: node positions
    :param N_sub: number of subnetworks
    :param sub_center: normalised center coordinate of the subnetworks
    :return Subnetwork : list of all subnetworks
    '''
    Subnetwork = [nx.Graph() for i in range(N_sub)]
    node_data = grid_graph.nodes.data()
    node_data_dict = dict(node_data)
    N = len(grid_graph.nodes)

    #     print(node_data)
    node_pos_dict = {}
    for node_id, item in node_data:
        node_pos_dict[node_id] = [item['Longitude'], item['Latitude']]

    #     print(node_pos_dict)

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

    normalized_pos = scaler.fit_transform(list(node_pos_dict.values()))

    #     print(normalized_pos)

    for index, item in enumerate(node_pos_dict.items()):

        node_pos_dict[item[0]] = normalized_pos[index]
        d = np.zeros(N_sub)
        for i in range(N_sub):
            d[i] = (sub_center[i][0] - normalized_pos[index][0]) ** 2 + (
                    sub_center[i][1] - normalized_pos[index][1]) ** 2
        k = np.argmin(d)
        # print(item[0])
        # print(node_data_dict[item[0]])
        Subnetwork[k].add_node(item[0], Longitude=node_data_dict[item[0]]["Longitude"],
                               Latitude=node_data_dict[item[0]]["Latitude"])
        # Subnetwork[k][item[0]]["Longitude"] = node_data_dict[item[0]]["Longitude"]
        # Subnetwork[k][item[0]]["Latitude"] = node_data_dict[item[0]]["Latitude"]

    #     print(node_pos_dict)
    #     x = [a[0] for a in normalized_pos]
    #     y = [a[1] for a in normalized_pos]
    #     plt.scatter (x,y)

    return Subnetwork


def edge_partition(Subnetworks, E, N_sub, T_C):
    '''
    Method to calculate edges between every subnetwork
    :param Subnetworks: list of all subnetworks
    :param E: number of edges
    :param N_sub: number of subnetworks
    :param T_C: normalized traffic matrix
    :return E_sub: E_sub[i][j] is the assigned edge number between subnetwork i and j
                   E_sub[i][i] is the assigned edge number within subnetwork i
    '''
    E_sub = np.zeros((N_sub, N_sub))
    E_inter = E-N

    for i in range(N_sub):
        for j in range(i, N_sub):
            if i == j:
                E_sub[i][j] = len(Subnetworks[i])

            else:
                Total_traffic = 0
                for p in Subnetworks[i].nodes:
                    for q in Subnetworks[j].nodes:
                        Total_traffic += T_C[p - 1][q - 1]

                E_sub[i][j] = np.round(Total_traffic * E_inter)

                E_sub[j][i] = E_sub[i][j]

    if sum(sum(np.array(E_sub))) != E:
        error = sum(sum(np.array(E_sub))) - E
        #         for i in range(N_sub):
        #             if error > 0:
        #                 E_sub[i][i] -= 1
        #                 error -= 1
        #             elif error < 0:
        #                 E_sub[i][i] += 1
        #                 error += 1

        while error != 0:
            if error > 0:
                #                 x = np.argmax(E_sub,axis=1)
                #                 E_sub[x[0]][x[1]] -= 1
                #                 E_sub[x[1]][x[0]] -= 1
                E_sub[np.where(E_sub == np.max(E_sub))[0][0]][np.where(E_sub == np.max(E_sub))[1][0]] -= 1
                #                 E_sub[np.where(E_sub == np.max(E_sub))[1][0]][np.where(E_sub == np.max(E_sub))[0][0]] -= 1
                error -= 1
            elif error < 0:
                E_sub[np.where(E_sub == np.min(E_sub))[0][0]][np.where(E_sub == np.min(E_sub))[1][0]] += 1
                #                 E_sub[np.where(E_sub == np.min(E_sub))[1][0]][np.where(E_sub == np.min(E_sub))[0][0]] += 1
                error += 1


    return E_sub.astype(int)


def relabel_nodes_to_original(subnetworks, mappings):
    """
    Method to relabel 1 indexed subnetworks to the original node labels
    :param subnetworks: list of subnetworks to relabel
    :param mappings:    mappings from 1 index to original node labels
    :return:            return the relabeled subnetworks.
    """
    subnetworks_relabelled = []
    for subnetwork, mapping in zip(subnetworks, mappings):
        subnetwork = nx.relabel_nodes(subnetwork, mapping)
        subnetworks_relabelled.append(subnetwork)
    return subnetworks_relabelled


def relabel_nodes_to_index(subnetworks):
    """
    Method to relabel the nodes from original node labels to those indexed from position 1.
    :param subnetworks: List of subnetworks to relabel.
    :return:    Return new list of subnetworks
    """
    return_list = []
    for subnetwork in subnetworks:
        nodes = list(subnetwork.nodes)
        mapping_to_original = {i: nodes[i - 1] for i in range(1, len(subnetwork) + 1)}
        mapping_to_index = {nodes[i - 1]: i for i in range(1, len(subnetwork) + 1)}
        subnetwork = nx.relabel_nodes(subnetwork, mapping_to_index)
        return_list.append((subnetwork, mapping_to_original))
    return return_list


def extract_T_c_subnetwork(subnetwork, T_c, mapping):
    """
    Method to extract a normalised traffic matrix of a subnetwork from the original T_c matrix.
    :param mapping:     Mapping of relabeled graphs (indexed 1) to the original node labels
    :param subnetwork:  The subnetwork for which to do this.
    :param T_c:         The original normalised traffic matrix.
    :return:            The new T_c, which is for the subnetwork based on the original T_c.
    """

    T_c = np.array(T_c)
    T_c_subnetwork = np.zeros((len(subnetwork), len(subnetwork)))
    for i in range(len(subnetwork)):
        for j in range(len(subnetwork)):
            T_c_subnetwork[i, j] = T_c[mapping[i + 1] - 1, mapping[j + 1] - 1]

    return T_c_subnetwork


def htd_network_design(graph, E, T_c, N_sub, alpha):
    """
    Method to input an original grid-graph to then use the hierarchical design process (HTD).
    :param alpha: The alpha variable for the dwc in ga process.
    :param N_sub: Number of subnetworks to split the original graph into.
    :param T_c: Normalised traffic matrix in terms of connections.
    :param graph: Original grid-graph for which to use the longitude and lattitude coordinates.
    :param E: Number of edges to use for the design process.
    :return: Designed graph (nx.Graph)
    """
    design_graph = nx.Graph()
    # Add existing nodes from grid graph
    design_graph.add_nodes_from(list(graph.nodes))
    # Add existing node coordinates and attributes
    nx.set_node_attributes(design_graph, dict(graph.nodes.data()))
    # Generate the subcenter of the network
    coordinate = subcenter_generation(N_sub)
    # Partition the network into its subnetworks for htd design
    subnetworks = subnetwork_partition(design_graph, N_sub, coordinate)
    # Allocate edge resources according the the traffic matrix
    edge_matrix = edge_partition(subnetworks, E, len(subnetworks), T_c)

    # Determine the gatenodes for the different subnetworks
    all_gatenodes = decide_gate_nodes_all(design_graph, subnetworks, T_c, edge_matrix)
    #     print(edge_matrix)
    #     print(all_gatenodes)
    #     print(len(all_gatenodes))

    #     exit()
    # Determine the edges that connect the different subnetworks
    design_graph = determine_inter_subnetwork_gate_edges_all(design_graph, subnetworks, all_gatenodes, edge_matrix)
    print(edge_matrix)
    print(edge_matrix.sum())
    print(design_graph.number_of_edges())
    #     print("design graph is connected: {}".format(nx.is_connected(design_graph)))
    #     print("number of connected components? {}".format(nx.number_connected_components(design_graph)))
    #     print(list(design_graph.degree(list(design_graph.nodes))))
    #     print(len(list(filter(lambda x: x[1] == 0, list(design_graph.degree(list(design_graph.nodes)))))))
    #     exit()
    # Relabel the graphs so that they start with index 1, and return the mappings to the original node labels
    subnetworks_mappings = relabel_nodes_to_index(subnetworks)
    subnetworks, mappings = list(map(lambda x: x[0], subnetworks_mappings)), list(
        map(lambda x: x[1], subnetworks_mappings))

    # Initialise ray instance
    ray.init()

    # Design the intra subnetworks with ga and distribute
    design_tasks = [
        GA_run.remote(len(subnetwork), edge_matrix[i, i], extract_T_c_subnetwork(subnetwork, T_c, mappings[i]), alpha,
                      subnetwork, 1, 0, False) for i, subnetwork in enumerate(subnetworks)]
    results = ray.get(design_tasks)
    ray.shutdown()
    # Add the edges from the intra subnetwork generation to original design graph and relabel the subnetworks with original node labels
    print("ga finished")
    subnetworks = [result[0]["graph"] for result in results]
    subnetworks = relabel_nodes_to_original(subnetworks, mappings)
    print("relabelling finished")

    for subnetwork in subnetworks:
        edges = list(subnetwork.edges)
        print(edge_matrix)
        print("edges of subnetwork: {}".format(len(edges)))
        design_graph.add_edges_from(edges)
    print("added edges from subnetworks")

    top = nt.Topology.Topology()
    design_graph = top.assign_distances_grid(design_graph, pythagorus=False,harvesine=True)


    return design_graph


# class PTD():
#     '''
#     Class of GA topology design
#     :param N: number of nodes
#     :param E: number of edges
#     :param T_c: normalized traffic matrix
#     :param alpha: weight list for kth shortest paths
#     :param grid_graph: the graph used to decide node locations
#     :return dictionary: {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}

#     '''

#     def __init__(self, N, E, T_c, alpha, grid_graph):
#         self.N = N
#         self.E = E
#         self.T_c = [T_c]
#         self.alpha = alpha
#         self.top = nt.Topology.Topology()
#         self.solution_graph = None
#         self.algorithm_param = {'max_num_iteration': 500,
#                                 'population_size': 500,
#                                 'mutation_probability': 0.1,
#                                 'elit_ratio': 0.01,
#                                 'crossover_probability': 0.8,
#                                 'parents_portion': 0.3,
#                                 'crossover_type': 'uniform',
#                                 'max_iteration_without_improv': None}
#         self.grid_graph = grid_graph

#     def build_graph_from_vector(self, graph_vector):
#         edges_poss = [(i + 1, j + 1) for i in range(self.N) for j in range(self.N) if j > i]
#         edges = []
#         for ind, item in enumerate(graph_vector):
#             if item == True:
#                 edges.append(edges_poss[ind])
#         graph = nx.Graph()
#         graph.add_nodes_from(list(range(1, self.N + 1)))
#         graph.add_edges_from(edges)
#         #         assign_distances(graph, self.grid_graph, write=False, collection=None, db="Topology_Data", _id=None)

#         nx.set_node_attributes(graph, dict(self.grid_graph.nodes.data()))
#         top = nt.Topology.Topology()
#         graph = top.assign_distances_grid(graph, pythagorus=False,
#                                           harvesine=True)
#         return graph

#     def objective(self, graph_vector):
#         objective_value = 0
#         # take vector
#         # create nx.graph
#         graph = self.build_graph_from_vector(graph_vector)
#         # print(graph.edges)
#         # print(graph.nodes)
#         #         if nx.is_connected(graph) and len(list(graph.edges)) == self.E and self.top.check_bridges(graph) and self.top.check_min_degree(graph):
#         if nx.is_connected(graph) and len(list(graph.edges)) == self.E and self.top.check_min_degree(graph):
#             objective_value = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], self.T_c, self.alpha)[0]
#         else:
#             objective_value += 1000

#         # Penalty function for graphs that don't meet this objective
#         # if not self.top.check_bridges(graph) or not self.top.check_min_degree(graph):
#         #     objective_value += 1000
#         # objective_value += graph.number_of_edges()
#         return objective_value

#     def run_ga(self):
#         self.varbound = np.array([[0, 1]] * int((self.N ** 2 - self.N) / 2))
#         self.model = ga(function=self.objective, dimension=int((self.N ** 2 - self.N) / 2), variable_type='bool',
#                         variable_boundaries=self.varbound, algorithm_parameters=self.algorithm_param)
#         self.model.run()
#         best_solution = self.get_solution()
#         objective_value = self.model.output_dict["function"]
#         solution_report = self.model.report
#         #         print(solution_report)
#         return {"graph": best_solution, "objective_value": objective_value, 'solution_report': solution_report}

#     def get_solution(self):
#         graph_vector = self.model.output_dict["variable"]
#         self.solution_graph = self.build_graph_from_vector(graph_vector)
#         return self.solution_graph


# def distribute_func(func, N, E, T_C, alpha, grid_graph, topology_num, gamma_value, write=False, workers=10):
#     '''
#     Method of distributing the GA function on mutiple servers
#     :param func: the running function
#     :param N: number of nodes
#     :param E: number of edges
#     :param T_c: normalized traffic matrix
#     :param alpha: weight list for kth shortest paths
#     :param grid_graph: the graph used to decide node locations
#     :param topology_num: the number of topologies need to be designed
#     :param gamma_value: the gamma_value which indicates the skew of the traffic
#     :param write: whether write the graph into the database
#     :param workers: the number of workers in running
#     :return dictionary: {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}
#     '''
#     # get indeces [0,1,2] for example for data 0-2
#     indeces = nt.Tools.create_start_stop_list(topology_num, workers)
#     #     print(indeces)
#     # Run all the ray instances
#     results = ray.get(
#         [func.remote(N, E, T_C, alpha, grid_graph, indeces[i + 1] - indeces[i], gamma_value, write) for i in
#          range(workers)])
#     return results


# @ray.remote
# def GA_run(N, E, T_C, alpha, grid_graph, topology_num, gamma_value, write):
#     Solutions = []
#     for i in range(topology_num):
#         ptd = PTD(N, E, T_C, alpha, grid_graph)
#         solution = ptd.run_ga()
#         Solutions.append(solution)
#         if write == True:
#             graph = solution['graph']
#             dwc = solution["objective_value"]
#             topology_data = nt.Tools.graph_to_database_topology(graph)
#             node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
#             nt.Database.insert_graph(graph, "Topology_Data", "ta", node_data=True, use_pickle=True, type="ga", T_c=T_C,
#                                      gamma=gamma_value, DWC_distance=dwc, purpose="ga-analysis", alpha=alpha,
#                                      topology_data=topology_data)

#     return Solutions

class PTD():
    '''
    Class of GA topology design
    :param N: number of nodes
    :param E: number of edges
    :param T_c: normalized traffic matrix
    :param alpha: weight list for kth shortest paths
    :param grid_graph: the graph used to decide node locations
    :return dictionary: {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}
    '''
    def __init__(self, N, E, T_c, alpha,grid_graph):
        self.N = N
        print(self.N)
        self.E = E
        print(self.E)
        self.T_c = [T_c]
        self.alpha = alpha
        self.top = nt.Topology.Topology()
        self.solution_graph = None
        self.algorithm_param =  {'max_num_iteration': 10,
                                 'population_size':10,
                                 'mutation_probability':0.1,
                                 'elit_ratio': 0.01,
                                 'crossover_probability': 0.8,
                                 'parents_portion': 0.3,
                                 'crossover_type':'uniform',
                                 'max_iteration_without_improv':None}
        self.grid_graph = grid_graph

    def build_graph_from_vector(self, Prufer_sq):
        #         edges_poss = [(i + 1, j + 1) for i in range(self.N) for j in range(self.N) if j > i]
        #         edges = []
        #         for ind, item in enumerate(graph_vector):
        #             if item == True:
        #                 edges.append(edges_poss[ind])
        graph = nx.Graph()
        graph.add_nodes_from(list(np.arange(1,self.N+1)))
        All_node = np.arange(1,self.N+1)

        #         while True:
        #         Prufer_sq = list(np.random.randint(1,N+1,size = N-2))
        Prufer_sq = list(Prufer_sq.astype(int))
        #         print(Prufer_sq)
        #         print(len(Prufer_sq))
        missing_nodes = list(set(All_node).difference(set(Prufer_sq)))
        #             if len(eligible_nodes) == E-(N-1):
        #                 break
        #         print(missing_nodes)
        #         print(Prufer_sq)

        new_Prufer_sq = list(dict.fromkeys(Prufer_sq))

        #         print(len(new_Prufer_sq))

        for i in range(len(missing_nodes)-2):
            new_Prufer_sq.append(missing_nodes[i])

        #         print(new_Prufer_sq)
        #         print(len(new_Prufer_sq))

        eligible_nodes = list(set(All_node).difference(set(new_Prufer_sq)))

        sorted(eligible_nodes)
        #         print(len(eligible_nodes))

        leaf_nodes = eligible_nodes.copy()


        temp_sq = list(new_Prufer_sq.copy())

        for i in range(len(new_Prufer_sq)):

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

        #         graph.add_edge(leaf_nodes[0],leaf_nodes[-1])

        #         print(len(graph.edges))

        nx.set_node_attributes(graph, dict(self.grid_graph.nodes.data()))
        top = nt.Topology.Topology()
        graph = top.assign_distances_grid(graph, pythagorus=False,
                                          harvesine=True)
        #         print(graph.edges.data())
        return graph

    #     def build_graph_from_vector(self, ring_sq):

    #         graph = nx.Graph()
    #         graph.add_nodes_from(list(np.arange(1,self.N+1)))
    #         All_node = np.arange(1,N+1)

    # #         print('ring_sq_len:{}'.format(len(ring_sq)))
    #         missing_nodes = list(set(All_node).difference(set(ring_sq)))
    #         print(missing_nodes)


    #         for i in range(len(ring_sq)-1):

    #             graph.add_edge(ring_sq[i],ring_sq[i+1])


    #         graph.add_edge(ring_sq[0],ring_sq[-1])

    # #         print(len(graph.edges))

    #         nx.set_node_attributes(graph, dict(grid_graph.nodes.data()))
    #         top = nt.Topology.Topology()
    #         graph = top.assign_distances_grid(graph, pythagorus=False,
    #                                            harvesine=True)
    # #         print(graph.edges.data())
    #         return graph

    def objective(self, graph_vector):
        objective_value = 0
        # take vector
        # create nx.graph
        graph = self.build_graph_from_vector(graph_vector)
        #         print(graph.edges)
        #         print(graph.nodes)
        #         if nx.is_connected(graph) and len(list(graph.edges)) == self.E and self.top.check_bridges(graph) and self.top.check_min_degree(graph):
        if len(list(graph.edges)) == self.E:
            objective_value = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], self.T_c, self.alpha)[0]
        else:
            objective_value+=1000

        # Penalty function for graphs that don't meet this objective
        # if not self.top.check_bridges(graph) or not self.top.check_min_degree(graph):
        #     objective_value += 1000
        # objective_value += graph.number_of_edges()
        return objective_value

    def pop_initial(self):
        initial_pop = []
        All_node = np.arange(1, self.N + 1)
        for i in range(self.algorithm_param['population_size']):
            ring_sq = list(random.sample(range(1, self.N + 1), self.N-2))
            #             print(len(ring_sq))
            initial_pop.append(ring_sq)
        return initial_pop

    def run_ga(self):
        self.varbound = np.array([[1,self.N]]*int(self.N-2))
        self.model = ga(function=self.objective, dimension=self.N-2,variable_type='int',
                        variable_boundaries=self.varbound, algorithm_parameters=self.algorithm_param,convergence_curve=False)
        pop = self.pop_initial()
        print(pop[0])

        self.model.run(pop)
        best_solution = self.get_solution()
        objective_value = self.model.output_dict["function"]
        solution_report = self.model.report
        #         print(solution_report)
        return {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}

    def get_solution(self):
        graph_vector = self.model.output_dict["variable"]
        self.solution_graph = self.build_graph_from_vector(graph_vector)
        return self.solution_graph

def distribute_func(func, N,E, T_C,alpha,grid_graph,topology_num,write=False,workers=10, gamma=None):
    '''
    Method of distributing the GA function on mutiple servers
    :param func: the running function
    :param N: number of nodes
    :param E: number of edges
    :param T_c: normalized traffic matrix
    :param alpha: weight list for kth shortest paths
    :param grid_graph: the graph used to decide node locations
    :param topology_num: the number of topologies need to be designed
    :param gamma_value: the gamma_value which indicates the skew of the traffic
    :param write: whether write the graph into the database
    :param workers: the number of workers in running
    :return dictionary: {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}
    '''
    # get indeces [0,1,2] for example for data 0-2
    indeces = nt.Tools.create_start_stop_list(topology_num, workers)
    #     print(indeces)
    # Run all the ray instances
    results = ray.get([func.remote(N,E, T_C,alpha,grid_graph,indeces[i+1]-indeces[i],write, gamma) for i in range(workers)])
    return results

@ray.remote
def GA_run(N,E, T_C,alpha,grid_graph,topology_num,write,gamma):

    Solutions = []
    for i in range(topology_num):
        ptd = PTD(N,E, T_C,alpha,grid_graph)
        solution = ptd.run_ga()
        print("finished running ga")
        Solutions.append(solution)
        if write == True:
            graph = solution['graph']
            dwc = solution["objective_value"]
            topology_data = nt.Tools.graph_to_database_topology(graph)
            node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
            nt.Database.insert_graph(graph, "Topology_Data", "prufer-select-ga", node_data=True, use_pickle=True,
                                     type = "prufer-select-ga", T_c=T_C.tolist(), DWC = dwc,alpha=alpha, topology_data = topology_data,
                                     gamma=gamma)

    return Solutions


if __name__ == "__main__":
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
                                                        find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
                                                        max_count=1, use_pickle=True)
    dataset = nt.Database.read_data_into_pandas("Topology_Data", "HTD-test",
                                                find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
                                                max_count=1)
    T_c = dataset["T_c"][0]

    N = dataset["nodes"][0]
    E = 140

    graph = graph_list[0][0]
    N_sub = 5
    alpha = [1]
    print(T_c)
    designed_graph = htd_network_design(graph, E, T_c, N_sub, alpha)

    print(len(designed_graph))
    print(designed_graph.number_of_edges())
    print(nt.Tools.get_demand_weighted_cost_distance([[designed_graph, 1]], [T_c], alpha)[0])
    print("graph is connected: {}".format(nx.is_connected(designed_graph)))
#     sub_center = subcenter_generation(N_sub)

#     Subnetworks = subnetwork_partition(grid_graph, N_sub, sub_center)

#     print([Subnetworks[i].nodes for i in range(N_sub)])

#     E_sub = edge_partition(Subnetworks, E, N_sub, T_C)
#     print(E_sub)

#     results = distribute_func(GA_run, N,E, T_C,alpha,grid_graph,topology_num,gamma_value,write=False,workers=10)
