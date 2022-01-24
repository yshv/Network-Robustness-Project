import logging
import math
import sys
from functools import reduce
import NetworkToolkit.Database as Database
# import Database
import NetworkToolkit.Tools as Tools
# import Tools
# import dask
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# from sklearn.neighbors import KernelDensity
import copy
# from geneticalgorithm import geneticalgorithm as ga

import os

logger = logging.getLogger('Topology')
logger.setLevel(logging.INFO)

import NetworkToolkit as nt

# Useful lambda functions
_filter = lambda N_S, graph, max_degree: np.delete(
    np.asarray([0 if graph.degree[node] + 1 > max_degree else 1 for node in graph.nodes]),
    np.append(np.asarray([list(graph.nodes).index(
        neighbor) for neighbor in graph.neighbors(N_S)]),
        np.asarray(list(graph.nodes).index(N_S))).astype(int))

alpha_graph = lambda N_ACMN, L_ACMN: (2 * L_ACMN) / (
        N_ACMN * (N_ACMN - 1))  # lambda function for connectivity

degree_list = lambda graph: np.asarray(list(
    map(lambda x: x[1],
        list(graph.degree))))  # get list of degrees for all nodes
degree_list_NS = lambda N_S, graph: np.delete(np.asarray(list(
    map(lambda x: x[1],
        list(graph.degree)))), np.append(np.asarray([list(graph.nodes).index(
    neighbor) for neighbor in graph.neighbors(N_S)]),
    np.asarray(list(graph.nodes).index(N_S))).astype(int))  # get list of degrees for all nodes
degree_sum = lambda N_S, graph: np.sum(
    degree_list_NS(N_S, graph))  # get sum of degrees for all nodes
degree_prob = lambda graph: degree_list(
    graph) / degree_sum(
    graph)  # get probabilities of all nodes to be chosen to have a link added
N_S = lambda N, graph: (int(np.random.choice(np.arange(1, N + 1), p=degree_prob(
    graph))))  # randomly choose source node given probabilites (BA probabilities for source node)

distances_list = lambda N_S, graph: np.delete(np.asarray(
    [calculate_harvesine_distance(graph.nodes.data()[N_S]["Latitude"],
                                  # get a list of distances given the source node
                                  graph.nodes.data()[node]["Latitude"],
                                  graph.nodes.data()[N_S]["Longitude"],
                                  graph.nodes.data()[node]["Longitude"]) / 80
     for
     node in graph.nodes()]), np.append(np.asarray([list(graph.nodes).index(
    neighbor) for neighbor in graph.neighbors(N_S)]),
    np.asarray(list(graph.nodes).index(N_S))).astype(int))
distances_list_with_neighbours = lambda N_S, graph: np.delete(np.asarray(
    [calculate_harvesine_distance(graph.nodes.data()[N_S]["Latitude"],
                                  # get a list of distances given the source node
                                  graph.nodes.data()[node]["Latitude"],
                                  graph.nodes.data()[N_S]["Longitude"],
                                  graph.nodes.data()[node]["Longitude"]) / 80
     for
     node in graph.nodes()]),
    np.asarray(list(graph.nodes).index(N_S)))

distance_between_nodes = lambda N_S, N_D, graph: calculate_harvesine_distance(
    graph.nodes.data()[N_S]["Latitude"], graph.nodes.data()[N_D]["Latitude"],
    graph.nodes.data()[
        N_S]["Longitude"], graph.nodes.data()[N_D]["Longitude"])

distances_sum = lambda N_S, graph: np.sum(
    distances_list(N_S,
                   graph))  # get a sum of all distances given source node
distances_prob = lambda N_S, graph: distances_list(N_S, graph) / distances_sum(
    N_S,
    graph)  # get probabilities of all destination nodes to be picked


def N_D(N_S, N, graph, m, choice_prob, max_degree=9):
    """
    Method to choose nodes to add to a graph given the probabilities and source node.
    :param N_S:         Source node to add an edge to
    :param N:           Nodes to choose from
    :param graph:       graph to use
    :param m:           amount of edges to add
    :param choice_prob: function to choose the probabilties taking source node and graph as inputs
    :max_degree:        maximum degree to obey
    :return:            list of nodes to add
    """
    graph_copy = graph.copy()
    # choice_prob = lambda N_S, ACMN: degree_list_NS(N_S, ACMN) / \
    #                                 np.multiply(np.power(distances_list(
    #                                     N_S, ACMN), alpha),
    #                                     degree_sum(N_S, ACMN))

    # if math.isnan(choice_prob(N_S, graph)[0]):
    #     print("N_S: {}".format(N_S))
    #     print("N: {}".format(len(graph)))
    #     print("E: {}".format(graph.number_of_edges()))
    #     print(choice_prob(N_S, graph))
    #     print(graph.nodes.data())
    #     print("degree list all: {}".format(degree_list(graph)))
    #     print("degree list: {}".format(degree_list_NS(N_S, graph)))
    #     print("distances list: {}".format(distances_list(N_S, graph)))
    #     print("degree_sum: {}".format(degree_sum(N_S, graph)))
    #     print("m: {}".format(m))
    #     return None
    # else:
    #     print("N_S: {}".format(N_S))
    #     print("N: {}".format(len(graph)))
    #     print("E: {}".format(graph.number_of_edges()))
    #     print("choice prob: {}".format(choice_prob(N_S, graph)))
    #     print("node data: {}".format(graph.nodes.data()))
    #     print("degree list all: {}".format(degree_list(graph)))
    #     print("degree list: {}".format(degree_list_NS(N_S, graph)))
    #     print("distances list: {}".format(distances_list(N_S, graph)))
    #     print("degree_sum: {}".format(degree_sum(N_S, graph)))
    nodes = (np.random.choice(np.delete(N, np.append(np.asarray([list(graph.nodes).index(
        neighbor) for neighbor in graph.neighbors(N_S)]),
        list(graph.nodes).index(N_S)).astype(int)), size=m, replace=False,
                              p=choice_prob(N_S,
                                            graph)))  # randomly choose destination node
    # given probabilities
    for node in nodes:
        chosen_nodes = []
        while (graph_copy.degree[node] + 1) > max_degree:
            node = (
                np.random.choice(np.delete(N, np.append(np.asarray([list(graph.nodes).index(
                    neighbor) for neighbor in graph.neighbors(N_S)]),
                    list(graph.nodes).index(N_S))), size=1, replace=False,
                                 p=choice_prob(N_S,
                                               graph)))[0]  # randomly choose
            # destination node given probabilities
        assert graph.degree[node] + 1 <= max_degree
        graph_copy.add_edge(N_S, node)
        chosen_nodes.append(node)
    return np.asarray(chosen_nodes)[0]


def N_D_random(N_S, N, graph, m, choice_prob, max_degree=9):
    """
    Method to choose the destiantion node ranodmly:
    :param N_S:         Source node to add an edge to - int
    :param N:           Nodes to allow for selection - list
    :param graph:       Graph to use for probabilties - nx.Graph()
    :param m:           Amount of edges to add - int
    :param choice_prob: Function to calculate probabilities - function(N_S, graph)
    :param max_degree:  Maximum degree to obey - int
    :return:            list of nodes to add

    """
    graph_copy = graph.copy()
    nodes = (np.random.choice(np.delete(N, np.append(np.asarray([list(graph.nodes).index(
        neighbor) for neighbor in graph.neighbors(N_S)]),
        list(graph.nodes).index(N_S))), size=m, replace=False,
                              ))  # randomly choose destination node given uniform probabilities
    for node in nodes:
        chosen_nodes = []
        while (graph_copy.degree[node] + 1) > max_degree:
            node = (np.random.choice(np.delete(N, np.append(
                np.asarray([list(graph.nodes).index(
                    neighbor) for neighbor in graph.neighbors(N_S)]),
                list(graph.nodes).index(N_S))), size=1, replace=False,
                                     ))[0]  # randomly choose destination node given uniform
            # probabilities
        assert graph.degree[node] + 1 <= max_degree
        graph_copy.add_edge(N_S, node)
        chosen_nodes.append(node)
    return np.asarray(chosen_nodes)[0]


def choose_random_source(graph, max_degree=9):
    """
    Method to choose a random source node from a graph obeying maximum degree.
    :param graph:       Graph to choose from - nx.Graph()
    :param max_degree:  Maximum degree to obey - int
    :return:            Source node - int
    """
    source = np.random.choice(np.arange(1, len(graph) + 1),
                              1)[0]
    while graph.degree[source] + 1 > max_degree:
        source = np.random.choice(np.arange(1, len(graph) + 1), 1)[0]
    return source


def remove_bridge_components(graph, max_degree=9):
    """
    Method to find bridges in a graph and add edges to remove these components so that
    there are only bi-connected graphs.
    :param graph:       networkx graph for which to do this - nx.Graph()
    :param max_degree:  Maximum degree to obey - int
    :return:            graph - nx.Graph()
    """
    # find subgraphs that have a bridge
    bridge_sub_graphs = nx.k_edge_components(
        graph, k=2)
    # sort these in reverse order according to size of subgraph
    bridge_sub_graphs = sorted(bridge_sub_graphs, key=lambda x: len(x), reverse=True)

    # while there are still bridge components
    while len(bridge_sub_graphs) > 1:
        # create list for closest sub graphs
        closest_sub_graphs = {}

        # iterate through the different sub graphs
        for _ind, sub_graph in enumerate(bridge_sub_graphs):
            # get a list of indeces without the current sub graph
            sub_graph_ind = list(range(len(bridge_sub_graphs)))
            sub_graph_ind.remove(_ind)

            distances = []

            # iterate through current subgraph
            for node in sub_graph:
                # iterate through every other subgraph and its nodes
                for i in sub_graph_ind:
                    for node_other in bridge_sub_graphs[i]:
                        # find there distances
                        if (node, node_other) in list(graph.edges) or (node_other,
                                                                       node) in list(
                            graph.edges):
                            continue
                        distances.append((distance_between_nodes(node, node_other,
                                                                 graph), i, node, node_other))

            # sort these distances
            distances = sorted(distances, key=lambda x: x[0])
            # find the closest sub graph
            closest_sub_graph = distances[0][1]
            # find the distance of this subgraph
            closest_distance = distances[0][0]
            # find the edge that constitues this distance
            closest_edge = (distances[0][2], distances[0][3])
            closest_sub_graphs[_ind] = {"subgraph": closest_sub_graph,
                                        "distance": closest_distance, "edge": closest_edge}

        closest_sub_graph_distances = [closest_sub_graphs[key]["distance"] for key in
                                       closest_sub_graphs.keys()]
        closest_sub_graph_indeces = np.argsort(closest_sub_graph_distances)
        closest_subgraph_edge = np.asarray([closest_sub_graphs[key]["edge"] for key
                                            in closest_sub_graphs.keys()])[
            closest_sub_graph_indeces]
        if graph.degree[closest_subgraph_edge[0][0]] + 1 > max_degree or graph.degree[
            closest_subgraph_edge[0][1]] + 1 > max_degree:
            continue
        graph.add_edge(closest_subgraph_edge[0][0], closest_subgraph_edge[0][1])
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("edges added: {}-{}".format(closest_subgraph_edge[0][0],
                                                     closest_subgraph_edge[0][1]))
            logger.debug("removing bridges")
            plt.figure(figsize=(16, 10))
            nt.Plotting.plot_graph(graph, with_pos=True, with_labels=True, node_size=6)
        # update the bridge components
        bridge_sub_graphs = nx.k_edge_components(
            graph, k=2)
        bridge_sub_graphs = sorted(bridge_sub_graphs, key=lambda x: len(x),
                                   reverse=True)
        logger.debug("new bridge subgraphs: {}".format(bridge_sub_graphs))


def create_node_order(graph, random_adding=True, sequential_adding=False,
                      random_start_node=False, centre_start_node=False,
                      first_start_node=True, numeric_adding=False):
    """
    Method to create the node order for creating BA style graphs.
    :param graph:               Graph to use for creating the node order - nx.Graph()
    :param random_adding:       Adding nodes at random - Boolean
    :param sequential_adding:   Adding nodes in sequential order by adding the node that is closest to all nodes in that graph at a specific timestep - Boolean
    :param random_start_node:   Start node is randomly chosen - Boolean
    :param cetre_start_node:    Start node is chosen as the node which is closest to the centroid of the graph - Boolean
    :param first_start_node:    Start node is chosen to be the node which is labeled as node 1
    :param numeric_adding:      Nodes are added by following numeric order (node1 - 2 - 3 ...)
    :return:                    List of nodes in order of adding - list
    """
    graph_new = nx.Graph()

    if random_start_node:
        start_node = choose_starting_node_random(graph)
    elif centre_start_node:
        start_node = choose_starting_node_centroid(graph)
    else:
        start_node = 1

    if sequential_adding:
        nodes = []
        nodes.append(start_node)
        graph_new.add_node(start_node)  # add the node to the graph
        node_attr = [(_node, graph.nodes.data()[_node]) for _node in
                     graph_new.nodes]  # copy attributes etc... see above
        node_attr = dict(node_attr)
        nx.set_node_attributes(graph_new,
                               node_attr)  # setting the copied node attributes
        for i in range(len(list(graph.nodes)) - 1):
            node = choose_source_node_sequential(graph, graph_new)
            # print("node: {}".format(node))
            graph_new.add_node(node)  # add the node to the graph
            node_attr = [(_node, graph.nodes.data()[_node]) for _node in
                         graph_new.nodes]  # copy attributes etc... see above
            node_attr = dict(node_attr)
            nx.set_node_attributes(graph_new,
                                   node_attr)  # setting the copied node attributes
            nodes.append(node)
    elif random_adding:

        nodes_to_add = list(graph.nodes)
        nodes_to_add.remove(start_node)
        nodes = np.random.choice(nodes_to_add, len(nodes_to_add), replace=False)
        nodes = nodes.tolist()
        nodes = [start_node] + nodes
    elif numeric_adding:
        nodes = list(range(1, len(graph) + 1))

    return nodes


def remove_edges_C1_C2(graph):
    """
    Method to create list of edges that are safe to remove without violating min degree and bridge constarints
    of optical networks.
    :param graph:   graph to get the list of edges from - nx.Graph()
    :return:        list of edges that are safe to remove - list
    """
    edges_to_remove = list(graph.edges)
    degrees = degree_list(graph)  # get the up to date degree of each node
    nodes = np.flip(np.argsort(
        degrees)) + 1  # sort the arguments, flip them so that largest degrees are at the beginning, add 1 due to indexing...
    nodes_safe = np.where(degrees <= 2)[
                     0] + 1
    edges_to_remove = list(filter(lambda x: x[0] not in nodes_safe,
                                  edges_to_remove))  #  filter edges including safe nodes out
    edges_to_remove = list(filter(lambda x: x[1] not in nodes_safe,
                                  edges_to_remove))
    edges_to_remove = list(filter(lambda x:
                                  nx.algorithms.connectivity.is_locally_k_edge_connected(graph, x[0], x[1], k=3),
                                  edges_to_remove))
    return edges_to_remove


def remove_edges(graph, remove_len, remove_func):
    """
    Method to remove edges, these edges are chosen by a particular function.
    :param graph:       Graph from which to remove the edges - nx.Graph()
    :param remove_len:  Amount of edges to remove - int
    :param remove_func: Function to get the list of edges to remove - function(graph) return edge_to_remove
    :return:            None
    """
    for i in range(remove_len):
        edges_to_remove = remove_func(graph.copy())
        logger.debug("edges of graph: {}".format(edges_to_remove))
        logger.debug("edges in remove list: {}".format(len(edges_to_remove)))
        logger.debug("remove length: {}".format(remove_len - i))
        degrees = degree_list(graph)  # get the up to date degree of each node
        nodes = np.flip(np.argsort(
            degrees)) + 1  # sort the arguments, flip them so that largest degrees are at the beginning, add 1 due to indexing...
        nodes_safe = np.where(degrees <= 2)[
                         0] + 1  # update the nodes that are safe from edge removal(degree of 2 or less)
        logger.debug("nodes safe: {}".format(nodes_safe))

        # for node in nodes:
        #    edges_to_remove = list(filter(lambda x: x[0] == node, edges_to_remove))
        #    edges_to_remove = list(filter(lambda x: x[1] == node, edges_to_remove))

        edges_to_remove = list(filter(lambda x: x[0] not in nodes_safe,
                                      edges_to_remove))  #  filter edges including safe nodes out
        edges_to_remove = list(filter(lambda x: x[1] not in nodes_safe,
                                      edges_to_remove))
        logger.debug("edges of graph: {}".format(edges_to_remove))
        # print("edges to remove: {}".format(edges_to_remove))
        # print("edges to remove: {}".format(edges_to_remove))
        try:
            graph.remove_edge(edges_to_remove[0][0], edges_to_remove[0][1])
        except:
            return False


def get_centroid(graph, nodes):
    """
    Method to find the centroid between all of the nodes given in a graph.
    :param graph:   Graph to use for the longitude and latitude values - nx.Graph()
    :param nodes:   Nodes to use for the centroid calculation - list
    :return:        Mean longitude and latitude - tuple
    """
    longitude, latitude = [], []
    for node in nodes:
        longitude.append(graph.nodes.data()[node]["Longitude"])
        latitude.append(graph.nodes.data()[node]["Latitude"])
    mean_lon = np.mean(longitude)
    mean_lat = np.mean(latitude)
    return mean_lon, mean_lat


def choose_starting_node_random(graph):
    """
    Method to choose a starting node at random.
    :param graph:   Graph from which to choose the starting node - nx.Graph()
    :return:        start node - int
    """
    nodes_list = list(graph.nodes)
    node = np.random.choice(nodes_list, 1)
    return node[0]


def choose_starting_node_centroid(graph):
    """
    Method to choose a starting node that is closest to the centroid of all
    coordinates of the graph.
    :param graph:   Graph from which to pick the starting node - nx.Graph()
    :return:        Starting node - int
    """
    graph = graph.copy()
    nodes = np.asarray(list(graph.nodes))
    lon, lat = get_centroid(graph, graph.nodes)
    distances = distances_list_coordinate(lat, lon, graph)
    distances_sorted = np.argsort(distances)
    nodes_sorted = [list(graph.nodes)[ind] for ind in distances_sorted]
    return nodes_sorted[0]


def find_links_to_remove(graph, cutoff=2, SNR_thresh=0.5):
    """
    Method to find unlikely links for an optical network in a graph.
    :param graph:       Graph to scan through - nx.Graph()
    :param cutoff:      Cutoff value for paths - int
    :param SNR_thresh:  Threshold value for SNR loss in db - default = 0.5dB - float
    :return:            Edges that are unlikely to occur due to neglible SNR loss - list
    """
    distances = []
    unfeasible_edges = []
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            distance_direct = calculate_harvesine_distance(graph.nodes.data()[node][
                                                               "Latitude"],
                                                           graph.nodes.data()[
                                                               neighbor]["Latitude"],
                                                           graph.nodes.data()[node][
                                                               "Longitude"],
                                                           graph.nodes.data()[
                                                               neighbor]["Longitude"])
            span_direct = distance_direct / 80
            paths = nx.all_simple_paths(graph, node, neighbor, cutoff=cutoff)
            for path in paths:
                for ind, _node in enumerate(path):
                    if ind == 0:
                        continue
                    distances.append(calculate_harvesine_distance(graph.nodes.data()[
                                                                      path[ind - 1]][
                                                                      "Latitude"],
                                                                  graph.nodes.data()[
                                                                      path[ind]][
                                                                      "Latitude"],
                                                                  graph.nodes.data()[
                                                                      path[
                                                                          ind - 1]][
                                                                      "Longitude"],
                                                                  graph.nodes.data()[
                                                                      path[ind]][
                                                                      "Longitude"]))
                summed_distances = np.asarray(distances).sum()
                span_sum = summed_distances / 80
                if 10 * np.log10(span_sum / span_direct) > SNR_thresh:
                    pass  # way around gives large SNR penalties -> direct connection
                    # is possible
                else:
                    unfeasible_edges.append((node, neighbor))  # way around does not
                    # give SNR benefits
    return unfeasible_edges


def scatter_nodes(nodes, scale_lon=(-180, 180), scale_lat=(-90, 90), _mean=None,
                  _std=None, uniform=False):
    """
    Method for scattering nodes randomly on a grid to assign longitudinal and
    lattiudanal coordinates.
    :param nodes:       Amount of nodes to scatter - int
    :param scale_lon:   Scale to choose longitude value from - tuple (min, max)
    :param scale_lat:   Scale to choose lattitude value from - tuple (min, max)
    :param _mean:       Mean for normal distribution - float
    :param _std:        Standard deviation for normal distribution - float
    :param uniform:     Whether to use uniform distribution instead - Boolean
    :return:            Attribute dictionary - dict
    """
    lon_list = []
    lat_list = []
    attr_dict = {}
    graph = nx.Graph()
    for node in range(1, nodes + 1):
        if uniform:
            if node == 1:
                lon = np.random.uniform(23, 31)
                lat = np.random.uniform(68, 107)
            else:
                lon = np.random.uniform(23, 31)
                lat = np.random.uniform(68, 107)
                a = distances_list_coordinate(lat, lon, graph)
                while len(a[a < 500]) != 0:
                    lon = np.random.uniform(23, 31)
                    lat = np.random.uniform(68, 107)
                    a = distances_list_coordinate(lat, lon, graph)
        elif _mean and _std:
            if node == 1:

                lon = np.random.normal(_mean[0], _std[0])
                lat = np.random.normal(_mean[1], _std[1])
            else:

                lon = np.random.normal(_mean[0], _std[0])
                lat = np.random.normal(_mean[1], _std[1])
                a = distances_list_coordinate(lat, lon, graph)
                while len(a[a < 500]) != 0:
                    lon = np.random.normal(_mean[0], _std[0])
                    lat = np.random.normal(_mean[1], _std[1])
                    a = distances_list_coordinate(lat, lon, graph)

        else:
            lon = np.random.normal((scale_lon[1] - scale_lon[0]) / 2, (scale_lon[
                                                                           1] -
                                                                       scale_lon[
                                                                           0]) / 4)
            lat = np.random.normal((scale_lat[1] - scale_lat[0]) / 2, (scale_lat[
                                                                           1] -
                                                                       scale_lat[
                                                                           0]) / 4)
        # print(nt.Topology.distances_list_coordinate(lat,lon, graph))
        # _mean[0] = lon
        # _mean[1] = lat
        attr_dict[node] = {"Latitude": lat, "Longitude": lon}
        graph.add_node(node)
        nx.set_node_attributes(graph, attr_dict)
        lat_list.append(lat)
        lon_list.append(lon)
    return graph


def relative_neighbourhood_threshold(func):
    """
    Decorator to throw on top of a N_D function to choose destination nodes enforcing relative
    neighbourhood threshold.
    :param func:    Normally N_D function to decorate - function(N_S, N, graph, m, choice_prob) see N_D above
    :return:        Function
    """

    def choose_nodes(*args, **kwargs):
        GT = False
        i = args[0]
        node_choices = args[1]
        graph = args[2]
        edges = args[3]
        choice_prob = args[4]
        while GT == False:
            j = func(i, node_choices, graph, edges, choice_prob)  # destination node
            for _j in j:
                distances_j = distances_list_with_neighbours(_j, graph)
                distances_i = distances_list_with_neighbours(i, graph)
                distance_i_j = distance_between_nodes(i, _j, graph)
                distances_k = np.append(distances_i, distances_j)
                distance_max = np.max(distances_k)
                distance_max = distance_max[distance_max >= distance_i_j]
                if len(distance_max) == 0:
                    GT = True
                else:
                    GT = False
                    node_choices.remove(_j)
        return j

    return choose_nodes


def gabriel_threshold(func):
    """
    Decorator to ensure gabriel threshold.
    :param func: Method to choose destination node - same as relative neighbourhood
    :return: func
    """

    def choose_nodes(*args, **kwargs):
        GT = False
        i = args[0]
        node_choices = args[1]
        graph = args[2]
        edges = args[3]
        choice_prob = args[4]
        while GT == False:
            j = func(i, node_choices, graph, edges, choice_prob)  # destination node
            for _j in j:
                distances_j = distances_list_with_neighbours(_j, graph)
                distances_i = distances_list_with_neighbours(i, graph)
                distance_i_j = distance_between_nodes(i, _j, graph)
                distances_k_2 = distances_i ** 2 + distances_j ** 2
                distances_k_2 = distances_k_2[distances_k_2 >= distance_i_j ** 2]
                if len(distances_k_2) == 0:
                    GT = True
                else:
                    GT = False
                    node_choices.remove(_j)
        return j

    return choose_nodes


available_nodes = lambda N_S, N, graph: list(
    filter(lambda x: (N_S, x) not in list(graph.edges), np.arange(1, N + 1)))

N_D_not_available = lambda N_S, N, graph, m, choice_prob: np.random.choice(
    available_nodes(N_S, N, graph), size=m, replace=False, p=choice_prob(N_S,
                                                                         graph))  # randomly choose destination node given probabilities

nodes_distances_list = lambda N_S, graph: np.delete(
    np.asarray([node for node in graph.nodes()]),
    np.append(np.asarray(list(graph.neighbors(N_S))), N_S) - 1)

distances_list_coordinate = lambda lat, lon, graph: np.asarray([
    calculate_harvesine_distance(
        lat,
        graph.nodes.data()[
            node]["Latitude"],
        lon,
        graph.nodes.data()[
            node]["Latitude"])
    for node in
    graph.nodes()])
distances_sum = lambda N_S, graph: np.sum(
    distances_list(N_S,
                   graph))  # get a sum of all distances given source node
distances_prob = lambda N_S, graph: distances_list(N_S, graph) / distances_sum(
    N_S,
    graph)  # get probabilities of all destination nodes to be picked


def get_closest_nodes(graph, N_S, closet_num=5):
    """
    Method to return the closest nodes of a graph given a source node.
    :param graph:       Graph to use for calculations - nx.Graph()
    :param N_S:         Source node to use - int
    :param closest_num: Amount of nodes to return - int
    :return:            List of nodes closest to N_S - list
    """
    distances = distances_list(N_S, graph)
    nodes_list = nodes_distances_list(N_S, graph)
    sorted_dist = np.argsort(distances)
    closest = nodes_list[sorted_dist]
    closest = closest[:closet_num]
    return closest


def assign_distances_grid(graph, pythagorus=False, harvesine=False, lon="Longitude",
                          lat="Latitude"):
    """
    This method assigns the distances of a graph using a grid system.
    :param graph:       Graph to assign distances - nx.Graph()
    :param pythagorus:  If set to true, it calculates distances purely from pythagorus - boolean
    :param harvesine:   If set to true, it calculates distances accordingly using harvesine formula - boolean
    :param lon:         Node attribute name for longitude - default "y" - only applies to harvesine calculation - String
    :param lat:         Node attribute name for lattitude - default "x" - only applies to harvesine calculation - String
    :return:            Graph with distances assigned - nx.Graph()
    """
    node_data = graph.nodes.data()
    if pythagorus:
        for edge in graph.edges:
            x_1 = node_data[edge[0]][lat]
            x_2 = node_data[edge[1]][lat]
            y_1 = node_data[edge[0]][lon]
            y_2 = node_data[edge[1]][lon]
            x_diff = np.abs(x_1 - x_2)
            y_diff = np.abs(y_1 - y_2)
            dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
            graph[edge[0]][edge[1]]["weight"] = dist
    elif harvesine:
        if type(graph) == nx.classes.multigraph.MultiGraph:
            # case for when the graph is a multigraph
            for edge in graph.edges(keys=True):
                try:
                    lat_1 = node_data[edge[0]][lat]
                    lat_2 = node_data[edge[1]][lat]
                    lon_1 = node_data[edge[0]][lon]
                    lon_2 = node_data[edge[1]][lon]
                    d = calculate_harvesine_distance(lat_1, lat_2, lon_1,
                                                     lon_2)
                    graph[edge[0]][edge[1]][edge[2]]["weight"] = math.ceil(d / 80)
                except:
                    print("missing coordinates")
                    continue
        else:
            for edge in graph.edges:
                try:
                    lat_1 = node_data[edge[0]][lat]
                    lat_2 = node_data[edge[1]][lat]
                    lon_1 = node_data[edge[0]][lon]
                    lon_2 = node_data[edge[1]][lon]
                    d = calculate_harvesine_distance(lat_1, lat_2, lon_1, lon_2)
                    graph[edge[0]][edge[1]]["weight"] = math.ceil(d / 80)
                except:
                    print("missing coordinates")
                    continue

    return graph


def calculate_harvesine_distance(lat_1, lat_2, lon_1, lon_2):
    """
    Method to calculate the harvesine distance between two aets of geodetic coordinates.
    :param lat_1:   First latitude - float
    :param lat_2:   Second latitude - float
    :param lon_1:   First longitude - float
    :param lon_2:   Second longitude - float
    :return:        Distance - float
    """
    lat_diff = np.abs(lat_1 - lat_2)
    lon_diff = np.abs(lon_1 - lon_2)
    a = np.sin(lat_diff * np.pi / 180 / 2) * np.sin(
        lat_diff * np.pi / 180 / 2) + np.cos(lat_1 * np.pi / 180) * np.cos(
        lat_2 * np.pi / 180) * np.sin(lon_diff * np.pi / 180 / 2) * np.sin(
        lon_diff * np.pi / 180 / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = 6371 * c
    if d < 1000:
        d *= 1.5
    elif d <= 1200 and d >= 1000:
        d = 1500
    elif d > 1200:
        d *= 1.25
    return d


def choose_source_node_sequential(graph1, graph2):
    """
    Method to choose a next node from graph1 given that graph2 have already been added.
    :param graph1:  graph of which to evaluate the next node - nx.Graph()
    :param graph2:  graph where already picked nodes are in - nx.Graph()
    :return:        next node in sequence picked from graph1 evaluated on distances to graph2 nodes - int
    """
    node_list1 = list(graph1.nodes())
    node_list2 = list(graph2.nodes())

    for node in node_list1.copy():
        if node in node_list2:
            node_list1.remove(node)

    node_distance_mean = []
    for node in node_list1:
        distances = distances_list_coordinate(graph1.nodes.data()[node]["Latitude"],
                                              graph1.nodes.data()[node]["Longitude"],
                                              graph2)
        node_distance_mean.append(np.mean(distances))

    sorted_node_distance_mean = np.argsort(node_distance_mean)
    return node_list1[sorted_node_distance_mean[0]]


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

WEIGHTS_N = [1520, 1440, 1040, 3520, 2160, 1040, 1040, 1120, 880, 1840, 2960, 880, 1200,
             560, 560, 640, 400,
             1280, 2480, 1520, 2640]


class Topology():
    def __init__(self):
        self.topology_graph = nx.Graph()
        self.init_nsf()
        self.degree_list = lambda graph: np.asarray(list(
            map(lambda x: x[1],
                list(graph.degree))))  # get list of degrees for all nodes

    def init_btcore(self):
        s_UK = [1, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 11, 11, 12,
                12, 13, 14, 15, 15, 15, 16, 17, 18, 18, 19, 20, 21, 22]
        t_UK = [2, 4, 3, 4, 6, 5, 6, 10, 7, 22, 8, 21, 9, 19, 10, 13, 11, 12, 14, 16,
                13, 14, 14, 15, 16, 17, 19, 17, 18, 19, 20, 20, 21, 22, 1]
        weights_UK = [240, 120, 48, 686, 419, 163, 87, 275, 99, 109, 73, 75, 163, 89,
                      105, 127, 183, 7, 2,
                      115, 23, 5, 20, 59, 55, 197, 203, 226, 71, 234, 439, 27, 160, 62,
                      182]
        nodes = np.arange(1, len(weights_UK))
        topology_graph = nx.Graph()
        topology_graph.add_nodes_from(nodes)
        topology_graph.add_weighted_edges_from(self.make_weighted_edge_list(s_UK, t_UK,
                                                                            weights_UK))

    def init_google_b4(self):
        s_Go = [1, 1, 2, 2, 2, 3, 3, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11]
        t_Go = [2, 3, 3, 4, 6, 4, 5, 6, 7, 9, 7, 9, 8, 10, 9, 11, 10, 12, 12]
        weights_Go = [2320, 2560, 320, 1120, 8560, 1200, 8320, 480, 720, 1680, 480,
                      2080, 600, 4560,
                      880, 15440, 2640, 16480, 3840]
        nodes = np.arange(1, len(weights_Go))
        topology_graph = nx.Graph()
        topology_graph.add_nodes_from(nodes)
        topology_graph.add_weighted_edges_from(self.make_weighted_edge_list(s_Go, t_Go,
                                                                            weights_Go))

    def init_uk_net(self):
        UK_NET = [[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]
        topology_graph = nx.Graph()
        topology_graph.add_nodes_from(range(1, len(UK_NET)))
        for i in range(len(UK_NET)):
            for j in range(len(UK_NET)):
                if i == j:
                    pass
                elif j < i:
                    pass
                if UK_NET[i][j] == 1:
                    topology_graph.add_edge(i, j)
                    topology_graph[i][j]["weight"] = 1
                else:
                    pass

    def init_dtag(self):
        s_D = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 7, 8, 9, 9, 10, 11, 11, 12,
               13]
        t_D = [2, 10, 14, 3, 10, 4, 10, 5, 9, 10, 6, 8, 9, 7, 8, 9, 10, 12, 11,
               12, 14, 13, 14]
        weights_D = [160, 160, 320, 400, 160, 240, 400, 320, 480, 320, 240, 240, 320,
                     160, 80, 240, 400,
                     240, 240, 80, 80, 80, 80]
        names_D = [
            'BR', 'HG', 'BN', 'LG', 'NG', 'MN', 'UM', 'ST', 'FT', 'HR', 'DD', 'KN',
            'DF', 'EN']

        nodes = np.arange(1, len(names_D))
        topology_graph = nx.Graph()
        topology_graph.add_nodes_from(nodes)
        topology_graph.add_weighted_edges_from(self.make_weighted_edge_list(s_D, t_D,
                                                                            weights_D))

    def init_nsf(self):
        s_N = [1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 13]
        t_N = [2, 6, 4, 3, 14, 4, 10, 5, 6, 13, 7, 8, 9, 10, 11, 10, 11, 12, 13, 13, 14]
        weights_N = [1520, 1440, 1040, 3520, 2160, 1040, 1040, 1120, 880, 1840, 2960,
                     880, 1200, 560, 560, 640, 400,
                     1280, 2480, 1520, 2640]
        self.weights = weights_N
        names_N = ['C1', 'WA', 'IL', 'NE', 'CO', 'UT', 'MI', 'NY', 'NJ', 'PA', 'DC',
                   'GA', 'TX', 'C2']
        num_nodes = 14
        num_edges = len(s_N)
        connectivity = 0
        efficiency = 0
        nodes = np.arange(1, num_nodes)
        self.topology_graph.add_nodes_from(nodes)
        self.topology_graph.add_weighted_edges_from(
            self.make_weighted_edge_list(s_N, t_N, weights_N))
        self.topology_graph = self.assign_congestion(self.topology_graph)

    def init_three_node(self):
        self.topology_graph = nx.Graph()
        three_node = [[0, 1, 1, 0, 0, 1],
                      [1, 0, 0, 0, 1, 0],
                      [1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 0, 1],
                      [1, 0, 0, 1, 1, 0]]
        self.topology_graph.add_nodes_from(range(1, 6))
        for i in range(6):
            for j in range(6):
                if i == j:
                    pass
                elif j < i:
                    pass
                if three_node[i][j] == 1:
                    self.topology_graph.add_edge(i, j)
                    self.topology_graph[i][j]["weight"] = 1
                else:
                    pass

    def init_EURO_core(self):
        euro = [[0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]]
        self.topology_graph.add_nodes_from(range(1, 12))
        for i in range(11):
            for j in range(11):
                if i == j:
                    pass
                elif j < i:
                    pass
                if euro[i][j] == 1:
                    self.topology_graph.add_edge(i, j)
                    self.topology_graph[i][j]["weight"] = 1
                else:
                    pass
        self.topology_graph = self.assign_congestion(self.topology_graph)
        # print(self.topology_graph.edges)

    def plot_topology(self):
        plt.plot()
        nx.draw(self.topology_graph, with_labels=True)
        plt.show()

    def plot_graph(self, graph):
        plt.plot()
        nx.draw(graph, with_labels=True)
        plt.show()

    def read_topology_vector_dataset(self):
        """This method automatically reads the current saved topology vectors

        :return: topology_vector_dataset
        :rtype: dict"""
        topology_vector_df = pd.read_json(path_or_buf="Data/_topology_vectors.json",
                                          lines=True)

        topology_vector = topology_vector_df["topology_vector"].to_list()

        return topology_vector

    def write_topology_vector_dataset(self):
        """
        This method creates a pandas dataframe from the global topology_vector_dataset and saves as .csv

        :return: None
        :rtype: None
        """

        self.topology_vector_dataset_df = pd.DataFrame(self.topology_vector_dataset)
        self.topology_vector_dataset_df.to_json(
            path_or_buf="Data/_topology_vectors.json")

    def save_topology(self):
        nx.write_adjlist(self.topology_graph,
                         path="Topology/{}".format(self.type + ".adjlist"))

    def load_topology(self):
        nx.read_adjlist("Topology/{}".format(self.type + ".adjlist"))

    def save_graph_json(self, graph, name):
        """

        :param graph:
        :param name:
        :return:
        """
        import json
        data1 = nx.json_graph.node_link_data(graph, {'link': 'edges', 'source': 'from',
                                                     'target': 'to'})

        with open("{}.json".format(name), "w") as write_file:
            s2 = json.dump(data1, write_file,
                           default={'link': 'edges', 'source': 'from', 'target': 'to'})

    def save_graph_gexf(self, graph, name):
        """

        :param graph:
        :param name:
        :return:
        """
        nx.write_gexf(graph, "{}.gexf".format(name))

    def save_graph(self, graph, name):
        """
        Method that saves graphs as a weighted edge list.

        :param graph: Graph to be saved
        :param name: Name to be saved under in Topolgy Directory
        :return: None
        :rtype: None
        """

        nx.write_weighted_edgelist(graph, path="Topology/{}".format(
            name + ".weighted.edgelist"))

    def load_graph(self, name):
        """
        Method to load graphs from weighted edge lists stored in Topology

        :param name: Name of graph to be loaded
        :return: returns graph with congestion and NSR assignments
        :rtype: nx.Graph()
        """
        graph = nx.read_weighted_edgelist(
            path="/home/uceeatz/Code/Optical-Networks/Topology/{}".format(
                name + ".weighted.edgelist"),
            create_using=nx.Graph(), nodetype=int)
        graph = self.assign_congestion(graph)
        graph = self.assign_NSR(graph)
        return graph

    def make_edge_list(self, s, t):
        edge_list = []
        for i in range(0, len(s)):
            edge_list.append((s[i], t[i]))
        return edge_list

    def make_weighted_edge_list(self, s, t, w):
        edge_list = []
        for i in range(0, len(s)):
            edge_list.append((s[i], t[i], w[i] / 80))

        return edge_list

    def save_graph_database(self, graph, db_name, collection_name, k=None,
                            scaling_factor=None, node_data=None, name=None):
        # index = len(list(Database.read_data("Topology_Data", "topology_data",{})))
        # print(index)
        alpha_graph = lambda N_ACMN, L_ACMN: (2 * L_ACMN) / (
                N_ACMN * (N_ACMN - 1))  # lambda function for connectivity
        graph_data = nx.to_dict_of_dicts(graph)
        # print(graph_data)
        graph_data = {
            str(y): {str(z): str(graph_data[y][z]) for z in graph_data[y].keys()} for y
            in graph_data.keys()}
        topology_dict = Database.topology_data_template.copy()
        topology_dict["topology data"] = graph_data
        topology_dict["connectivity"] = alpha_ACMN(len(list(graph.nodes)),
                                                   len(list(graph.edges)))
        topology_dict["nodes"] = len(list(graph.nodes()))
        topology_dict["edges"] = len(list(graph.edges()))
        topology_dict["topology vector"] = self.create_binary_topology_vector(
            graph).tolist()
        topology_dict["mean k"] = k
        if scaling_factor != None:
            topology_dict["scaling factor"] = scaling_factor
        if node_data == True:
            node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()))
            topology_dict["node data"] = node_data
        # if name != None:
        #    topology_dict["name"] = name
        # print(topology_dict)
        # print(type(topology_dict))
        Database.insert_data(db_name, collection_name, topology_dict)

    def create_random_ACMN_dataset(self, N, amount, database_name, collection_name,
                                   use_dask=True, BA=False, WS=False, ER=False,
                                   random=False, pref_grid=False, alpha=None, k=None,
                                   p=None, grid_graph=None, L=None, bar=None):
        """This method is used to create ACMN datasets, which creates the specified amount of topologies randomly, with the specified amount of distance realisations.


        :param N: Number of nodes of ACMN
        :param L: Number of links of ACMN
        :param alpha: Connectivity of graph
        :param amount: Amount of ACMN networks to create
        :param distance_realisations: Amount of network distance realisations to create
        :return: None, saves all topologies to disk in Optical-Networks ->Topology
        :rtype: None"""

        # self.topology_vector_dataset = {"topology_vector": []}
        # self.write_topology_vector_dataset()
        if use_dask:
            tasks = []
            for i in range(0, amount):
                self.progress = (i / amount) * 100
                a = dask.delayed(self.create_save_database_ACMN)
                tasks.append(
                    a(N, collection_name, database_name, L=L, grid_graph=grid_graph,
                      BA=BA, WS=WS, ER=ER,
                      random=random, pref_grid=pref_grid, alpha=alpha, k=k, p=p))


        else:
            tasks = []
            for i in range(amount):
                self.create_save_database_ACMN(N, collection_name, database_name, L=L,
                                               grid_graph=grid_graph, BA=BA, WS=WS,
                                               ER=ER, random=random,
                                               pref_grid=pref_grid, alpha=alpha, k=k,
                                               p=p)
                if bar:
                    bar.next()

            """if not save_database:
                self.save_graph(ACMN, name + "{}_{}".format(alpha, i))
            if distance_realisations_bool:
                for j in range(1, distance_realisations):
                    ACMN = self.assign_distances(ACMN, scaling_factor=(640 / 1463) + j * 0.0164)
                    if save_database:
                        self.save_graph_database(ACMN, alpha)
                    else:
                        self.save_graph(ACMN, "ACMN{}_{}_{}".format(alpha, i, j))
                logging.debug("Progress: {}%".format(int((i / amount) * 100)))

           #print("Progress: {}%".format(int((i / amount) * 100)), end='\r', flush=True)"""

        return tasks

    def find_degree_vector_ij(self, degree_vector_2D, degree_vector_3D_1,
                              degree_vector_max1, diameter, graph):
        degree_vector_ij = []

        if len(degree_vector_2D) > 1:
            lengths = list(list(
                nx.shortest_path_length(graph, node[0], degree_vector_2D[i][0]) for i in
                # create a list of lists with non-repeated lengths between degree node pairs
                range(0, len(degree_vector_2D))) for node in degree_vector_2D)

            ij1 = list(
                lengths[i][i + 1:] for i in range(0, len(
                    lengths)))  # get the unique values (right from zero diagonl)
            ij1 = list(map(lambda x: x.count(diameter),
                           ij1))  # count instances of paths with hops of length diameter
            ij1 = reduce(lambda x, y: x + y, ij1)  # sum the instances together

            degree_vector_ij.append(
                ij1)  # append them to the ij part of the topology vector (repeat for the others with D-1 and 1 in respect)

        else:
            degree_vector_ij.append(0)
        if len(degree_vector_3D_1) > 1:
            lengths = list(list(
                nx.shortest_path_length(graph, node[0], degree_vector_3D_1[i][0]) for i
                in
                range(0, len(degree_vector_3D_1))) for node in degree_vector_3D_1)

            ij2 = list(lengths[i][i + 1:] for i in range(0, len(lengths)))

            ij2 = list(map(lambda x: x.count(diameter - 1), ij2))
            ij2 = reduce(lambda x, y: x + y, ij2)

            degree_vector_ij.append(ij2)
        else:
            degree_vector_ij.append(0)
        if len(degree_vector_max1) > 1:
            lengths = list(list(
                nx.shortest_path_length(graph, node[0], degree_vector_max1[i][0]) for i
                in
                range(0, len(degree_vector_max1))) for node in degree_vector_max1)

            ij3 = list(lengths[i][i + 1:] for i in range(0, len(lengths)))

            ij1 = list(map(lambda x: x.count(1), ij3))
            ij3 = reduce(lambda x, y: x + y, ij3)

            degree_vector_ij.append(ij3)

        else:
            degree_vector_ij.append(0)
        return degree_vector_ij

    #
    # def create_topology_vector(self, graph):
    #     """
    #     This method creates the topology vector for a given input graph
    #
    #     :param graph: the graph from which to derive the topology vector
    #     :return: topology_vector
    #     :rtype: List
    #     """
    #     d = lambda x: list(x.count(i) for i in range(2, max(
    #         x) + 1))  # count number of nodes with same degree
    #
    #     diameter = nx.diameter(graph)
    #     degrees = graph.degree()  # Get the degree of all nodes [(node, degree)...]
    #     degree_vector = list(
    #         map(lambda x: x[1], degrees))  # Extract only the degree of the node
    #     degree_vector_2D = list(
    #         filter(lambda x: x[1] is 2, degrees))  # sort degrees into degree of 2
    #     degree_vector_3D_1 = list(
    #         filter(lambda x: x[1] is 2, degrees))  # sort degrees into degree of 3
    #     degree_vector_max1 = list(filter(lambda x: x[1] is max(degrees),
    #                                      degrees))  # sort degrees into degree of max
    #     degree_vector_ij = self.find_degree_vector_ij(degree_vector_2D,
    #                                                   degree_vector_3D_1,
    #                                                   degree_vector_max1,
    #                                                   diameter,
    #                                                   graph)  # Go through node pairs of degree two, three and max and see if the are within D, D-1 and max hops
    #
    #     topology_vector = d(degree_vector) + degree_vector_ij + [diameter] + [
    #         round(nx.average_shortest_path_length(graph),
    #               2)]  # Concatenate the topology vector
    #
    #     return topology_vector

    def create_save_database_ACMN(self, N, collection_name, database_name,
                                  grid_graph=None, L=None, BA=False, ER=False, WS=False,
                                  random=False, pref_grid=False, alpha=None, k=None,
                                  p=None, node_data=False):
        """
        Method to create one of the random graph structures and save it to a database connected to.
        :param N: Number of nodes
        :param collection_name: collection name to save the graph under
        :param database_name: database name under which to save the graph
        :param BA: Set to True for Barabase-Alvert Graph
        :param ER: Set to True for Erdoes-Renyi Graph
        :param WS: Set to True for Watts-Strogatz
        :param random: Set to True for random uniform
        :param alpha: connectivity if creating random uniform graph
        :param k: mean degree k for WS, BA
        :param p: parameter p for ER, WS
        :return: None
        """
        try:
            if random:
                ACMN = self.create_random_ACMN(N, alpha,
                                               collection_name=collection_name,
                                               database_name=database_name)
            elif WS:
                ACMN = self.create_WS_ACMN(N, k, p, collection_name=collection_name,
                                           database_name=database_name)
            elif ER:
                ACMN = self.create_ER_ACMN(N, p, collection_name=collection_name,
                                           database_name=database_name)
            elif BA:

                ACMN = self.create_BA_ACMN(N, k, collection_name=collection_name,
                                           database_name=database_name)
            elif pref_grid:
                ACMN = self.create_real_based_grid_graph(grid_graph, L,
                                                         database_name=database_name,
                                                         collection_name=collection_name)
                node_data = True

            if alpha == None:
                alpha = len(list(ACMN.edges())) / (N * (N - 1) / 2)
            if k != None:
                self.save_graph_database(ACMN, database_name, collection_name, k)
            else:
                self.save_graph_database(ACMN, database_name, collection_name,
                                         node_data=node_data)
        except Exception as err:
            print("Save error occured: {}".format(err))

    def create_BA_ACMN(self, N, k, collection_name=None, database_name=None):
        """
        Method to create a Barabási and Albert based random graph, which works on the principle of "rich get richer"
        :param N: The number of nodes in the graph
        :param k: mean k to create the graphs with (avg_degree)
        :param collection_name: collection name under which to make the check for uniqueness
        :param database_name: Name of database to save in.
        :return: graph (nx graph)
        """
        E = k / 2

        ACMN = nx.barabasi_albert_graph(N, int(E))
        if self.check_bridges(ACMN) == True and self.check_min_degree(
                ACMN) == True and nx.is_connected(
            ACMN) == True:  # check constrains C1 and C2
            topology_vector = self.create_binary_topology_vector(ACMN)
        else:
            alpha = len(list(ACMN.edges())) / (N * (N - 1))
            ACMN = self.create_BA_ACMN(N, k, collection_name=collection_name,
                                       database_name=database_name)  # if C1 or C2 fails try again
        topology_vector = self.create_binary_topology_vector(ACMN)
        alpha = len(list(ACMN.edges())) / (N * (N - 1))
        if not self.check_unique_binary_topology_vector(topology_vector, k, N,
                                                        collection_name, database_name):
            ACMN = self.create_BA_ACMN(N, k, collection_name=collection_name,
                                       database_name=database_name)  # if C1 or C2 fails try again

        ACMN = self.assign_distances(ACMN)
        ACMN = self.assign_congestion(ACMN)
        ACMN = self.assign_NSR(ACMN)
        return ACMN

    def create_ER_ACMN(self, N, p, collection_name=None, database_name=None):
        """
        Method to create Erdoes-Renyi graphs with checks for minimal topological charachteristiscs for OPtical Networks.
        :param N: The number of nodes in the graph
        :param p: the probability of having a link added (N*p = avg_degree)
        :param collection_name: collection name under which to make the check for uniqueness
        :param database_name: Name of database to save in.
        :return: graph (nx graph)
        """
        ACMN = nx.erdos_renyi_graph(N, p)
        alpha = len(list(ACMN.edges())) / (N * (N - 1) / 2)

        if self.check_bridges(ACMN) == True and self.check_min_degree(
                ACMN) == True and nx.is_connected(
            ACMN) == True:  # check constrains C1 and C2
            topology_vector = self.create_binary_topology_vector(ACMN)
        else:
            ACMN = self.create_ER_ACMN(N, p, collection_name=collection_name,
                                       database_name=database_name)  # if C1 or C2 fails try again
        topology_vector = self.create_binary_topology_vector(ACMN)
        if not self.check_unique_binary_topology_vector(topology_vector, int(N * p), N,
                                                        collection_name, database_name):
            ACMN = self.create_ER_ACMN(N, p, collection_name=collection_name,
                                       database_name=database_name)  # if C1 or C2 fails try again

        ACMN = self.assign_distances(ACMN)
        ACMN = self.assign_congestion(ACMN)
        ACMN = self.assign_NSR(ACMN)
        return ACMN

    def create_WS_ACMN(self, N, k, p, collection_name=None, database_name=None):
        """
        Method to create Watts-Strogatz graph that returns a Watts strogatz graph, with minimal topological parameters for optical networks.
        :param N: Number of nodes in the graph
        :param k: mean k to create the graphs with (avg_degree)
        :param p: possibility of making a connection
        :param collection_name: collection name under which to make the check for uniqueness
        :param database_name: database name under which to make the check for uniqueness
        :return: graph (nx graph)
        """
        ACMN = nx.connected_watts_strogatz_graph(N, k, p)
        alpha = len(list(ACMN.edges())) / (N * (N - 1) / 2)
        if self.check_bridges(ACMN) == True and self.check_min_degree(
                ACMN) == True and nx.is_connected(
            ACMN) == True:  # check constrains C1 and C2
            topology_vector = self.create_binary_topology_vector(ACMN)
        else:
            ACMN = self.create_WS_ACMN(N, k, p, collection_name=collection_name,
                                       database_name=database_name)  # if C1 or C2 fails try again
        topology_vector = self.create_binary_topology_vector(ACMN)
        if not self.check_unique_binary_topology_vector(topology_vector, k, N,
                                                        collection_name, database_name):
            ACMN = self.create_WS_ACMN(N, k, p, collection_name=collection_name,
                                       database_name=database_name)  # if C1 or C2 fails try again

        ACMN = self.assign_distances(ACMN)
        ACMN = self.assign_congestion(ACMN)
        ACMN = self.assign_NSR(ACMN)

        return ACMN

    def create_random_ACMN(self, N, alpha, collection_name=None, database_name=None):
        """
        This method creates a randomly generated graph that is subject to constraints C1, C2 and unique topology within the
        _topology_vectors.csv file. It return a graph and topology vector for said graph.

        :param N: Number of Nodes
        :param L: Number of Links
        :param alpha: Connectivity of graph
        :return: graph, topology_vector
        :rtype: nx.Graph()
        """
        # print("creating ACMN")
        degree_max = math.ceil((N - 1) * alpha + 2 * 2)
        nodes = np.arange(1, N + 1)  # list with nodes in it
        ACMN = nx.Graph()  # creating graph
        ACMN.add_nodes_from(nodes)  # adding nodes
        alpha_ACMN = lambda N_ACMN, L_ACMN: (2 * L_ACMN) / (
                N_ACMN * (N_ACMN - 1))  # lambda function for connectivity
        link_ACMN = lambda N: (int(np.random.uniform(1, N)), int(
            np.random.uniform(1,
                              N)))  # lambda function for choosing random link from uniform distribution
        while alpha_ACMN(N, ACMN.number_of_edges()) < alpha:
            link = link_ACMN(N + 1)
            degree_1 = (ACMN.degree[link[0]] + 1)
            degree_2 = (ACMN.degree[link[1]] + 1)
            if link[0] != link[1] and degree_1 <= degree_max and degree_2 <= degree_max:
                ACMN.add_edge(link[0], link[1])  # add new link
        topology_vector = self.create_binary_topology_vector(ACMN)
        if self.check_bridges(ACMN) == True and self.check_min_degree(
                ACMN) == True and nx.is_connected(
            ACMN) == True:  # check constrains C1 and C2
            topology_vector = self.create_binary_topology_vector(ACMN)

        else:
            ACMN = self.create_random_ACMN(N, alpha, collection_name=collection_name,
                                           database_name=database_name)  # if C1 or C2 fails try again

        if not self.check_unique_binary_topology_vector(topology_vector, alpha, N,
                                                        collection_name, database_name):
            ACMN = self.create_random_ACMN(N, alpha, collection_name=collection_name,
                                           database_name=database_name)
        ACMN = self.assign_distances(ACMN)
        ACMN = self.assign_congestion(ACMN)
        ACMN = self.assign_NSR(ACMN)

        return ACMN

    def create_random_ACMN_grid(self, N, L, pos, collection_name=None,
                                database_name=None, degree_max=5, scaling_factor=1.0,
                                pythagorus=False, harvesine=False, weighted=False):
        """
        Method to create random ACMN with a pre-determined grid structure. DEPRECATED Use create_real_based_grid_graph()
        :param N: Number of node
        :param L: Number of links
        :param pos: list of tuples with x and y pos
        :param collection_name: database collection name
        :param database_name: database name
        :param degree_max: max degree allowed
        :return: ACMN - nx.Graph()
        """

        nodes = np.arange(1, N + 1)
        # print(len(pos))
        nodes_attr = {}
        for ind, i in enumerate(nodes):
            nodes_attr[i] = dict(x=pos[ind][0] * scaling_factor,
                                 y=pos[ind][1] * scaling_factor)
        # print(nodes)
        # print(nodes_attr)
        ACMN = nx.Graph()

        ACMN.add_nodes_from(nodes)
        nx.set_node_attributes(ACMN, nodes_attr)
        # print(dict(ACMN.nodes.data()))
        alpha_ACMN = lambda N_ACMN, L_ACMN: (2 * L_ACMN) / (
                N_ACMN * (N_ACMN - 1))  # lambda function for connectivity
        link_ACMN = lambda N: (int(np.random.uniform(1, N)), int(
            np.random.uniform(1,
                              N)))  # lambda function for choosing random link from uniform distribution
        if weighted:
            # lambda functions to calculate probabilities and to pick source destination nodes for link placement

            degree_sum = lambda ACMN: np.sum(
                degree_list(ACMN))  # get sum of degrees for all nodes
            degree_prob = lambda ACMN: degree_list(
                ACMN) / degree_sum(
                ACMN)  # get probabilities of all nodes to be chosen to have a link added
            N_S = lambda N, ACMN: (int(np.random.choice(N + 1, 1, degree_prob(
                ACMN))))  # randomly choose source node given probabilites (BA probabilities for source node)
            distances_list = lambda N_S, ACMN: np.asarray(
                [self.calculate_harvesine_distance(ACMN.nodes.data()[N_S]["Latitude"],
                                                   # get a list of distances given the source node
                                                   ACMN.nodes.data()[node]["Latitude"],
                                                   ACMN.nodes.data()[N_S]["Longitude"],
                                                   ACMN.nodes.data()[node]["Longitude"])
                 for node in ACMN.nodes()])
            distances_sum = lambda N_S, ACMN: np.sum(
                distances_list(N_S,
                               ACMN))  # get a sum of all distances given source node
            distances_prob = lambda N_S, ACMN: degree_list(N_S, ACMN) / distances_sum(
                N_S,
                ACMN)  # get probabilities of all destination nodes to be picked
            N_D = lambda N_S, N, ACMN: (
                int(np.random.choice(N + 1, 1, distances_prob(N_S,
                                                              ACMN))))  # randomly choose destination node given probabilities

            ACMN = self.add_regular_degree(ACMN)
            import NetworkToolkit as nt
            nt.Plotting.plot_graph(ACMN)
            plt.savefig("debuggraph.png")
            exit()
            # print(degree_list(ACMN))
            # print(degree_sum(ACMN))
            # print(degree_prob(ACMN))
            # print([ACMN.nodes.data()[node]["Latitude"] for node in ACMN.nodes])
            # print(self.calculate_harvesine_distance(51.2, 51.2, 8.79, 8.79))
            # print(distances_list(1, ACMN))
            # (weighted distance probabilities for destination node)
        links = 0
        while links < L:
            if weighted:
                source = N_S(N, ACMN)  # get source node
                destination = N_D(source, N, ACMN)  # get destination node
                link = (source, destination)  # create link
                degree_1 = (ACMN.degree[link[
                    0]] + 1)  # check that the degree on source side is not above max
                degree_2 = (ACMN.degree[link[
                    1]] + 1)  # check that the degree of the destination side not above max
                if self.check_max_degree_for_link(ACMN, link, degree_max):
                    ACMN.add_edge(link[0], link[1])  # add new link
                    links += 1  # increment link counter
            else:
                link = link_ACMN(N + 1)
                if self.check_max_degree_for_link(ACMN, link, degree_max):
                    ACMN.add_edge(link[0], link[1])  # add new link
                    links += 1  # increment link counter

        topology_vector = self.create_binary_topology_vector(ACMN)
        if self.check_bridges(ACMN) == True and self.check_min_degree(
                ACMN) == True and nx.is_connected(ACMN) == True:
            pass
        else:
            ACMN = self.create_random_ACMN_grid(N, L, pos,
                                                collection_name=collection_name,
                                                database_name=database_name,
                                                degree_max=degree_max,
                                                scaling_factor=scaling_factor,
                                                harvesine=harvesine)
        if not self.check_unique_binary_topology_vector(topology_vector,
                                                        alpha_ACMN(N, L), N,
                                                        collection_name, database_name,
                                                        scaling_factor=scaling_factor):
            ACMN = self.create_random_ACMN_grid(N, L, pos,
                                                collection_name=collection_name,
                                                database_name=database_name,
                                                degree_max=degree_max,
                                                scaling_factor=scaling_factor,
                                                harvesine=harvesine)

        ACMN = self.assign_distances_grid(ACMN, pythagorus=pythagorus,
                                          harvesine=harvesine)
        ACMN = self.assign_congestion(ACMN)
        ACMN = self.assign_NSR(ACMN)
        # print(list(ACMN.nodes.data()))
        # print(list(ACMN.edges.data()))
        return ACMN

    def choice_prob(self, alpha=None, beta=None, BA_pure=False, log_distance=False,
                    waxman_graph=False, SBAG=False, normalise=True, BA_plus_dist=False):
        """
        Method to create a destination node probability list given the source node
        and the graph on which to perform this.
        :param N_S:     Source node for which the probability list is meant for.
        :param graph:   Graph for which to create the probability list from (needs x
                        and y coordinate (longitude and lattitude))
        :return:        list of probabilites corresponding to the choice of nodes
        """
        # implement Spatial Barabasi Albert probabilities
        # implement flexible version of choice_prob to cater for different
        # scenarios

        if waxman_graph and normalise:
            choice_prob = lambda N_S, ACMN: alpha * np.exp(
                np.divide(-distances_list(
                    N_S, ACMN), beta * np.max(distances_list(N_S, ACMN))))
            choice_prob = Tools.normalise_func_out(choice_prob)
        elif waxman_graph and not normalise:
            choice_prob = lambda N_S, ACMN: alpha * np.exp(
                np.divide(-distances_list(
                    N_S, ACMN), beta * np.max(distances_list(N_S, ACMN))))

        elif SBAG and normalise:  # spatial preferential attachement with normalised
            # output
            choice_prob = lambda N_S, ACMN: degree_list_NS(N_S, ACMN) / \
                                            (np.power(distances_list(
                                                N_S, ACMN), alpha) * degree_sum(N_S,
                                                                                ACMN))
            choice_prob = Tools.normalise_func_out(choice_prob)
        elif SBAG and not normalise:  # spatial preferential attachement without
            # normalised output
            choice_prob = lambda N_S, ACMN: degree_list_NS(N_S, ACMN) / \
                                            np.multiply(np.power(distances_list(
                                                N_S, ACMN), alpha),
                                                degree_sum(N_S, ACMN))
        elif BA_pure:
            choice_prob = lambda N_S, ACMN: degree_list_NS(N_S, ACMN) / degree_sum(N_S,
                                                                                   ACMN)
        elif BA_plus_dist:
            choice_prob = lambda N_S, ACMN: np.add(distances_list(N_S, ACMN),
                                                   degree_list_NS(N_S, ACMN)) / (
                                                    degree_sum(N_S,
                                                               ACMN) + distances_sum(
                                                N_S, ACMN))
        elif log_distance and normalise:
            choice_prob = lambda N_S, ACMN: np.add(np.log(distances_list(N_S, ACMN)),
                                                   degree_list_NS(N_S, ACMN)) / (
                                                    degree_sum(N_S,
                                                               ACMN) + distances_sum(
                                                N_S, ACMN))
            choice_prob = Tools.normalise_func_out(choice_prob)
        else:
            choice_prob = self.choice_prob(BA_plus_dist=True)

        return choice_prob

    def create_real_based_grid_graph(self, grid_graph, L, choice_prob=None,
                                     database_name=None, collection_name=None,
                                     ARP=False, waxman_BA=False,
                                     sequential_adding=False, random_adding=False,
                                     numeric_adding=False, gabriel_constraint=False,
                                     relative_neighbourhood_constraint=False,
                                     alpha=None, beta=None, BA_pure=False,
                                     log_distance=False, waxman_graph=False,
                                     random=False, SBAG=False, normalise=True,
                                     BA_plus_dist=False, plot_sequential_graphs=False,
                                     return_intermittent_graphs=False,
                                     print_probs=False, overshoot=False,
                                     undershoot=True, first_start_node=False,
                                     random_start_node=False, centre_start_node=False,
                                     remove_unfeasible_edges=False, verbosity=False,
                                     remove_C1_C2_edges=False, max_degree=9, pref_const=2,
                                     ignore_constraints=True, return_sequence=False):
        """
        Method to create graphs that are based on real node locations, however are randomly connected
        in a fashion that mimics real network topologies. Weighting probabilites either by degree or
        distances between nodes. Other features now included, see below:
        :param grid_graph:                          Graph from real topology
        :param L:                                   Number of links to insert into graph
        :param choice_prob:                         Function to calculate the edge adding probabilities
        :param database_name:                       Database to cross-reference the graphs with - to check uniqueness
        :param collection_name:                     Collection to cross-reference the graphs with - to check uniqueness
        :param ARP:                                 Use ARP probabilities for the adding of nodes (alternating preferrential and random attachment)
        :param sequential_adding:                   Adding nodes to the graph that are closest to all nodes in the graph already
        :param random_adding:                       Adding nodes to the graph at random (uniform probabilities)
        :param numeric_adding:                      Adding nodes in numeric order (1-2-3...)
        :param gabriel constraint:                  All edges follow gabriel constraint - NOT VERIFIED
        :param relative_neighbourhood_constraint:   All edges follow the relative neighbourhood_constraint - NOT VERIFIED
        :param alpha:                               Alpha value for waxman, SBAG and ARP probabilities
        :param beta:                                Beta value for the waxman graph probabilities
        :param BA_pure:                             Use pure Barabasi Albert probabilties for the addition of edges
        :param log_distance:                        Use SBAG with log of the distances for the addition of edges
        :param waxman_graph:                        Use Waxman graph probabilities for the addition of edges
        :param random:                              Use random uniform probabilities for the addition of edges
        :param SBAG:                                Use Spatial Barabasi and Albert probabilties for the addition of edges
        :param normalise:                           Normalise the probabilities
        :param BA_plus_dist:                        Use 50-50 degree and distance probabilties - similar to SBAG, recommend to use SBAG
        :param plot_sequential_graphs:              Plot graphs at each stage of algorithm - use with verbosity = True
        :param return_intermittent_graphs:          Whether to return a list of intermitent graphs to analyse the
        graph creations.
        :param print_probs:                         Print the probabilities at each addition of an edge
        :param overshoot:                           Add too many edges then remove
        :param undershoot:                          Add too few edges and then add
        :param first_start_node:                    Starting node taken from numerical sequence (i.e. node 1)
        :param random_start_node:                   Starting node taken at random (uniform probabilties)
        :param centre_start_node:                   Starting node taken as node closest to centroid of graph
        :param remove_unfeasible_edges:             Remove edges that make direct connections without providing significant SNR benefits
        :param verbosity:                           Debug mode when set to True
        :param remove_C1_C2_edges:                  Removes edges whoms removal does not violate C1 and C2
        :param max_degree:                          Maximum degree to allow in the graph
        :param pref_const:                          The constant to divide the ARP count - if 2 then every second edge is chosen randomly, if 3 every third node is chosen randomly
        :param return_sequence:                     Whether to return the sequence in which nodes were added
        :return:                                    graph - nx.Graph()
        """

        if verbosity:
            logger.setLevel(level=logging.DEBUG)
            logger.debug("Debugging Mode On")
        else:
            logger.setLevel(level=logging.WARNING)

        N = len(grid_graph)
        graph = nx.Graph()
        num_nodes = len(list(grid_graph))
        if overshoot:  # undershooting the amount of edges added
            m_n = math.ceil(
                L / num_nodes)  # amount of edges to add per new node (First Phase)
            m_r = L % num_nodes
        else:  # overshooting the amount of edges added
            m_n = math.floor(L / num_nodes)
            m_r = L % num_nodes
        if waxman_graph and normalise:
            choice_prob = lambda N_S, ACMN: np.multiply(alpha * np.exp(
                np.divide(-distances_list(
                    N_S, ACMN), beta * np.max(distances_list(N_S, ACMN)))), _filter(N_S, graph, max_degree))
            choice_prob = Tools.normalise_func_out(choice_prob)
        elif waxman_graph and not normalise:
            choice_prob = lambda N_S, ACMN: alpha * np.exp(
                np.divide(-distances_list(
                    N_S, ACMN), beta * np.max(distances_list(N_S, ACMN))))

        elif SBAG and normalise:  # spatial preferential attachement with normalised
            # output
            choice_prob = lambda N_S, ACMN: np.multiply(degree_list_NS(N_S, ACMN) / \
                                                        (np.power(distances_list(
                                                            N_S, ACMN), alpha) * degree_sum(N_S,
                                                                                            ACMN)),
                                                        _filter(N_S, graph, max_degree))
            choice_prob = Tools.normalise_func_out(choice_prob)

        elif waxman_BA and normalise:
            choice_prob = lambda N_S, ACMN: np.multiply(degree_list_NS(N_S, ACMN) / \
                                                        (np.exp(distances_list(
                                                            N_S, ACMN)) * alpha * degree_sum(N_S,
                                                                                             ACMN)),
                                                        _filter(N_S, graph, max_degree))
            choice_prob = Tools.normalise_func_out(choice_prob)

        elif SBAG and not normalise:  # spatial preferential attachement without
            # normalised output
            choice_prob = lambda N_S, ACMN: degree_list_NS(N_S, ACMN) / \
                                            np.multiply(np.power(distances_list(
                                                N_S, ACMN), alpha),
                                                degree_sum(N_S, ACMN))
        elif BA_pure:
            choice_prob = lambda N_S, ACMN: degree_list_NS(N_S, ACMN) / degree_sum(N_S,
                                                                                   ACMN)
        elif BA_plus_dist:
            choice_prob = lambda N_S, ACMN: np.add(distances_list(N_S, ACMN),
                                                   degree_list_NS(N_S, ACMN)) / (
                                                    degree_sum(N_S,
                                                               ACMN) + distances_sum(
                                                N_S, ACMN))
        elif log_distance and normalise:
            choice_prob = lambda N_S, ACMN: np.add(np.log(distances_list(N_S, ACMN)),
                                                   degree_list_NS(N_S, ACMN)) / (
                                                    degree_sum(N_S,
                                                               ACMN) + distances_sum(
                                                N_S, ACMN))
            choice_prob = Tools.normalise_func_out(choice_prob)
        elif random:
            choice_prob = lambda N_S, ACMN: None
        elif ARP:
            choice_prob = lambda N_S, ACMN: np.multiply(degree_list_NS(N_S, ACMN) / \
                                                        (np.power(distances_list(
                                                            N_S, ACMN), alpha) * degree_sum(N_S,
                                                                                            ACMN)),
                                                        _filter(N_S, graph, max_degree))
            choice_prob = Tools.normalise_func_out(choice_prob)
        else:
            choice_prob = self.choice_prob(BA_plus_dist=True)
        if plot_sequential_graphs:
            import NetworkToolkit as nt
        if ARP:
            counter = 0
        if gabriel_constraint:
            choose_dest = gabriel_threshold(N_D)
            choose_dest_random = gabriel_threshold(N_D_random)
        elif relative_neighbourhood_constraint:
            choose_dest = relative_neighbourhood_threshold(N_D)
            choose_dest_random = relative_neighbourhood_threshold(N_D_random)
        else:
            choose_dest = N_D
            choose_dest_random = N_D_random
        node_order = create_node_order(grid_graph, random_adding=random_adding,
                                       sequential_adding=sequential_adding,
                                       random_start_node=random_start_node,
                                       centre_start_node=centre_start_node,
                                       first_start_node=first_start_node,
                                       numeric_adding=numeric_adding)
        if return_intermittent_graphs:
            intermittent_graphs = []
        for node in node_order:
            # For every node in the real topology lets add a node to the graph
            if len(graph) == 0:  # if the graph has no nodes added yet
                graph.add_node(node)  # add the first node #
                node_attr = [(_node, grid_graph.nodes.data()[_node]) for _node in
                             graph.nodes]
                # getting and transforming
                # the node attributes (
                # x and y locations)
                node_attr = dict(node_attr)
                nx.set_node_attributes(graph,
                                       node_attr)
                # copying the
                # node attributes onto the new
                # graph - probably
                # develop a function for this...
            elif len(graph) == 1:  # if the second node is to be added
                graph.add_node(node)  # add the node
                node_attr = [(_node, grid_graph.nodes.data()[_node]) for _node in
                             graph.nodes]  # getting and transforming the node attributes from the real graph (x and y locations)
                node_attr = dict(node_attr)
                nx.set_node_attributes(graph,
                                       node_attr)  # setting these node attributes
                node_list = list(graph.nodes())
                graph.add_edge(node_list[0], node_list[1])  # adding an edge betweeen
                # the
                # two nodes
            else:  # for all other nodes to be added
                # print("size of graph: {}".format(len(graph)))
                # print("m_n: {}".format(m_n))
                # print("number of edges to be added: {}".format(L))
                graph.add_node(node)  # add the node to the graph
                node_attr = [(_node, grid_graph.nodes.data()[_node]) for _node in
                             graph.nodes]  # copy attributes etc... see above
                node_attr = dict(node_attr)
                nx.set_node_attributes(graph,
                                       node_attr)  # setting the copied node attributes
                s = node  # setting source node to current node
                if plot_sequential_graphs:
                    if print_probs:
                        print("m_n: {}".format(m_n))
                        print(choice_prob(node, graph))
                    nt.Plotting.plot_graph(graph, with_pos=True, x="Latitude",
                                           y="Longitude", node_size=6,
                                           with_labels=False)
                if return_intermittent_graphs:
                    intermittent_graphs.append(copy.deepcopy(graph))

                # print("attributes: {}".format(graph.nodes.data()))
                # print("distances: {}".format(distances_list(node, graph)))
                # print("choice prob: {}".format(choice_prob(node, graph)))
                # print(np.divide(degree_list_NS(node, graph), np.power(
                #    distances_list(node, graph),
                # 1)*degree_sum(node, graph)))

                # print(choice_prob(node,graph))
                if m_n > len(graph) - 1:  # if the amount of edges to add is larger
                    # than the
                    # amount of nodes in the graph currently
                    if ARP:
                        d = []
                        for i in range(len(list(graph.nodes)) - 1):
                            counter += 1
                            if counter % pref_const == 0:
                                destination = choose_dest_random(node, list(graph.nodes), graph, 1, choice_prob,
                                                                 max_degree=max_degree)
                            else:
                                destination = choose_dest(node, list(graph.nodes), graph, 1, choice_prob,
                                                          max_degree=max_degree)
                            graph.add_edge(node, destination)
                    else:
                        for i in range(len(list(graph.nodes)) - 1):
                            graph.add_edge(node, choose_dest(node, list(graph.nodes), graph,
                                                             1, choice_prob, max_degree=max_degree))
                else:
                    if ARP:
                        d = []
                        for m in range(m_n):
                            counter += 1
                            if counter % pref_const == 0:
                                destination = choose_dest_random(node, list(graph.nodes), graph, 1, choice_prob,
                                                                 max_degree=max_degree)
                            else:
                                destination = choose_dest(node, list(graph.nodes), graph, 1, choice_prob,
                                                          max_degree=max_degree)
                            graph.add_edge(node, destination)

                    else:
                        for m in range(m_n):
                            graph.add_edge(node, choose_dest(node, list(graph.nodes),
                                                             graph,
                                                             1, choice_prob, max_degree=max_degree))
                            #
                            #  otherwise choose destination nodes with
                        # number of edges to add for every node m_n

        # if BA_pure:
        #     graph = nx.barabasi_albert_graph(grid_graph, m_n)
        nodes = np.where(degree_list(graph) <= 1)[
            0]  # finding the nodes that have a degree of 1, +1 because nodes in the graph are indexed starting at 1...
        bridge_edge = list(nx.bridges(graph))

        m_r -= len(nodes)  #  finding the remaining amount of edges to add
        nodes = [list(graph.nodes)[node] for node in nodes]
        for ind, node in enumerate(nodes):  # for the nodes of degree 1
            if ind > 0:
                if node not in new_nodes:
                    continue
            # print("node: {}".format(node))
            s = node  # setting the current node to source
            if ARP:
                counter += 1
                if counter % pref_const == 0:
                    d = choose_dest_random(node, list(graph.nodes), graph, 1, choice_prob, max_degree=max_degree)
                else:
                    d = choose_dest(node, list(graph.nodes), graph, 1, choice_prob, max_degree=max_degree)
            else:
                d = choose_dest(node, list(graph.nodes), graph, 1, choice_prob,
                                max_degree=max_degree)  # finding a suitable
                # destination
            graph.add_edge(s, d)  # adding this edge
            new_nodes = np.where(degree_list(graph) == 1)[
                0]
            new_nodes = [list(graph.nodes)[node] for node in new_nodes]  # update
            # nodes list

            if plot_sequential_graphs:
                print("adding edges to degree 1")
                nt.Plotting.plot_graph(graph, with_pos=True, x="Latitude",
                                       y="Longitude",
                                       with_labels=False, node_size=6)
            if return_intermittent_graphs:
                intermittent_graphs.append(copy.deepcopy(graph))
        if len(bridge_edge) > 0:
            logger.debug("minimum edge cut: {}".format(
                list(nx.k_edge_components(
                    graph, k=2))))
            logger.debug("bridge edges: {}".format(bridge_edge))
            logger.debug("edges of graph: {}".format(list(graph.edges)))
            remove_bridge_components(graph, max_degree=max_degree)

            if plot_sequential_graphs:
                logger.debug("removed bridges")
                plt.figure(figsize=(16, 10))
                nt.Plotting.plot_graph(graph, with_pos=True, x="Latitude",
                                       y="Longitude",
                                       with_labels=True, node_size=6)
            if return_intermittent_graphs:
                intermittent_graphs.append(copy.deepcopy(graph))
            # import NetworkToolkit as nt
            # nx.draw(graph, with_labels=True)
            # plt.savefig("test.png")
            # for source, dest in bridge_edge:
            #    d = choose_dest(source, list(graph.nodes), graph, 1, choice_prob)
            #    graph.add_edge(source, d[0])
        #     nt.Plotting.plot_graph(graph, with_pos=True, x="x", y="y") # plotting the graph

        # Three distinct cases can occur now 1. graph has perfect amount of edges - done 2. graph has too many edges - remove some edges without destroying fundamental feature 3. graph has too little edges - add some more edges
        if len(list(graph.edges)) > L:  # if the graph has to many edges
            remove_len = len(
                list(graph.edges)) - L  # calculating amount of edges to remove

            if remove_unfeasible_edges:
                edges_func = find_links_to_remove
            elif remove_C1_C2_edges:
                edges_func = remove_edges_C1_C2
            else:
                edges_func = lambda graph: list(graph.edges)

            try:
                # print("remove len: {}".format(remove_len))
                # print("edges wanted: {}".format(L))
                # print("edges got: {}".format(len(list(graph.edges))))
                logger.debug("{} <= {} <= alpha".format((1 / len(list(
                    graph.nodes))), alpha_graph(
                    len(
                        list(
                            graph.nodes)), L)))
                if plot_sequential_graphs:
                    logger.debug("removing edges")
                    # print("remove len: {}".format(remove_len))
                    # print("edges wanted: {}".format(L))
                    # print("edges got: {}".format(len(list(graph.edges))))
                    logger.debug("remove len: {}".format(remove_len))
                    logger.debug("edges wanted: {}".format(L))
                    logger.debug("edges got: {}".format(len(list(graph.edges))))

                    nt.Plotting.plot_graph(graph, with_pos=True, x="Latitude",
                                           y="Longitude",
                                           with_labels=False, node_size=6)
                if return_intermittent_graphs:
                    intermittent_graphs.append(copy.deepcopy(graph))
                remove_edges_action = remove_edges(graph, remove_len, edges_func)


            except Exception as err:
                # print("Failed to remove edges - {}".format(err))
                print("Failed to remove edges")
                logger.debug("Failed to remove edges - {}".format(err))
                graph = self.create_real_based_grid_graph(grid_graph, L,
                                                          choice_prob=choice_prob,
                                                          database_name=database_name,
                                                          collection_name=collection_name,
                                                          ARP=ARP, waxman_BA=waxman_BA,
                                                          sequential_adding=sequential_adding,
                                                          numeric_adding=numeric_adding,
                                                          gabriel_constraint=gabriel_constraint,
                                                          relative_neighbourhood_constraint=relative_neighbourhood_constraint,
                                                          alpha=alpha, beta=beta,
                                                          BA_pure=BA_pure,
                                                          log_distance=log_distance,
                                                          waxman_graph=waxman_graph,
                                                          random=random, SBAG=SBAG,
                                                          normalise=normalise,
                                                          BA_plus_dist=BA_plus_dist,
                                                          plot_sequential_graphs=plot_sequential_graphs,
                                                          print_probs=print_probs,
                                                          overshoot=overshoot,
                                                          undershoot=undershoot,
                                                          first_start_node=first_start_node,
                                                          random_start_node=random_start_node,
                                                          centre_start_node=centre_start_node,
                                                          remove_unfeasible_edges=remove_unfeasible_edges,
                                                          verbosity=verbosity,
                                                          remove_C1_C2_edges=remove_C1_C2_edges,
                                                          max_degree=max_degree,
                                                          pref_const=pref_const, ignore_constraints=ignore_constraints)


        elif len(list(graph.edges)) < L:  # if the graph has too few edges
            add_len = L - len(list(graph.edges))  # calculate the amount of edges to add

            for _ in range(add_len):
                if len(list(graph.edges)) == (len(graph) * (len(
                        graph) - 1)) / 2:  # if graph is already fully connected break
                    break
                source = choose_random_source(graph, max_degree=max_degree)
                while len(list(graph.neighbors(source))) >= len(
                        graph) - 1:  # if source is already connected to all other nodes choose a new one and delete source
                    source = choose_random_source(graph, max_degree=max_degree)
                if ARP:
                    counter += 1
                    if counter % pref_const == 0:
                        destination = choose_dest_random(source, list(graph.nodes), graph, 1, choice_prob,
                                                         max_degree=max_degree)
                    else:
                        destination = choose_dest(source, list(graph.nodes), graph, 1, choice_prob,
                                                  max_degree=max_degree)
                else:
                    destination = choose_dest(source, list(graph.nodes), graph,
                                              1, choice_prob,
                                              max_degree=max_degree)  #  choosing a suitable destination node
                graph.add_edge(source, destination)  # add this edge to the graph
                if plot_sequential_graphs:
                    logger.debug("adding edges")
                    nt.Plotting.plot_graph(graph, with_pos=True, x="Latitude",
                                           y="Longitude", node_size=6)
                if return_intermittent_graphs:
                    intermittent_graphs.append(copy.deepcopy(graph))

        assert [0, 1] not in degree_list(graph)  #  make sure no degree smaller than 2
        assert len(list(filter(lambda x: x > max_degree, degree_list(graph)))) == 0
        if plot_sequential_graphs:
            logger.debug("graph checks to begin")
            nt.Plotting.plot_graph(graph, with_pos=True, x="Latitude",
                                   y="Longitude",
                                   with_labels=False, node_size=6)
        if return_intermittent_graphs:
            intermittent_graphs.append(copy.deepcopy(graph))
        # nt.Plotting.plot_graph(graph)
        # nt.Plotting.plot_graph(graph, with_pos=True, x="x", y="y")
        if not ignore_constraints:
            if SBAG or ARP:
                # adding alpha parameter to lessen the unique constraint
                graph_is_unique = self.check_graph_unique(database_name, collection_name, graph, alpha=alpha)
            else:
                # if not relevant cross reference just on graph data
                graph_is_unique = self.check_graph_unique(database_name, collection_name, graph)
        check_bridges = self.check_bridges(graph)
        check_min_degree = self.check_min_degree(graph)
        check_is_connected = nx.is_connected(graph)
        if not ignore_constraints:
            if check_bridges == True and check_min_degree == True and check_is_connected == True and graph_is_unique == \
                    True:
                pass
            else:
                print("failed check")
                print("bridges: \t{}\n"
                      "degree: \t{}\n"
                      "connected: \t{}\n"
                      "unique: \t{}")
                graph = self.create_real_based_grid_graph(grid_graph, L,
                                                          choice_prob=choice_prob,
                                                          database_name=database_name,
                                                          collection_name=collection_name,
                                                          ARP=ARP, waxman_BA=waxman_BA,
                                                          sequential_adding=sequential_adding,
                                                          numeric_adding=numeric_adding,
                                                          gabriel_constraint=gabriel_constraint,
                                                          relative_neighbourhood_constraint=relative_neighbourhood_constraint,
                                                          alpha=alpha, beta=beta,
                                                          BA_pure=BA_pure,
                                                          log_distance=log_distance,
                                                          waxman_graph=waxman_graph,
                                                          random=random, SBAG=SBAG,
                                                          normalise=normalise,
                                                          BA_plus_dist=BA_plus_dist,
                                                          plot_sequential_graphs=plot_sequential_graphs,
                                                          print_probs=print_probs,
                                                          overshoot=overshoot,
                                                          undershoot=undershoot,
                                                          first_start_node=first_start_node,
                                                          random_start_node=random_start_node,
                                                          centre_start_node=centre_start_node,
                                                          remove_unfeasible_edges=remove_unfeasible_edges,
                                                          verbosity=verbosity,
                                                          remove_C1_C2_edges=remove_C1_C2_edges,
                                                          max_degree=max_degree,
                                                          pref_const=pref_const,
                                                          ignore_constraints=ignore_constraints)

        graph = self.assign_distances_grid(graph, pythagorus=False,
                                           harvesine=True)
        graph = self.assign_congestion(graph)
        graph = self.assign_NSR(graph)
        logger.debug("edges asked: {}".format(L))
        logger.debug("edges got: {}".format(len(list(graph.edges))))
        logger.debug("min degree: {}".format(np.min(degree_list(graph))))
        if return_intermittent_graphs:
            return graph, intermittent_graphs
        elif return_sequence:
            return graph, node_order
        else:
            return graph

    def check_max_degree_for_link(self, graph, link, degree_max):
        """
        Method to check whether the addition of a link violates maximum degree constraints.
        :param graph:       graph to check
        :param link:        edge to check
        :param degree_max:  maximum degree constraint
        :return:            Boolean - True or False
        """
        degree_1 = (graph.degree[link[
            0]] + 1)  # check that the degree on source side is not above max
        degree_2 = (graph.degree[link[
            1]] + 1)  # check that the degree of the destination side not above max
        if link[0] != link[1] and degree_1 <= degree_max and degree_2 <= degree_max:
            return True
        else:
            return False

    def assign_distances(self, graph, scaling_factor=1.0):
        """
        This method is for assigning distances from the kernel density estimation of the NSF net

        :param scaling_factor:
        :param graph: graph to be assigned distances
        :param scaling_factor: factor by which to scale distances by: avg_link_dist/1463
        :return: graph
        :rtype: nx.Graph()
        """
        kde = self.kernel_density_pdf(
            plot=False)  # get the kernel density estimation of distances from nsf
        samples = kde.sample(len(graph.edges))  # draw samples from estimation
        samples = list(
            map(lambda x: x * scaling_factor,
                samples))  # scale them by the scaling factor (see API - assign distances)
        if list(map(lambda x: math.ceil(x / 80), samples)) == 0:
            # print("zero links")
            pass
        distances = list(
            map(lambda x, y: (x[0], x[1], abs(math.ceil((y / 80)))), graph.edges,
                samples))  # assign distances to the edges
        # logging.debug("distances: {}".format(distances))
        graph.clear()  # clear graph and assign new vertives
        graph.add_weighted_edges_from(distances)

        # print(graph.edges.data('weight'))
        # self.plot_graph(graph)
        return graph

    def calculate_harvesine_distance(self, lat_1, lat_2, lon_1, lon_2):
        """
        Method to calculate the harvesine distance between two aets of geodetic coordinates.
        :param lat_1: first latitude
        :param lat_2: second latitude
        :param lon_1: first longitude
        :param lon_2: second longitude
        """
        lat_diff = np.abs(lat_1 - lat_2)
        lon_diff = np.abs(lon_1 - lon_2)
        a = np.sin(lat_diff * np.pi / 180 / 2) * np.sin(
            lat_diff * np.pi / 180 / 2) + np.cos(lat_1 * np.pi / 180) * np.cos(
            lat_2 * np.pi / 180) * np.sin(lon_diff * np.pi / 180 / 2) * np.sin(
            lon_diff * np.pi / 180 / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = 6371 * c
        if d < 1000:
            d *= 1.5
        elif d <= 1200 and d >= 1000:
            d = 1500
        elif d > 1200:
            d *= 1.25
        return d

    def scatter_nodes_deprecated(self, nodes, scale_lon=(-180, 180), scale_lat=(-90, 90)):
        """
        DEPRECATED - Not even scattering
        Method for scattering nodes randomly on a grid to assign longitudinal and
        lattiudanal coordinates.
        :param nodes: amount of nodes to scatter.
        :return: attribute dictionary
        """
        attr_dict = {}
        for node in range(1, nodes + 1):
            lon = np.random.normal((scale_lon[1] - scale_lon[0]) / 2, (scale_lon[
                                                                           1] -
                                                                       scale_lon[
                                                                           0]) / 4, )
            lat = np.random.normal((scale_lat[1] - scale_lat[0]) / 2, (scale_lat[
                                                                           1] -
                                                                       scale_lat[
                                                                           0]) / 4, )
            attr_dict[node] = {"Longitude": lon, "Latitude": lat}
        return attr_dict

    @staticmethod
    def scatter_nodes(N, radius=400, y=[31.687, 48.1261], x=[-124.8621, -67.3952]):
        """
        Method to scatter the nodes.
        :param N:       Amount of nodes to add
        :param radius:  Minimum distances allowed between nodes
        :param y:       Latitudes of rectangle
        :param x:       Longitudes of rectangle
        :return:        nodes - dictionary {node:{"Latitude":.., "Longitude":..}}
        """
        nodes = []
        for n in range(N):
            distance_flag = True
            while distance_flag == True:
                y_pos = y[0] + np.random.random([1, ]) * (y[1] - y[0])
                x_pos = x[0] + np.random.random([1, ]) * (x[1] - x[0])
                _distances = np.array([nt.Topology.calculate_harvesine_distance(y_pos, y, x_pos, x) for y, x in nodes])
                _distance_filter = np.where(_distances < radius)

                distance_flag = np.any(_distance_filter)
            nodes.append((x_pos, y_pos))
        nodes_data = {ind + 1: {"Longitude": node[0][0], "Latitude": node[1][0]} for ind, node in enumerate(nodes)}
        return nodes_data

    def calculate_distances_grid_all(self, graph, pythagorus=False, harvesine=True,
                                     lon="Longitude",
                                     lat="Latitude",
                                     spans=True):
        """
        Method to calculate the distances between all nodes within a particular graph.
        :param graph:       nx.Graph to calculate the distances between all nodes
        :param pythagorus:  whether to use pythagorus
        :param harvesine:   whether to use harvesine formula
        :param lon:         longitude key
        :param lat:         latitude key
        :return:            np.array of node pairs and distances
        """

        node_data = graph.nodes.data()
        N = len(graph)
        distances = np.zeros((N, N))
        if pythagorus:

            for i in range(N):
                for j in range(N):
                    if j > i:
                        x_1 = node_data[i + 1][lat]
                        x_2 = node_data[j + 1][lat]
                        y_1 = node_data[i + 1][lon]
                        y_2 = node_data[j + 1][lon]
                        x_diff = np.abs(x_1 - x_2)
                        y_diff = np.abs(y_1 - y_2)
                        dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
                        distances[i][j] = dist
                        distances[j][i] = dist

        elif harvesine:
            for i in range(N):
                for j in range(N):
                    if j > i:
                        try:
                            lat_1 = node_data[i + 1][lat]
                            lat_2 = node_data[j + 1][lat]
                            lon_1 = node_data[i + 1][lon]
                            lon_2 = node_data[i + 1][lon]
                            dist = self.calculate_harvesine_distance(lat_1, lat_2, lon_1,
                                                                     lon_2)
                            if spans:
                                distances[i][j] = math.ceil(dist / 80)
                                distances[j][i] = math.ceil(dist / 80)
                            else:
                                distances[i][j] = dist
                                distances[j][i] = dist

                        except:
                            print("missing coordinates")
                            continue

        return distances

    def assign_distances_grid(self, graph, pythagorus=False, harvesine=False,
                              lon="Longitude",
                              lat="Latitude"):
        """
        This method assigns the distances of a graph using a grid system.
        :param graph: graph to assign distances
        :param pythagorus: if set to true, it calculates distances purely from pythagorus
        :param harvesine: if set to true, it calculates distances accordingly using harvesine formula
        :param lon: node attribute name for longitude - default "y" - only applies to harvesine calculation
        :param lat: node attribute name for lattitude - default "x" - only applies to harvesine calculation
        :return: graph with distances assigned

        """
        node_data = graph.nodes.data()
        if pythagorus:
            for edge in graph.edges:
                x_1 = node_data[edge[0]][lat]
                x_2 = node_data[edge[1]][lat]
                y_1 = node_data[edge[0]][lon]
                y_2 = node_data[edge[1]][lon]
                x_diff = np.abs(x_1 - x_2)
                y_diff = np.abs(y_1 - y_2)
                dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
                graph[edge[0]][edge[1]]["weight"] = dist
        elif harvesine:
            if type(graph) == nx.classes.multigraph.MultiGraph:
                # case for when the graph is a multigraph
                for edge in graph.edges(keys=True):
                    try:
                        lat_1 = node_data[edge[0]][lat]
                        lat_2 = node_data[edge[1]][lat]
                        lon_1 = node_data[edge[0]][lon]
                        lon_2 = node_data[edge[1]][lon]
                        d = self.calculate_harvesine_distance(lat_1, lat_2, lon_1,
                                                              lon_2)
                        graph[edge[0]][edge[1]][edge[2]]["weight"] = math.ceil(d / 80)
                    except:
                        print("missing coordinates")
                        continue
            else:
                for edge in graph.edges:
                    try:
                        lat_1 = node_data[edge[0]][lat]
                        lat_2 = node_data[edge[1]][lat]
                        lon_1 = node_data[edge[0]][lon]
                        lon_2 = node_data[edge[1]][lon]
                        d = self.calculate_harvesine_distance(lat_1, lat_2, lon_1,
                                                              lon_2)
                        graph[edge[0]][edge[1]]["weight"] = math.ceil(d / 80)
                    except:
                        print("missing coordinates")
                        continue

        return graph

    def assign_congestion(self, graph):
        """

        :param graph:
        :return:
        """
        congestion = list(map(lambda x: (x[0], x[1], {"congestion": 0}), graph.edges))
        graph.add_edges_from(congestion)
        return graph

    def get_avg_distance(self, graph):
        """
        This method gets the average link distance for a given input graph

        :param graph: Graph to calculate the average link distance for
        :return: average link distance
        :rtype: float
        """
        edges = graph.edges()
        # logging.debug("edge attr: {}".format(nx.get_edge_attributes(graph, "weight")))
        edge_dist = list(map(lambda x: graph[x[0]][x[1]]["weight"] * 80, edges))
        # logging.debug("edge_dist: {}".format(edge_dist))
        avg_dist = sum(edge_dist) / len(edge_dist)
        return avg_dist

    def assign_NSR(self, graph):
        """

        :param graph:
        :return:
        """
        NSR = list(map(lambda x: (x[0], x[1], {"NSR": 0}), graph.edges))
        graph.add_edges_from(NSR)
        return graph

    def check_bridges(self, graph, k=2):
        """
        Method to check that not a single edge cut can cut the given
        graph into two or more components.
        :param graph:   graph to check
        :param k:       number of edge connectivity between all components
        :return:        Boolean - True or False
        """
        bridge_components = list(nx.k_edge_components(graph, k=k))
        if len(bridge_components) > 1:
            return False
        else:
            return True

    def check_min_degree(self, graph, min_degree=2):
        """
        Method to check whether minimum degree present in graph.
        :param graph:       graph to check
        :param min_degree:  minimum degree to obey
        :return:            Boolean - True or False
        """
        for item in graph.degree:
            #   logging.debug(item)
            if item[1] < min_degree:
                return False

        return True

    def check_max_degree(self, graph, max_degree=9):
        """
        Method to check whether maximum degree is present in graph.
        :param graph:       graph to check
        :param max_degree:  maximum degree to obey
        :return:            Boolean - True or False
        """
        for item in graph.degree:
            #   logging.debug(item)
            if item[1] > max_degree:
                return False

        return True

    def check_graph_unique(self, db_name, collection, graph, **kwargs):
        """
        Method for to check whether a graph is unique within a collection and database.
        :param db_name:     name of database to cross-reference
        :param collection:  name of collection to cross-reference
        :param graph:       graph to use - networkx graph
        :return:            boolean - True or False
        """
        topology_vector = self.create_binary_topology_vector(graph).tolist()
        # print(topology_vector)
        query = {"nodes": len(graph), "edges": len(list(graph.edges))}
        for key, value in kwargs.items():
            query[key] = value
        df = Database.read_data_into_pandas(db_name, collection, find_dic=query)
        if len(df) == 0:
            return True
        topology_vectors = df["topology vector"].tolist()
        # print(topology_vectors[:3])
        if topology_vector in topology_vectors:
            return False
        else:
            return True

    def create_binary_topology_vector(self, graph):
        """
        Method to create a binary topology vector for any graph from its adjacency matrix.
        :param graph:   Graph to create binary topology vector from
        :return:        numpy array with topology vector
        """
        for edge in graph.edges():
            graph[edge[0]][edge[1]]["weight"] = 1
        A = nx.to_numpy_matrix(graph)
        N = len(list(graph.nodes()))
        topology_vector = np.ndarray([])
        for i in range(1, N):
            topology_vector = np.append(topology_vector, np.diagonal(A, i))
        topology_vector.flatten()

        return topology_vector

    def check_unique_binary_topology_vector(self, topology_vector, N, k,
                                            collection_name, database_name,
                                            scaling_factor=None, **kwargs):
        """
        Method to check that a binary topology vector is unique within a database and collection. ARCHIVE METHOD
        :param topology_vector: Binary topology vector of which to check uniqueness
        :param N:               Nodes to check in database
        :param collection_name: Collection name to cross-reference
        :param database_name:   Database name to cross-reference
        :return:                Boolean - True or False
        """
        # topology_vector_data = Database.read_data("Topology_Data", "topology_data", {"connectivity": alpha})
        if scaling_factor != None:
            df = Database.read_data_into_pandas(database_name, collection_name,
                                                {"nodes": N,
                                                 "scaling factor": scaling_factor})
        else:
            df = Database.read_data_into_pandas(database_name, collection_name,
                                                {"nodes": N, "mean k": k})
        for key, value in kwargs.items():
            topology_dict[key] = value

        # vector_data_df = pd.DataFrame({"topology vector": []})
        if len(df) == 0:
            return True
        topology_vectors = df["topology vector"].tolist()
        # for item in topology_vector_data:

        if topology_vector.tolist() in topology_vectors:
            #    vector_data_df = vector_data_df.append({"topology vector" :item["topology vector"]}, ignore_index=True)
            # print("vector_data: {}".format(type(vector_data_df["topology vector"].to_list())))
            # print("topology test data: {}".format(type(topology_vector)))
            return False
        else:
            return True

    def check_enough_topologies(self, N, k, collection_name, database_name, _len=3000):
        """

        :param N:
        :param k:
        :param collection_name:
        :param database_name:
        :return:
        """
        df = Database.read_data_into_pandas(database_name, collection_name,
                                            {"nodes": N, "mean k": k})
        if len(df) >= _len:
            return True
        else:
            return False

    def check_unique_topology_vector(self, topology_vector):
        """

        :param topology_vector: The topology vector to be tested
        :return: True or False - depending on condition
        :rtype: Boolean
        """
        topology_vector_dataset = self.read_topology_vector_dataset()
        # logging.debug(topology_vector_dataset)
        if topology_vector in topology_vector_dataset:
            return False
        else:
            return True

    def gaussian_pdf(self, span, bin_size=30, sample_number=1000, plot=True):
        """
        :param span:
        :param bin_size:
        :param sample_number:
        :param plot:
        :return:
        """
        avg = np.mean(self.weights)
        var = np.var(self.weights)
        sd = var ** (0.5)
        pd = np.random.normal(avg, sd, (sample_number))
        if plot == True:
            count, bins, ignored = plt.hist(pd, bin_size, density=True)
            plt.plot(bins, 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(
                (-(bins - avg) ** 2 / (2 * sd ** 2))), linewidth=2,
                     color='r')
            plt.show()
        return pd

    def kernel_density_pdf(self, bandwidth=250, plot=True):
        """

        :param bandwidth:
        :param plot:
        :return:
        """
        X = np.asarray(self.weights)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
        if plot == True:
            X_plot = np.linspace(0, max(self.weights), 10000)[:, np.newaxis]
            log_dens = kde.score_samples(X_plot)
            plt.hist(self.weights, 70, density=True)
            plt.plot(X_plot, np.exp(log_dens), 'r-', lw=2, alpha=0.6, label='norm pdf')
            plt.show()

        return kde


def log_all_real_topologies():
    nt.Database.delete_collection("Topology_Data", "real")
    real_topology_logger_csv("CONUS")
    sdn_lib_topology_logger("SDN")
    internet_topology_zoo("InternetZooFibre")


def real_topology_logger_csv(folder, db="Topology_Data", collection="real"):
    """
    Method to log the real CONUS topologies in the db:"Topology_Data" collection:
    "real". For the CONUS USA topologies in .csv
    :param folder: the folder with all .csv files in them.
    :return: None
    """
    import csv
    import os
    import ast
    top = Topology()
    file_nodes = []
    file_size = []
    file_pos = []
    names = []
    for filename in os.listdir(folder + "/" + "nodes"):

        names.append(filename)

        with open(folder + "/" + "nodes" + "/" + filename) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0

            pos = []
            node_names = []
            for row in csv_reader:
                pos.append((row["Latitude"], row["Longitude"]))
                node_names.append(row[r"Name"])
                line_count += 1
            print(pos[0])
            N = line_count

            print("N: {}".format(N))
        file_nodes.append(node_names)
        file_size.append(N)
        file_pos.append(pos)
    _file_nodes = iter(file_nodes.copy())
    file_edges = []
    for filename in names:  # os.listdir(folder + "/" + "links"):
        filename = filename.replace("nodes", "links")
        print(filename)
        nodes = _file_nodes.__next__()
        print(nodes)
        edges = []
        with open(folder + "/" + "links" + "/" + filename) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=",")
            line_count = 0
            pos = []
            for row in csv_reader:
                # print(nodes.index(row["S"]))
                # print(row[2])
                edges.append((nodes.index(row["S"]) + 1, nodes.index(row["D"]) + 1))
        file_edges.append(edges)
    for edges, graph_size, pos, name in zip(file_edges, file_size, file_pos, names):
        nodes = np.arange(1, graph_size + 1)
        node_attr = {}
        for i in nodes:
            node_attr[i] = {"Latitude": ast.literal_eval(pos[i - 1][0]),
                            "Longitude": ast.literal_eval(pos[i - 1][1])}

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        nx.set_node_attributes(graph, node_attr)
        graph = top.assign_distances_grid(graph, harvesine=True)
        graph = top.assign_NSR(graph)
        graph = top.assign_congestion(graph)
        node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()))
        nt.Database.insert_graph(graph, db, collection, name=name.replace(
            ".csv", ""),
                                 node_data=True, source="CONUS")


def internet_topology_zoo(folder):
    """
    Method to log topologies from internet topology zoo
    :param folder: path to folder with topologies in it.
    :return: None - saves topologies to db: "Topology_Data" collection: "real"
    """
    import os
    i = 0
    top = Topology()
    for filename in os.listdir(folder):
        if ".gml" not in filename:
            continue

        try:
            # with open(folder+'/'+filename, "r+") as file:
            # print("trying to re-write...")
            #    j = 1
            #    lines = file.readlines()
            # print(lines)
            # print("read lines of file...")
            # for line in lines:
            #    if line.startswith("  Backbone"):
            # print(line)
            #        if "0" in line:
            #            network_type = "other"
            #        elif "1" in line:
            #            network_type = "core"
            #        print(str(line))
            #        file.write(line.replace(str(line), str(j)))
            #        j +=1
            # print("about to read gml")

            graph = read_gml(folder + '/' + filename)
            graph = get_rid_of_non_connected_nodes(graph)
            print(filename)
            graph = add_estimated_missing_node_location(graph)
            # print(graph.nodes.data())
            # nt.Plotting.plot_graph(graph)

            graph = top.assign_distances_grid(graph, harvesine=True, lon="Longitude",
                                              lat="Latitude")

            graph = top.assign_NSR(graph)
            graph = top.assign_congestion(graph)
            # print(graph.nodes.data())
            nt.Database.insert_graph(graph, "Topology_Data", "real", node_data=True,
                                     name=filename.replace(".gml.txt", ""),
                                     source="ITZ")
            # nt.Plotting.plot_graph(graph)
            # ACMN = top.assign_NSR(ACMN)
            # ACMN = top.assign_congestion(ACMN)

            # print(ACMN.edges())
            # nt.Plotting.plot_graph(ACMN)
            # node_data = nt.Tools.node_data_to_database(dict(ACMN.nodes.data()))
            # print(node_data)
            # top.save_graph_database(ACMN, "Topology_Data", "real", name=filename, node_data=node_data)
            # print("saved data!!!")
            # print(filename+"    "+network_type)

        #  print(filename.replace(".gml.txt", ""))

        except Exception as err:
            handle_topology_zoo_err(err, folder, filename)


def add_estimated_missing_node_location(graph):
    """
    Method to estimate node locations for when unknown, by simply finding the centroid between all the known        neighbours.
    :param graph: graph for which to add these estimations.
    :return: graph with added new estimated locations
    """
    import re
    for ind, (node, data) in enumerate(list(graph.nodes.data())):

        try:
            Latitude = data["Latitude"]
        except:
            neighbours = graph.neighbors(node)
            # print(list(neighbours))
            x = []
            y = []
            for node_n in list(neighbours):
                #    print(graph.nodes[node_n])
                # print(graph[node_n])
                try:
                    x.append(graph.nodes[node_n]["Latitude"])
                    y.append(graph.nodes[node_n]["Longitude"])
                except:
                    pass
            x_bar = np.sum(x) / len(x)
            y_bar = np.sum(y) / len(y)
            if x_bar == np.nan or y_bar == np.nan:
                print(list(neighbours))
            # print("x_bar: {} y_bar: {}".format(x_bar, y_bar) )
            # print(node)
            attr = {node: {"Longitude": y_bar, "Latitude": x_bar}}
            nx.set_node_attributes(graph, attr)
    return graph


def handle_topology_zoo_err(err, folder, filename):
    print("error: {} in {}".format(err, filename))
    import re
    if "edge" in str(err):
        with open(folder + '/' + filename, "r+") as file:
            lines = file.readlines()

            lines.insert(1, "  multigraph 1\n")
            # print(lines)
        writing_file = open(folder + '/' + filename, "w")
        writing_file.writelines(lines)
        writing_file.close()

    #  ("error")
    if "node" in str(err):
        # print("node error")
        node = re.findall("'([^']*)'", str(err))[0]
        count = 0
        with open(folder + '/' + filename, "r+") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('    label "{}"'.format(node)) or line.startswith(
                        '		label	"{}"'.format(node)):
                    count += 1
                    lines[lines.index(line)] = '    label "{}"\n'.format(
                        node + str(count))
            file.close()

        writing_file = open(folder + '/' + filename, "w")
        writing_file.writelines(lines)
        writing_file.close()
        internet_topology_zoo(folder)
    if "input" in str(err):
        count_input = 0
        with open(folder + '/' + filename, "r+") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith('graph'):
                    count_input += 1
                    if count_input > 1:
                        index = lines.index(line)
                        print(index)
            if count_input > 1:
                lines = lines[:index]
        writing_file = open(folder + '/' + filename, "w")
        writing_file.writelines(lines)
        writing_file.close()


def get_rid_of_non_connected_nodes(graph):
    for node in list(graph.nodes):
        if graph.degree()[node] == 0:
            graph.remove_node(node)
        nodes = graph.nodes
    """for name, properties in list(graph.nodes.data()):
        try:
            lon = properties["Longitude"]
        except:
            graph.remove_node(name)
   # print(list(graph.nodes.data()))"""
    return graph


def read_gml(path, simple_graph=True):
    graph = nx.read_gml(path)
    if simple_graph:
        if type(graph) == nx.classes.multigraph.MultiGraph:
            graph = nx.Graph(graph)
    else:
        pass
    return graph


def sdn_lib_topology_logger(folder, collection="real"):
    """
    Method for to log topologies from sdnlib in .txt format.
    :param folder: Folder with .txt files inside.
    :return: None - saves topologies to db: "Topology_Data" collection: "real"
    """
    import os
    top = Topology()
    for filename in os.listdir(folder):
        print(filename)
        if filename == ".DS_Store":
            continue
        nodes = []
        edges = []
        x = []
        y = []
        node_num = 0
        start_reading_nodes = False
        start_reading_edges = False
        print(filename)
        # try:
        file = open(folder + "/" + filename, 'r')
        for line in file:
            if "NODES" in line:
                start_reading_nodes = True
                print("hello")
            if "LINKS" in line:
                start_reading_edges = True

            if line == ")\n":
                print("bye")
                start_reading_nodes = False
                start_reading_edges = False
            if start_reading_nodes:
                if "NODES" in line:
                    pass
                else:
                    print(line.split(" "))
                    line = line.split(" ")
                    nodes.append(line[2])
                    x.append(float(line[4]))
                    y.append(float(line[5]))
                    node_num += 1
            if start_reading_edges:
                if "LINKS" in line:
                    pass
                else:
                    line = line.split(" ")
                    edges.append((line[4], line[5]))

        node_range = np.arange(1, node_num + 1)
        edges = list(
            map(lambda edge: (nodes.index(edge[0]) + 1, nodes.index(edge[1]) + 1),
                edges))

        attr_dict = {node: {"Longitude": x[node - 1], "Latitude": y[node - 1]}
                     for
                     node
                     in node_range}
        ACMN = nx.Graph()
        ACMN.add_nodes_from(node_range)
        nx.set_node_attributes(ACMN, attr_dict)
        ACMN.add_edges_from(edges)
        ACMN = top.assign_distances_grid(ACMN, harvesine=True)
        ACMN = top.assign_NSR(ACMN)
        ACMN = top.assign_congestion(ACMN)

        print(ACMN.edges())
        nt.Plotting.plot_graph(ACMN)
        node_data = nt.Tools.node_data_to_database(dict(ACMN.nodes.data()))
        print(node_data)
        nt.Database.insert_graph(ACMN, "Topology_Data", collection,
                                 name=filename.replace(".txt", ""),
                                 node_data=True, source="SNDlib")
        print("saved data!!!")


class PTD():
    def __init__(self, N, E, T_c, alpha=[1]):
        self.N = N
        self.E = E
        self.T_c = [T_c]
        self.alpha = alpha
        self.top = nt.Topology.Topology()
        self.solution_graph = None
        self.algorithm_param = {'max_num_iteration': 1000,
                                'population_size': 500,
                                'mutation_probability': 0.1,
                                'elit_ratio': 0.01,
                                'crossover_probability': 0.8,
                                'parents_portion': 0.3,
                                'crossover_type': 'uniform',
                                'max_iteration_without_improv': None}

    def build_graph_from_vector(self, graph_vector):
        edges_poss = [(i + 1, j + 1) for i in range(self.N) for j in range(self.N) if j > i]
        edges = []
        for ind, item in enumerate(graph_vector):
            if item == True:
                edges.append(edges_poss[ind])
        graph = nx.Graph()
        graph.add_nodes_from(list(range(1, self.N + 1)))
        graph.add_edges_from(edges)
        return graph

    def objective(self, graph_vector):
        objective_value = 0
        # take vector
        # create nx.graph
        graph = self.build_graph_from_vector(graph_vector)
        # print(graph.edges)
        # print(graph.nodes)
        if nx.is_connected(graph) and len(list(graph.edges)) == self.E and self.top.check_bridges(
                graph) and self.top.check_min_degree(graph):
            objective_value = nt.Tools.get_demand_weighted_cost([[graph, 1]], self.T_c, self.alpha)[0]
        else:
            objective_value += 1000

        # Penalty function for graphs that don't meet this objective
        # if not self.top.check_bridges(graph) or not self.top.check_min_degree(graph):
        #     objective_value += 1000
        # objective_value += graph.number_of_edges()
        return objective_value

    def run_ga(self):
        self.varbound = np.array([[0, 1]] * int((self.N ** 2 - self.N) / 2))
        self.model = ga(function=self.objective, dimension=int((self.N ** 2 - self.N) / 2), variable_type='bool',
                        variable_boundaries=self.varbound, algorithm_parameters=self.algorithm_param)
        self.model.run()
        best_solution = self.get_solution()
        objective_value = self.model.output_dict["function"]
        return {"graph": best_solution, "objective_value": objective_value}

    def get_solution(self):
        graph_vector = self.model.output_dict["variable"]
        self.solution_graph = self.build_graph_from_vector(graph_vector)
        return self.solution_graph


if __name__ == "__main__":
    import NetworkToolkit as nt

    top = nt.Topology.Topology()
    # nt.Database.delete_collection("Topology_Data", "real")
    # sdn_lib_topology_logger("/home/uceeatz/Code/RealTopologies/SDN/")
    # real_topology_logger_csv("/home/uceeatz/Code/RealTopologies/CONUS/")
    # internet_topology_zoo("/home/uceeatz/Code/RealTopologies/InternetZoo/")

    """degree_list = lambda ACMN: np.asarray(list(
        map(lambda x: x[1],
            list(ACMN.degree))))  # get list of degrees for all nodes
    degree_list_NS = lambda N_S, ACMN: np.delete(np.asarray(list(
        map(lambda x: x[1],
            list(ACMN.degree)))), np.append(np.asarray(
        list(ACMN.neighbors(N_S))), N_S) - 1)  # get list of degrees for all nodes
    degree_sum = lambda N_S, ACMN: np.sum(
        degree_list_NS(N_S, ACMN))  # get sum of degrees for all nodes
    degree_prob = lambda ACMN: degree_list(
        ACMN) / degree_sum(
        ACMN)  # get probabilities of all nodes to be chosen to have a link added
    N_S = lambda N, ACMN: (int(np.random.choice(np.arange(1, N + 1), p=degree_prob(
        ACMN))))  # randomly choose source node given probabilites (BA probabilities for source node)

    distances_list = lambda N_S, ACMN: np.delete(np.asarray(
        [top.calculate_harvesine_distance(ACMN.nodes.data()[N_S]["x"],
                                           # get a list of distances given the source node
                                           ACMN.nodes.data()[node]["x"],
                                           ACMN.nodes.data()[N_S]["y"],
                                           ACMN.nodes.data()[node]["y"]) / 80 for
         node in ACMN.nodes()]), np.append(np.asarray(list(ACMN.neighbors(N_S))),
                                           N_S) - 1)
    distances_sum = lambda N_S, ACMN: np.sum(
        distances_list(N_S,
                       ACMN))  # get a sum of all distances given source node
    distances_prob = lambda N_S, ACMN: distances_list(N_S, ACMN) / distances_sum(
        N_S,
        ACMN)  # get probabilities of all destination nodes to be picked
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real", {}, node_data=True)
    graph = graph_list[0][0]
    choice_prob = top.choice_prob(BA_plus_dist=True)
    graph_new = top.create_real_based_grid_graph(graph,L=len(list(graph.edges())),
                                                 database_name="Topology_Data",
                                                 collection_name="test",
                                                 choice_prob_func=choice_prob)
    choice_prob = top.choice_prob(waxman_graph=True, normalise=True, alpha=1, beta=1)
    choice_prob_working = top.choice_prob(BA_pure=True)
    choice_prob_dist = top.choice_prob(BA_plus_dist=True)
    prob_list = choice_prob(2, graph)
    prob_list_working = choice_prob_working(2, graph)
    prob_list_dist = choice_prob_dist(2, graph)
    print("choice prob: {}".format(prob_list))
    print("choice prob working: {}".format(prob_list_working))
    print("choice prob distance: {}".format(prob_list_dist))
    print("sum of choice prob: {}".format(prob_list.sum()))
    print("N:{}".format(len(graph)))
    print("length of prob list: {}".format(len(prob_list)))
    print("length distance list: {}".format(len(distances_list(1,graph))))
    print("distance list: {}".format(distances_list(1, graph)))
    print("length degree list: {}".format(len(degree_list_NS(1, graph))))
    print("degree list: {}".format(degree_list_NS(1, graph)))
    print("degree list: {}".format(degree_list(graph)))

    #nt.Database.delete_collection("Topology_Data", "test")

    #i = 0
    #for graph, _id in graph_list:"""
    """
        bar = ShadyBar("preferential attachment topology {}".format(i),max=3000)        
        i += 1
        if i==3:
            bar.next()
            continue
        L = len(list(graph.edges))
        N = len(graph)
        top.create_random_ACMN_dataset(N, 3000, "Topology_Data", "test",
                                   use_dask=False, BA=False, WS=False, ER=False,
                                   random=False, pref_grid=True, alpha=None, k=None, p=None,grid_graph=graph, L=L, bar=bar)

        bar.finish()
    """

    """    pos_dtag = [(52.520008, 13.404954), (53.5753212, 10.0153399), (52.3705215, 9.7332201), (51.3396187, 12.3712902),
                 (51.2217216, 6.7761598), (50.110924, 8.682127), (49.460983, 11.061859), (48.7823, 9.177),
                 (48.1374283, 11.57549)]"""

