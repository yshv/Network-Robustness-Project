import logging
import math
import random

import networkx as nx
import numpy as np
import pandas as pd

from Heuristics import Heuristics
from ML import ML
from ILP import ILP
logging.basicConfig(level=26)


class RWA(ML, ILP, Heuristics):
    """class that can take in a graph and store RWA methods for this graph.

    :param graph: an input graph generated from Topology
    """

    def __init__(self, graph, channels, channel_bandwidth):

        self.wavelengths = {i: [] for i in range(channels)}
        self.RWA_graph = graph
        self.channels = channels
        self.channel_bandwidth = channel_bandwidth
        self.current_wavelength = 0
        self.wavelength_max = 0
        self.edge_num = len(list(self.RWA_graph.edges()))
        self.node_num = len(list(self.RWA_graph.nodes()))
        
    
    def get_paths_with_traffic(self, traffic_matrix_connection_requests, e=0):
        """
        Method to return the list of paths with regard to the traffic matrix.
        :param traffic_matrix_connection_requests: Traffic matrix to create the paths list
        :param e: MNH + e k shortest paths (int)
        :return: list of paths
        """
        self.get_k_shortest_paths_MNH(e=e)
        s_d_pairs = self.equal_cost_paths  # assign equal cost paths
        s_d_pairs = list(
            map(lambda x: (x[0][0], x[0][1], x[1]), s_d_pairs))  # converting s-d pairs without tuple for sd
        s_d_pairs = np.asarray(s_d_pairs)  # taking as numpy array - quicker
        s_d_pairs_with_traffic = np.zeros(
            (3,))  # creating numpy array for s-d based on traffic matrix TODO: better way to do this? -07/02/2020
        for item in s_d_pairs:  # create traffic based s_d array
            s_d_pairs_item = np.tile(item, (int(traffic_matrix_connection_requests[item[0] - 1, item[1] - 1]),
                                            1))  # repeating the s-d pairs based on demand matrix
            s_d_pairs_with_traffic = np.vstack(
                (s_d_pairs_with_traffic, s_d_pairs_item))  # stacking this on to the final s-d with traffic array
        s_d_pairs_with_traffic = np.delete(s_d_pairs_with_traffic, 0,
                                           0)  # delete the first zero item (initialised element) TODO: better way to do this? -07/02/2020
        paths_new = []
        for item in s_d_pairs_with_traffic:
            for path in item[2]:
                paths_new.append(path)
        return paths_new



    def sort_lightpath_routes_consecutive(self):
        """
        This method sorts the list of lightpaths according to cost (NSR) therefore reversed.
        :return: None
        """
        self.lightpath_routes_consecutive = sorted(self.lightpath_routes_consecutive, key=self.sort_cost, reverse=True)

    def assign_wavelengths(self, path, bidirectional_fiber=True, possible_fit=True):
        """
        This method assigns wavelengths to all the edges in the graph, as to tell which wavelengths are present in which link.
        :param path: path on which to assign wavelengths to graph
        :return:
        """
        print("WARNING!!!! Depricated - old inefficient WA")
        # what if the wavelengths would be a numpy 2D array, with list of tuple edgese for the paths and the np.isin reversed used to find which wavelengths are able to be assigned, the smallest index is returned.
        path_edges = self.nodes_to_edges(path)
        self.current_wavelength = 0
        assigned = False
        while not assigned:
            assigned = False
            if len(self.wavelengths[0]) == 0:
                if possible_fit:
                    return self.current_wavelength
                self.wavelengths[self.current_wavelength].append(path)
                return self.wavelengths, self.wavelength_max
            for lightpath, edge in ((w1, e1) for w1 in self.wavelengths[self.current_wavelength] for e1 in path_edges):
                if bidirectional_fiber:
                    if edge in self.nodes_to_edges(lightpath) or (
                            edge[1], edge[0]) in self.nodes_to_edges(lightpath):
                        self.current_wavelength += 1
                        assigned = False
                        break
                    else:
                        assigned = True
                        pass
                else:
                    if edge in self.nodes_to_edges(lightpath):
                        self.current_wavelength += 1
                        assigned = False
                        break
                    else:
                        assigned = True
                        pass

            if self.current_wavelength == self.wavelength_max + 1:

                if possible_fit:
                    return self.current_wavelength
                self.wavelength_max += 1
                self.wavelengths[self.wavelength_max] = [path]

                break
            elif assigned == True:

                if not possible_fit:
                    self.wavelengths[self.current_wavelength].append(path)
                elif possible_fit:
                    return self.current_wavelength
        return self.wavelengths, self.wavelength_max

    def convert_path_to_SD_cost_pair_SNR(self, path):
        """

        :param path:
        :return:
        """
        SD_cost_pair = ((path[0], path[-1]), self.path_cost(self.SNR_graph, path, weight=True), path)
        return SD_cost_pair



    def static_baroni_WA(self, initial_demand=False, bi_directional_fiber=True):
        """

        :return:
        """
        self.__init__(self.RWA_graph, self.channels, self.channel_bandwidth)
        self.W = np.zeros((len(list(self.RWA_graph.edges)), self.channels))
        for i in range(1, len(self.lightpath_routes_consecutive)):  # loop over all lightpaths
            logging.debug("i: {}".format(i))
            self.current_wavelength = 0
            wavelengths, wavelength_max = self.assign_wavelengths(
                self.lightpath_routes_consecutive[i][2], bidirectional_fiber=bi_directional_fiber,
                possible_fit=False)  # assign a wavelength on first fit principle
        if initial_demand:
            self.lambda_LL = 0
            for key in self.wavelengths:
                self.lambda_LL += 1

        logging.debug("wavelengths: {}".format(self.wavelengths))




if __name__ == "__main__":
    import NetworkToolkit as nt

    graph = nt.Tools.load_graph("ACMN_7_node")
    graph = nt.Tools.assign_congestion(graph)

    node_num = len(graph)
    demand_matrix = np.ones((node_num, node_num)) * 13
    demand_matrix[0, 2] = 20
    demand_matrix[2, 0] = 20
    rwa = RWA(graph, 156, 32e9)
    # rwa.kSP_CA_LA(traffic_matrix_connection_requests=demand_matrix)

    # print(timeit.timeit(code_FF_kSP, number=1))
    # rwa.FF_kSP(demand_matrix)
    rwa.k_SP_CA_FF(traffic_matrix_connection_requests=demand_matrix)
    rwa.test_RWA_correct()
    # rwa.kSP_CA_LA(demand_matrix)
    # rwa.FF_WA()
    print(rwa.wavelengths)
    rwa.k_SP_FF(traffic_matrix_connection_requests=demand_matrix)
    rwa.test_RWA_correct()
    print(rwa.wavelengths)
    rwa.FF_kSP(traffic_matrix_connection_requests=demand_matrix)
    rwa.test_RWA_correct()
    print(rwa.wavelengths)

    # rwa.static_baroni_WA_optimised()
    # rwa.static_baroni_WA()

# print(timeit.timeit(rwa.assign_wavelengths_optimised, number=1))

# Plan for 07/02/2020
# TODO: re-write kSP-FF-baroni optimised
# TODO: changing Topology to access methods without instantiating a class
