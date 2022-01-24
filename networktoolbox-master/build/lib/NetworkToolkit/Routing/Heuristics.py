import logging
import math
import random
from .Tools import *
from ..Tools import *
import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy


class Heuristics():
    def __init__(self, graph, channels, channel_bandwidth):
        self.graph = graph
        self.channels = channels
        self.channel_bandwidth = channel_bandwidth
        
    def k_SP_FF(self, traffic_matrix_connection_requests, e=0, k=1, rwa_assignment_previous=None):
        """
        Method that implements the k shortest path first fit algorithm for RWA.
        :param traffic_matrix_connection_requests: NxN matrix of connections to be established
        :param e: k+e length paths to be used
        :return: rwa assignment - dictionary
        """
        k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        LA = self.k_SP_LA(self.graph, k_SP, traffic_matrix_connection_requests)
        rwa_assignment = self.FF_WA(self.graph, LA, self.channels, self.channel_bandwidth, rwa_assignment_previous=rwa_assignment_previous)
        return rwa_assignment

    def k_SP_FF_revised(self, traffic_matrix_connection_requests, e=0, k=1, order_aware=False,
                        sort_length=False, rwa_assignment_previous=None, return_blocked=False,
                        random_sorting=False, connection_pairs=None, _ids=None, save_ids=False):
        """
        :param traffic_matrix_connection_requests:
        :param e:
        :param k:
        :param order_aware:
        :param sort_length:
        :param rwa_assignment_previous:
        :return:
        """
        if connection_pairs is None:
            k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
            s_d_pairs = k_SP  # assign equal cost paths
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
        elif connection_pairs is not None:
            s_d_pairs_with_traffic = []
            for s,d in connection_pairs: # iterate over the connection pairs
                # if (s, d) == (94, 21):
                #     print(rwa_assignment_previous)
                k_sp = k_shortest_paths(self.graph, s, d, k, weight="weight")  # find the k-shortest paths for the source destination pair
                # k_sp = list(islice(nx.all_simple_paths(self.graph, source=s, target=d), k))
                # k_sp = []
                # print(k_sp)
                s_d_pairs_with_traffic.append((s,d, k_sp)) # append these to the list with the kshortest paths




        current_path = np.zeros((len(s_d_pairs_with_traffic), 3),
                                object)  # initialise array to hold the path for a s-d connection request
        if rwa_assignment_previous is not None:
            rwa_assignment = rwa_assignment_previous
            W = np.zeros((len(list(self.graph.edges)), self.channels))
            for key in rwa_assignment:
                for path in rwa_assignment[key]:
                    W = self.add_wavelength_path_to_W(self.graph, W, path, key)  # altering the W matrix to add the wavelength
        else:
            rwa_assignment = {i: [] for i in range(self.channels)}
            W = np.zeros((len(list(self.graph.edges)), self.channels))
        if random_sorting: random.shuffle(s_d_pairs_with_traffic)
        blocked_connections = 0
        blocked_ids = []
        additional_id_info = []
        for idx, item in enumerate(s_d_pairs_with_traffic):  # loop over all equal cost paths according to demand matrix
            path_wavelength = np.zeros((len(item[2]),),
                                       dtype=np.int)  # intialise array for wavelength number of each path
            path_len = np.zeros((len(item[2]),),
                                       dtype=np.int)
            paths_wavelengths = []

            for index, path in enumerate(item[2]):

                FF = self.FF_return(self.graph, self.channels, W, path)
                path_wavelength[index] = FF  # find the wavelength number for FF for each path
                paths_wavelengths.append((path, FF))
                if FF > self.channels:
                    # print(FF)
                    # print(self.channels)
                    # print(path_len)
                    pass
                elif FF <= self.channels:
                    path_len[index] = len(path)
            # if (item[0], item[1]) == (85, 96):
            #     # print(item[2])
            #     print(paths_wavelengths)
            #     print(path_len)
            #     print(len(path_len))
            #     print(sorted(item[2], key=lambda x: len(x)))
            #     print(nx.dijkstra_path(self.graph, 85, 96, weight="test"))
                # _paths = nx.all_simple_paths(self.graph, source=item[0], target=item[1], cutoff=14)
                # print(list(_paths))
            # indeces = np.argsort(path_len)[np.where(path_len!=0)]

            path_wavelength = path_wavelength[np.where(path_len!=0)]
            paths = np.asarray(item[2])[np.where(path_len!=0)]
            paths_wavelengths = np.asarray(paths_wavelengths)[np.where(path_len != 0)]
            path_len = path_len[np.where(path_len!=0)]


            # print(path_len)
            # print(path_wavelength)
            if len(path_len) == 0:
                # print("blocked")
                if return_blocked:
                    blocked_connections += 1
                    blocked_ids.append(_ids[idx])
                    continue
                else:
                    return True

            index_array = np.argsort(path_len)  # sort and return index of sort
            # index_min = index_array[0]  # find min index
            index_min = index_array[0]



            current_path[idx, 0] = item[0]  # source
            current_path[idx, 1] = item[1]  # destination
            current_path[idx, 2] = paths[index_min]  # set index of path with lowest wavelength returned
            rwa_assignment[paths_wavelengths[index_min][1]].append(
                paths_wavelengths[index_min][0])  # adding path to wavelength dictionary
            if save_ids:
                additional_id_info.append({"id":int(_ids[idx]), "path":[int(node) for node in paths_wavelengths[index_min][0]], "wavelength":int(paths_wavelengths[index_min][1]),
                                                 "source":int(paths_wavelengths[index_min][0][0]), "destination":int(paths_wavelengths[index_min][0][-1])})
            # print(item[2][index_min])
            # if (paths_wavelengths[index_min][0], paths_wavelengths[index_min][1]) != (paths[index_min], path_wavelength[index_min]):
            #     print("path_wavelength: {}".format(path_wavelength))
            #     print("paths: {}".format(paths))
            #     print("path_len: {}".format(path_len))
            #     print("path wavelengths: {}".format(paths_wavelengths))
            #     print((paths[index_min], path_wavelength[index_min]))
            #     print((paths_wavelengths[index_min][0], paths_wavelengths[index_min][1]))
            # try:
            # assert (paths_wavelengths[index_min][0], paths_wavelengths[index_min][1]) == (paths[index_min].tolist(), path_wavelength[index_min])
            assert len(paths[index_min]) == min(path_len)
            # except:
            #     print(paths_wavelengths[index_min][0])
            #     print(paths_wavelengths[index_min][1])
            #     print((paths[index_min].tolist(), path_wavelength[index_min]))
            #     print(paths)
            W = self.add_wavelength_path_to_W(self.graph, W, paths_wavelengths[index_min][0],
                                              paths_wavelengths[index_min][1])  # altering the W matrix to add the wavelength
        if return_blocked and not save_ids:
            check_rwa_validity(rwa_assignment)
            return rwa_assignment, blocked_connections
        elif return_blocked and save_ids:
            check_rwa_validity(rwa_assignment)
            return rwa_assignment, blocked_connections, additional_id_info
        else:
            return rwa_assignment



    def k_SP_baroni_FF(self, traffic_matrix_connection_requests, e=0, k=1, rwa_assignment_previous=None):
        """
        Method to do the lightpath assignement with k-SP baroni style and the wavelength with first fit.

        :param traffic_matrix_connection_requests: The number NxN array that holds the traffic connection requests.
        :return: rwa assignment - dictionary
        """
        k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        LA = self.static_optimised_baroni_MNH_LA(self.graph, traffic_matrix_connection_requests, k_SP, e=e)
        rwa_assignment = self.FF_WA(self.graph, LA, self.channels, self.channel_bandwidth, rwa_assignment_previous=rwa_assignment_previous)
        return rwa_assignment


    def k_SP_CA_FF(self, traffic_matrix_connection_requests, e=0, k=1):
        """
        Method to do the lightpath assignment with k-shortest-paths congestion-aware and the wavelength assignment with first fit.
        :param traffic_matrix_connection_requests: The number NxN array that holds the traffic connection requests.
        :return: rwa dictionary - dictionary
        """
        k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        LA = self.kSP_CA_LA(self.graph, k_SP, traffic_matrix_connection_requests)
        rwa_assignment = self.FF_WA(self.graph, LA, self.channels, self.channel_bandwidth)
        return rwa_assignment
        
    def FF_kSP(self, traffic_matrix_connection_requests, e=0, k=1, order_aware=False, sort_length=False,
               rwa_assignment_previous=None, return_blocked=False, random_sorting=False, connection_pairs=None,
               _ids=None, save_ids=False):
        """
        This method implements a first fit k shortest paths algorithm for a optical transport network. Call a k-shortest path method before calling this method.

        :param traffic_matrix_connection_requests: The number NxN array that holds the traffic connection requests.
        :return: rwa assignment - assigns light paths and wavelengths to global variables that can then be accessed.
        :rtype: dictionary
        """
        if connection_pairs is None:
            k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
            s_d_pairs = k_SP  # assign equal cost paths
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
        else:
            s_d_pairs_with_traffic = []
            for s,d in connection_pairs:
                k_sp = k_shortest_paths(self.graph, s, d, k, weight="weight")
                s_d_pairs_with_traffic.append((s,d, k_sp))
        current_path = np.zeros((len(s_d_pairs_with_traffic), 3),
                                object)  # initialise array to hold the path for a s-d connection request
        if rwa_assignment_previous is not None:
            rwa_assignment = rwa_assignment_previous
            W = np.zeros((len(list(self.graph.edges)), self.channels))
            for key in rwa_assignment:
                for path in rwa_assignment[key]:
                    W = self.add_wavelength_path_to_W(self.graph, W, path, key)  # altering the W matrix to add the wavelength
        else:
            rwa_assignment = {i: [] for i in range(self.channels)}
            W = np.zeros((len(list(self.graph.edges)), self.channels))
        if random_sorting: random.shuffle(s_d_pairs_with_traffic)
        blocked_connections = 0
        additional_id_info = []
        for idx, item in enumerate(s_d_pairs_with_traffic):  # loop over all equal cost paths according to demand matrix
            path_wavelength = np.zeros((len(item[2]),),
                                       dtype=np.int)  # intialise array for wavelength number of each path
            paths_wavelengths = []

            for index, path in enumerate(item[2]):
                FF = self.FF_return(self.graph, self.channels, W, path)
                path_wavelength[index] = FF  # find the wavelength number for FF for each path
                paths_wavelengths.append((path, FF))
            index_array = np.argsort(path_wavelength)  # sort and return index of sort
            index_min = index_array[0]  # find min index
            if path_wavelength[index_min] == self.channels + 1:
                # print("blocked")
                if return_blocked:
                    blocked_connections += 1
                    continue
                else:
                    return True
            current_path[idx, 0] = item[0]  # source
            current_path[idx, 1] = item[1]  # destination
            current_path[idx, 2] = item[2][index_min]  # set index of path with lowest wavelength returned
            rwa_assignment[path_wavelength[index_min]].append(
                item[2][index_min])  # adding path to wavelength dictionary
            if save_ids:
                additional_id_info.append(
                    {"id": int(_ids[idx]), "path": [int(node) for node in paths_wavelengths[index_min][0]],
                     "wavelength": int(paths_wavelengths[index_min][1]),
                     "source": int(paths_wavelengths[index_min][0][0]),
                     "destination": int(paths_wavelengths[index_min][0][-1])})
            assert paths_wavelengths[index_min] == (item[2][index_min], path_wavelength[index_min])
            assert path_wavelength[index_min] == min(path_wavelength)
            W = self.add_wavelength_path_to_W(self.graph, W, item[2][index_min],
                                              path_wavelength[index_min])  # altering the W matrix to add the wavelength


        if return_blocked and not save_ids:
            check_rwa_validity(rwa_assignment)
            return rwa_assignment, blocked_connections
        elif return_blocked and save_ids:
            check_rwa_validity(rwa_assignment)
            return rwa_assignment, blocked_connections, additional_id_info
        else:
            return rwa_assignment
        
    def k_SP_LA(self, graph, k_SP, traffic_matrix_connection_requests):
        """
        Method for the lightpath assignment after the k-shortest path scheme.
        :param graph:                               Graph to use - nx.Graph()
        :param k_SP:                                list of k-shortest paths - list returned from Tools.get_k_shortest_paths_MNH(graph, e=e)
        :param traffic_matrix_connection_requests:  NxN numpy array that gives the connection requests for the scheme to assign. - nparray
        :return:                                    Lightpaths - list 
        """
        s_d_pairs = k_SP.copy()
        s_d_pairs = list(
            map(lambda x: (x[0][0], x[0][1], x[1]), s_d_pairs))  # converting s-d pairs without tuple for sd

        s_d_pairs = np.asarray(s_d_pairs)  # taking as numpy array
        s_d_pairs_with_traffic = np.zeros(
            (3,))  # creating numpy array for s-d based on traffic matrix TODO: better way to do this? -07/02/2020
        for item in s_d_pairs:
            s_d_pairs_item = np.tile(item, (int(traffic_matrix_connection_requests[item[0] - 1, item[1] - 1]),
                                            1))  # repeating the s-d pairs based on demand matrix

            s_d_pairs_with_traffic = np.vstack(
                (s_d_pairs_with_traffic, s_d_pairs_item))  # stacking this on to the final s-d with traffic array

        s_d_pairs_with_traffic = np.delete(s_d_pairs_with_traffic, 0,
                                           0)  # delete the first zero item (initialised element) TODO: better way to do this? -07/02/2020
        current_path = np.zeros((len(s_d_pairs_with_traffic), 3),
                                object)  # initialise array to hold the path for a s-d connection request

        for index, item in enumerate(s_d_pairs_with_traffic):  # loop over all equal cost paths

            logging.debug(item)

            len_paths = list(map(lambda x: len(x), item[2]))
            MNH_path = item[2][0]  # find MNH path in equal cost paths
            assert len(MNH_path) == min(len_paths)
            current_path[index, 0] = item[0]
            current_path[index, 1] = item[1]

            current_path[index, 2] = MNH_path
        return current_path
        
    def kSP_CA_LA(self, graph, k_SP, traffic_matrix_connection_requests):
        """
        This method assigns the lightpaths for MNH weighting and congestion awareness.
        :param graph:                               Graph to use - nx.Graph()
        :param k_SP:                                List of k shortest paths - list
        :param traffic_matrix_connection_requests:  NxN numpy array that gives the connection requests for the scheme to assign. - nparray
        :return:                                    List of light paths - list
        """

        s_d_pairs = k_SP.copy()
        s_d_pairs = list(
            map(lambda x: (x[0][0], x[0][1], x[1]), s_d_pairs))  # converting s-d pairs without tuple for sd
        s_d_pairs = np.asarray(s_d_pairs)  # taking as numpy array
        s_d_pairs_with_traffic = np.zeros(
            (3,))  # creating numpy array for s-d based on traffic matrix TODO: better way to do this? -07/02/2020
        for item in s_d_pairs:
            s_d_pairs_item = np.tile(item, (int(traffic_matrix_connection_requests[item[0] - 1, item[1] - 1]),
                                            1))  # repeating the s-d pairs based on demand matrix
            s_d_pairs_with_traffic = np.vstack(
                (s_d_pairs_with_traffic, s_d_pairs_item))  # stacking this on to the final s-d with traffic array
        s_d_pairs_with_traffic = np.delete(s_d_pairs_with_traffic, 0,
                                           0)  # delete the first zero item (initialised element) TODO: better way to do this? -07/02/2020
        current_path = np.zeros((len(s_d_pairs_with_traffic), 3),
                                object)  # initialise array to hold the path for a s-d connection request
        for index, item in enumerate(s_d_pairs_with_traffic):  # loop over all equal cost paths
            len_paths = list(map(lambda x: len(x), item[2]))
            MNH_path = item[2][0]  # find MNH path in equal cost paths
            assert len(MNH_path) == min(len_paths)
            current_path[index, 0] = item[0]
            current_path[index, 1] = item[1]
            current_path[index, 2] = MNH_path
            add_congestion(graph, MNH_path)  # add congestion for the best SNR path
        while True:  # loop over all source destination pairs until no more subs can be made
            subbed = False  # variable to keep track of if anything was able to be substituted
            random.shuffle(s_d_pairs)
            for index, item in enumerate(s_d_pairs_with_traffic):  # loop over all source destination equal cost paths
                for path in item[2]:  # loop over all possible paths in that source destination pair
                    logging.debug("path: {}".format(path))
                    replace = check_congestion(graph, current_path[index, 2],
                                                    path)  # check if path should be replaced
                    if replace:
                        remove_congestion(graph, current_path[index, 2])  # remove 1 congestion from all previous links
                        add_congestion(graph, path)  # add 1 congestion to all new links
                        current_path[index, 2] = path  # set new current path
                        subbed = True  # smth was substituted
                    else:
                        pass
            if not subbed:
                logging.debug("not subbed")
                break  # break if nothing was subbed
        self.lightpath_routes_consecutive_single = current_path
        self.lightpath_routes_consecutive = current_path
        return current_path
    def phase2_baroni_LA(self, graph, Z):
            current_path = np.zeros((len(Z), 3),
                                object)  # initialise array to hold the path for a s-d connection request

            # Phase II: initial path assignment
            # print(len(Z))
            delta_A = [[0 for k in range(len(Z[z][2]))] for z in range(len(Z))]
            
            for ind_z, z in enumerate(Z):
                MAX_load = np.inf
                PATH_load = np.inf
                for ind_k, k in enumerate(z[2]):
                    M_load = get_most_loaded_path_link_cong(graph, k)
                    P_load = get_sum_congestion_path(graph, k)
                    # print("M_load-MAX_load: {}-{} P_load-PATH_load: {}-{}".format(M_load, MAX_load, P_load, PATH_load))
                    # print("k: {}".format(k))
                    if M_load < MAX_load or (M_load == MAX_load and P_load < PATH_load):
                        MAX_load = M_load
                        PATH_load = P_load
                        p_star = k
                        # print("p_star: {}".format(p_star))
                        current_path[ind_z][0] = k[0]
                        current_path[ind_z][1] = k[-1]
                        current_path[ind_z][2] = k
                        # current_path[ind_z][3] = z[3]
                    else:
                        current_path[ind_z][0] = k[0]
                        current_path[ind_z][1] = k[-1]
                        current_path[ind_z][2] = z[2][0]
                        # current_path[ind_z][3] = z[3]
                # print("list index: {}".format(z[1].index(current_path[z[0][0] - 1][z[0][1] - 1])))
                delta_A[ind_z][z[2].index(current_path[ind_z][2])] = 1
                add_congestion(graph, current_path[ind_z][2])
            return delta_A, current_path

    def static_optimised_baroni_MNH_LA(self, graph, traffic_matrix_connection_requests, k_SP, e=5):
        """
        This method implements the optimised baroni MNH lightpath assignment. Call k-shortest path algorithm before for the self.equal_cost_path variable to be available.
        It will assign a lightpath assignment to a single uniform demand between every node pair in the graph.
        :param graph:   Graph to use - nx.Graph()
        :param k_SP:    List of k-shortest paths - list
        :param e:       Len(MNH) + e to allow for finding new k-shortest paths - int
        :return:        List of lightpaths - list
        :rtype: list
        """
        current_path = [[0 for i in range(nx.number_of_edges(graph))] for j in
                        range(nx.number_of_edges(graph))]
        Z = get_connection_requests_k_shortest_paths(k_SP, traffic_matrix_connection_requests)
        Z = sorted(Z, reverse=True, key=lambda x: len(x[2][0]))
        current_path = np.zeros((len(Z), 3),
                                object)  # initialise array to hold the path for a s-d connection request

        # Phase II: initial path assignment
        # print(len(Z))
        delta_A, current_path = self.phase2_baroni_LA(graph, Z)

        # Phase III: subsequent optimisation
        delta_C = 0
        # print("delta_C: {} and delta_A: {}".format(delta_C, delta_A))
        Z = np.asarray([[Z[i][0], Z[i][1], Z[i][2], i] for i in range(len(Z))])
        import copy
        index = 0
        while delta_C != delta_A:
            index += 1
            if index == 1000:
                break

            delta_C = copy.deepcopy(delta_A)


            X = copy.deepcopy(Z)
            X = sorted(Z, reverse=True, key=lambda x: (len(x[2][0]), random.random()))
            # print("Z: {}".format(Z))
            # print("X: {}".format(X))
            for ind_z, z in enumerate(X):
                delta_C[z[3]][z[2].index(current_path[z[3]][2])] = 0
                remove_congestion(graph, current_path[z[3]][2])
                current_path[z[3]][2] = []
                MAX_load = np.inf
                PATH_load = np.inf

                for k in z[2]:
                    M_load = get_most_loaded_path_link_cong(graph, k)
                    P_load = get_most_loaded_path_link(graph, k)
                    if M_load < MAX_load or (M_load == MAX_load and P_load < PATH_load):
                        MAX_load = M_load
                        PATH_load = P_load
                        p_star = k
                        current_path[z[3]][2] = k
                    else:
                        current_path[z[3]][2] = z[2][0]
                #      print("delta_C: {} and delta_A: {}".format(delta_C, delta_A))
                # print("z index: {} k index: {}".format(Z.index(z), z[1].index(current_path[z[0][0] - 1][z[0][1] - 1])))
                # print("shape of delta A: {}".format(np.shape(delta_A[7])))
                # print(delta_A)
                delta_C[z[3]][z[2].index(current_path[z[3]][2])] = 1
                add_congestion(graph, current_path[z[3]][2])
        #    print("delta_C: {} and delta_A: {}".format(delta_C, delta_A))

        #  for z in Z:
        #     print("S-D: {}-{}".format(z[0][0], z[0][1]))
        #      print("current path: {}".format(current_path[z[0][0]-1][z[0][1]-1]))
        # print(current_path)
        #  print("max cong: {}".format(self.get_max_cong(self.RWA_graph)))
        # s_d_pairs = list(map(lambda x: x[0], k_SP))
        # # logging.debug("lightpath routes: {}".format(current_path[0][3]))
        # self.lightpath_routes = current_path
        # self.SD_pairs = s_d_pairs
        # paths = []
        # for edge in self.SD_pairs:
        #     path = self.lightpath_routes[edge[0] - 1][edge[1] - 1]
        #     cost = path_cost(graph, path, weight=True)
        #     paths.append((edge, cost, path))
        # paths = sorted(paths, key=sort_length, reverse=True)
        # return paths
        return current_path
        
    def static_baroni_simple_SNR_LA(self, graph, k_SP, iter=100000,
                                    initial_demand=True):
        """
        Deprecated -  update
        This method assigns lightpaths given SNR weighting and also congestion awareness.
        :param initial_demand:
        :param iter: limit for the iteration of to try and ease the congestion on highly congested links
        :return: None
        """
        from ..PhysicalLayer import  PhysicalLayer
        pl = PhysicalLayer(self.graph,self.channels, self.channel_bandwidth)
        pl.add_wavelengths_full_occupation(channels_full=self.channels)
        pl.add_uniform_launch_power_to_links(self.channels)
        pl.add_non_linear_NSR_to_links(channels_full=self.channels,
                                       channel_bandwidth=self.channel_bandwidth)
        SNR_list = pl.get_SNR_k_SP(self.channels, k_SP)

        avg_congestion = []
        equal_cost_paths = k_SP
        current_path = [[[] for i in range(nx.number_of_edges(graph))] for i in
                        range(nx.number_of_edges(graph))]
        for ind, item in enumerate(equal_cost_paths):  # loop over all equal cost paths
            logging.debug(item)
            best_SNR_path = item[1][SNR_list[ind][1].index(min(SNR_list[ind][1]))] # find best SNR path in equal cost paths
            # best_SNR_path = item[1][-1]
            current_path[item[0][0] - 1][
                item[0][1] - 1] = best_SNR_path  # assign this as initial lightpath for source destination pair
            add_congestion(graph, best_SNR_path)  # add congestion for the best SNR path
            subbed = False
        for i in range(0, iter):  # loop over all source destination pairs until no more subs can be made
            subbed = False  # variable to keep track of if anything was able to be substituted
            for item in equal_cost_paths:  # loop over all source destination equal cost paths
                for path in item[1]:  # loop over all possible paths in that source destination pair

                    replace = check_congestion(graph, current_path[item[0][0]][item[0][
                        1]],
                                                    path)  # check if path should be replaced
                    if replace:

                        remove_congestion(graph, current_path[item[0][0]][
                                                   item[0][
                                                       1]])  # remove 1 congestion from all previous links
                        add_congestion(graph, path)  # add 1 congestion to all new links
                        current_path[item[0][0]][item[0][1]] = path  # set new current path

                        link = get_most_loaded_path_link(path)

                        subbed = True  # smth was substituted
                    else:
                        pass
            if not subbed:
                break  # break if nothing was subbed
            avg_congestion.append(get_average_congestion())
        df = pd.DataFrame(avg_congestion)
        df.to_csv(path_or_buf="Data\htest.csv")

        s_d_pairs = list(map(lambda x: x[0], equal_cost_paths))

        self.lightpath_routes = current_path
        self.SD_pairs = s_d_pairs
        paths = []
        for edge in self.SD_pairs:
            path = self.lightpath_routes[edge[0]][edge[1]]
            cost = path_cost(graph, path, weight=True)
            paths.append((edge, cost, path))
        paths = sorted(paths, key=sort_cost, reverse=True)
        if initial_demand:
            self.lightpath_routes_consecutive = paths
        self.lightpath_routes_consecutive_single = paths
        
        
    def add_wavelength_path_to_W(self, graph, W, path, wavelength):
        """
        Method to add a wavelength to a W matrix
        :param graph:       Graph to use - nx.Graph()
        :param W:           W matrix to add to - ndarray
        :param path:        Path to add the wavelength to - list
        :param wavelength:  Wavelength to add to - int
        :return:            The new W matrix - ndarray
        """
        graph_edges = list(graph.edges())
        path_edges = nodes_to_edges(path)
        for index, (s,d) in enumerate(graph_edges):
            if (s,d) in path_edges or (d,s) in path_edges:
                W[index, wavelength] = 1
        return W

    def FF_return(self, graph, channels, W, path):
        """
        Method to return the first available wavelength for a path W: ExChannes matrix and a path.
        :param graph:       Graph to use - nx.Graph()
        :param channels:    Amount of Channels to use - int
        :param W:           Exchange matrix - ndarray
        :param path:        Path to find wavelength for - list
        :return:            Wavelength for path - int
        """
        P = np.zeros((len(list(graph.edges)), 1))  # path vector, Ex1, 1 if path includes edge, 0 if not
        graph_edges = list(graph.edges())
        path_edges = nodes_to_edges(path)
        for index, (s,d) in enumerate(graph_edges):  # creating path vector
            if (s,d) in path_edges or (d,s) in path_edges:
                P[index] = 1

        a = np.einsum('ij,ij->j', W, P)  # finding vector multiplication and then column sum
        #print(a)
        indeces = np.where(a == 0)  # only when the column sum is 0 means its a valid wavelength
        # print("graph edges: {}".format(graph_edges))
        # print("P: {}".format(P))
        # print("W: {}".format(W))
        # print("path: {}".format(path))
        # print("a: {}".format(a))
        # print("indeces: {}".format(indeces))
        # print(np.multiply(W, P))
        # print(np.sum(np.multiply(W, P), axis=1))
        try:
            min_wave = np.min(indeces)  # if no available wavelength it crashes
        except Exception as err:
            # print("ERROR: {}".format(err))
            return channels + 1  # return the +1 argument and handle externally
        return min_wave  # otherwise return the wavelength

    def FF_WA(self, graph, LA, channels, channel_bandwidth, rwa_assignment_previous=None):
        """
        Method to assign wavelengths for all paths previously assigned.
        :param graph:               Graph to use - nx.Graph()
        :param LA:                  Lightpath assignment - list
        :param channels:            Amount of channels available - int
        :param channel_bandwidth:   Channel bandwidth available - float
        :return:                    Wavelength assignment dictionary - dict
        """
        if rwa_assignment_previous is not None:
            rwa_assignment = rwa_assignment_previous
            W = np.zeros((len(list(self.graph.edges)), self.channels))
            for key in rwa_assignment:
                for path in rwa_assignment[key]:
                    W = self.add_wavelength_path_to_W(self.graph, W, path,
                                                      key)  # altering the W matrix to add the wavelength
            # print("W: {}".format(np.shape(W)))
        else:
            rwa_assignment = {i: [] for i in range(channels)}
            W = np.zeros((len(list(graph.edges)), channels))  # initialise W matrix

        for item in LA:  # loop over all lightpaths
            path = item[2]  # path of lightpath
            P = np.zeros((len(list(graph.edges)), 1))  # find the path vector P
            graph_edges = list(graph.edges())
            path_edges = nodes_to_edges(path)
            for index, item in enumerate(graph_edges):
                if item in path_edges or item[::-1] in path_edges:
                    P[index] = 1

            a = np.einsum('ij,ij->j', W,
                          P)  # find the column vector element wise multiplication and then the column vector sum
            indeces = np.where(a == 0)  # when the column vetor sum is 0 then it is a valid wavelength
            try:
                min_wave = np.min(indeces)
            except:
                #print("blocked")
                return True
            rwa_assignment[min_wave].append(path)
            for index, item in enumerate(graph_edges):
                if item in path_edges or item[::-1] in path_edges:
                    W[index, min_wave] = 1
        return rwa_assignment

    def edge_disjoint_routing(self, graph, connection_requests, T=8):
        connection_requests_list = []
        # current_path = self.k_SP_LA(graph, k_SP, connection_requests)
        # graph = update_congestion(graph, LA=current_path)
        # # current_path = [[[] for i in range(nx.number_of_edges(graph))] for i in
        # #                 range(nx.number_of_edges(graph))]
        # s_d_pairs = get_connection_requests_k_shortest_paths(k_SP, connection_requests)
        # graph.graph["current paths"] = current_path
        # graph.graph["connection requests"] = connection_requests
        # graph.graph["k-SP M"] = s_d_pairs

        for i in range(len(connection_requests)):
             for j in range(len(connection_requests)):
                if j > i:
                    for con in range(int(connection_requests[i, j])):
                        connection_requests_list.append((i, j))
        mu_len = len(connection_requests_list)
        num_nodes = len(graph)
        print(list(graph.nodes))
        edges = list(graph.edges)
        print(edges)
        # C = np.zeros((num_nodes, num_nodes))
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         if (i,j) in edges or (j, i) in edges:
        #             C[i,j] = graph[i][j]

        I = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        I_mu = np.asarray([np.asarray([np.asarray([0 for mu in range(mu_len)]) for i in range(num_nodes)]) for j in range(
                num_nodes)])
        lam = np.zeros((num_nodes, mu_len))
        for mu, item in enumerate(connection_requests_list):
            lam[item[0], mu] = 1
            lam[item[1], mu] = -1


        
        C_graph = nx.to_numpy_array(graph, weight="weight")
        C = np.zeros((num_nodes, num_nodes, 2*mu_len+1))
        for i in range(num_nodes):
            for j in range(num_nodes):
                for mu in range(2*mu_len+1):
                    if mu != 0:
                        C[i,j,mu] = C_graph[i,j]
                    else:
                        C[i, j, mu] = 0
        print(C)
        # exit()
        E_1 = np.zeros((num_nodes, num_nodes, 2*mu_len+1))
        E = np.random.rand(num_nodes, num_nodes, 2*mu_len+1)*0.1+3

        for t in range(T):
            if t == 0:
                pass
            else:

                d_t = self.calculate_convergence_variable(graph, E, E_1, C)

                E = np.copy(E_1)
                print("d_t: {}".format(d_t))
            if t == 150:
                exit()
            # cycle through time steps
            for node_i in graph.nodes:
                self.calculate_E_t_1(graph, node_i, E, E_1, C, t)

            # print("E(t+1) = {}".format(E_1))


            # for (i, j) in graph.edges:
            #     delta_i = list(graph.neighbors(i)) # neighbors of i without j
            #     delta_i.remove(j)
            #     for source, destination, paths in M:
            #         minimisation = []
            #         minimisation_paths = []
            #         for l in delta_i:
            #             graph_ijl_nodes = deepcopy(delta_i)
            #             graph_ijl_nodes.remove(i)
            #             graph_ijl_nodes.remove(l)
            #             graph_ijl = nx.complete_graph(graph_ijl_nodes)
            #             q_min = self.evaluate_q_min(graph, graph_ijl, (j,l))
            #             E_t, path = self.evaluate_min_cost(graph, (l, i), source, destination, paths)
            #             minimisation.append(q_min+E_t)
            #             minimisation_paths.append(path)
            #         minimised = np.argmin(minimisation)
            #         min_path = minimisation_paths[minimised]
            #         graph = remove_congestion(graph, graph.graph["current paths"][source-1][destination-1])
            #         graph = add_congestion(graph, min_path)
            #         graph.graph["current paths"][source-1][destination-1] = min_path
            #         cost = self.evaluate_edge_cost(graph, (i, j))

    def get_delta_i_minus(self, graph, i, _list):
        delta_i = list(graph.neighbors(i))
        for item in _list:
            delta_i.remove(item)
        return delta_i

    def calculate_convergence_variable(self, graph, E, E_1, C):
        M = int((np.shape(E)[2]-1)/2)
        d_t=0
        for i, j in graph.edges:
            mu_ijt = np.min(E[i-1,j-1,:]+E[i-1,j-1,:]-C[i-1,j-1, :])
            mu_ijt1 = np.min(E_1[i-1,j-1,:]+E_1[i-1,j-1,:]-C[i-1,j-1, :])
            print("delta_1: {}".format(mu_ijt))
            print("delta_2: {}".format(mu_ijt1))
            delta = mu_ijt1-mu_ijt

            d_t += 1-delta
        return d_t

    def calculate_E_t_1(self, graph, node_i, E, E_1, C, t):
        M = int((np.shape(E)[2]-1)/2)
        i = node_i - 1
        # for every node in the graph
        if nx.degree(graph, node_i) == 2:
            for node_j in graph.neighbors(node_i):
                j = node_j - 1
                for mu in range(np.shape(E)[2]):
                    L = self.get_delta_i_minus(graph, node_i, [node_j])
                    E_tli = []
                    for l in L:
                        # print("E[{},{},{}]: {}".format(l, i, mu, E[l-1, i, mu]))
                        # print(E[szszaeal-1,i])

                        E_tli.append(E[l - 1, i, mu])
                    E_tli = np.asarray(E_tli)
                    # print("min: {}".format(np.min(np.add(E_tli, q_min)) + C[i, j, mu]))
                    C[i, j, mu] = self.calculate_reinforcement_param(E, C, 0.002, i, mu, i, j)
                    E_1[i, j, mu] = E_tli[0] + C[i, j, mu]
        elif nx.degree(graph, node_i) == 3:
            for node_j in graph.neighbors(node_i):
                j = node_j - 1
                for mu in range(np.shape(E)[2]):
                    L = self.get_delta_i_minus(graph, node_i, [node_j])
                    E_tli = []
                    for l in L:
                        # print("E[{},{},{}]: {}".format(l, i, mu, E[l-1, i, mu]))
                        # print(E[l-1,i])

                        E_tli.append(E[l - 1, i, mu])
                    E_tli = np.asarray(E_tli)
                    # print("min: {}".format(np.min(np.add(E_tli, q_min)) + C[i, j, mu]))
                    C[i, j, mu] = self.calculate_reinforcement_param(E, C, 0.002, i, mu, i, j)
                    E_1[i, j, mu] = np.min(E_tli) + C[i, j, mu]
        else:
            graph_i = self.create_graph_i(graph, node_i)

            # calculating the weights of the graph_i
            for node_k, node_l in graph_i.edges:
                k = node_k - 1
                l = node_l - 1
                Q_edge = E[k, i, 0] + E[l, i, 0] - np.min(np.add(E[k, i, 1:M+1], E[l, i, M+1:]))
                print("E[k, i, 0] = {}".format(E[k, i, 0]))
                print("E[l, i, 0] = {}".format(E[l, i, 0]))
                print("E[k, i] = {}".format(E[k, i, 1:M+1]))
                print("E[i, l] = {}".format(E[i, l, M+1:]))
                # print("min(np.add(E[k, i], E[i, l]) = {}".format(np.min(np.add(E[k, i], E[i, l]))))
                print("Q edge: {}".format(Q_edge))
                graph_i[node_k][node_l]["weight"] = Q_edge

            for node_j in graph.neighbors(node_i):
                j = node_j - 1
                q_min = self.calculate_q_min(graph, graph_i, node_i, node_j, E)

                for mu in range(np.shape(E)[2]):
                    L = self.get_delta_i_minus(graph, node_i, [node_j])
                    E_tli = []
                    for l in L:
                        # print("E[{},{},{}]: {}".format(l, i, mu, E[l-1, i, mu]))
                        # print(E[l-1,i])

                        E_tli.append(E[l - 1, i, mu])
                    E_tli = np.asarray(E_tli)
                    # print("min: {}".format(np.min(np.add(E_tli, q_min)) + C[i, j, mu]))
                    C[i,j,mu] = self.calculate_reinforcement_param(E, C, 0.2, t, mu, i, j)
                    E_1[i, j, mu] = np.min(np.add(E_tli, q_min)) + C[i, j, mu]
                    if mu == 0:
                        print("E_t[0]:{}".format(E[i, j, mu]))
                        print("E_t_1[0]:{}".format(np.min(np.add(E_tli, q_min)) + C[i, j, mu]))

    def calculate_q_min(self, graph, graph_i, i, j, E):

        q_min = []
        L = self.get_delta_i_minus(graph, i, [j])
        for l in L:
            graph_ijl = self.create_graph_ijl(graph, i, j, l)
            for e1, e2 in graph_ijl.edges:
                graph_ijl[e1][e2]["weight"] = graph_i[e1][e2]["weight"]

            # M = nx.matching.max_weight_matching(graph_ijl)
            # M = nx.matching.maximal_matching(graph_ijl)

            M = nx.max_weight_matching(graph_ijl)

            # M_i = nx.matching.max_weight_matching(graph_i)
            print("M: {}".format(M))
            print("weights: {}".format([graph_ijl[e1][e2]["weight"] for e1,e2 in graph_ijl.edges()]))
            # print("M_i: {}".format(M_i))
            print("degree i: {}".format(nx.degree(graph, i)))
            print("nodes ijl: {}".format(list(graph_ijl.nodes)))
            print("edges ijl: {}".format(list(graph_ijl.edges)))
            M_weights = []
            for e1, e2 in M:
                M_weights.append(graph_ijl[e1][e2]["weight"])
            # print("Mjl: {}".format(M_weights))
            try:
                M_jl = np.max(M_weights)
            except:
                # M_weights = []
                # for e1, e2 in M:
                #     M_weights.append(graph_i[e1][e2]["weight"])
                # M_jl = np.max(M_weights)
                M_jl = 0

            # M_jl = np.max(M_weights)

            # print("M_jl: {}".format(M_jl))

            set_k = self.get_delta_i_minus(graph, i, [j, l])
            e_sum = 0
            for _k in set_k:
                e_sum += E[_k-1, i-1, 0]
            q_min_jl = -M_jl + e_sum
            q_min.append(q_min_jl)
        return np.asarray(q_min)

    def create_graph_i(self, graph, i):
        graph_i_nodes = list(graph.neighbors(i))
        graph_i = nx.complete_graph(graph_i_nodes)
        return graph_i

    def create_graph_ijl(self, graph, i, j, l):
        graph_ijl_nodes = list(graph.neighbors(i))
        graph_ijl_nodes.remove(j)
        graph_ijl_nodes.remove(l)
        graph_ijl = nx.complete_graph(graph_ijl_nodes)
        return graph_ijl

    def calculate_reinforcement_param(self, E, C, p, t, mu, i, j):
        gamma_t = t*p
        h_ijt = E[i,j,mu]+ E[i, j,mu] - C[i,j, mu]
        c_t_1 = C[i,j, mu] + gamma_t* h_ijt
        return c_t_1

    def evaluate_min_cost(self, graph, edge, source, destination, paths):
        cost_list = []

        remove_congestion(graph, graph.graph["current paths"][source-1][destination-1])
        for path in paths:
            add_congestion(graph, path)
            cost_list.append(graph[edge[0]][edge[1]]["congestion"])
            remove_congestion(graph, path)
        min_cost_ind = np.argmin(cost_list)
        return cost_list[min_cost_ind], paths[min_cost_ind]

    def evaluate_q_min(self, graph, graph_ijl, edge):

        M = nx.matching.max_weight_matching(graph_ijl)



    def evaluate_edge_cost(self, graph, edge):
        pass




