import NetworkToolkit as nt
import numpy as np
from progress.bar import Bar, ShadyBar
import pandas as pd
# import dask
# from dask.distributed import Client, progress
# import tensorflow as tf
# import matplotlib.pyplot as plt
import networkx as nx
import ast
from tqdm import tqdm
import ray

# @ray.remote

# @ray.remote
# class GraphProperties():
#     @staticmethod
#     def update_capacity_from_rwa_assignment(graph_data, collection_name):
#         ind = 0
#         for graph, _id in tqdm(zip(graph_data["topology data"].to_list(), graph_data["_id"].to_list())):
#             try:
#                 graph = nt.Tools.read_database_topology(graph)
#                 new_rwa = {}
#                 for key in graph_data["ILP capacity RWA assignment"].to_list()[ind]:
#                     new_rwa[ast.literal_eval(key)] = graph_data["ILP capacity RWA assignment"].to_list()[ind][key]
#
#                 network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
#                 network.physical_layer.add_uniform_launch_power_to_links(network.channels)
#                 network.physical_layer.add_wavelengths_to_links(new_rwa)
#                 network.physical_layer.add_non_linear_NSR_to_links()
#                 max_capacity = network.physical_layer.get_lightpath_capacities_PLI(new_rwa)
#                 # print(max_capacity)
#                 node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
#                 nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
#                                                 {"$set": {"ILP Capacity": max_capacity[0],
#                                                           "ILP node pair capacities":node_pair_capacities}})
#                 ind += 1
#             except:
#                 print("failed - pass")
#                 pass
#
#     @staticmethod
#     def update_Dmax_value(graph, collection, wavelength_num):
#         """
#         Method to update the Dmax value (the number of uniform traffic set which the network can accommodate) of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :param wavelength_num: Number of wavlength on each link
#         :return: None
#         """
#         Dmax_value = np.floor(wavelength_num*nt.Tools.get_h_brute_force(graph[0]))
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Dmax value": Dmax_value}})
#
#     @staticmethod
#     def update_limiting_cut_value(graph, collection):
#         """
#         Method to update the limiting cut value (ceil(1/h)) of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         limiting_cut_value = nt.Tools.get_limiting_cut(graph[0])[3]
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Limiting cut value": limiting_cut_value}})
#
#     @staticmethod
#     def update_shortest_path_cost(graph, collection):
#         """
#         Method to update the average and variance of shortest paths costs of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         shortest_path = nt.Routing.get_shortest_dijikstra_all(graph[0])
#         sp_cost = []
#         for path in shortest_path:
#             path_cost = 0
#             for i in range(len(path[2])-1):
#                 path_cost += graph[0][path[2][i]][path[2][i+1]]["weight"]
#             sp_cost.append(path_cost)
#         average_sp_cost = np.mean(sp_cost)
#         variance_sp_cost = np.var(sp_cost)
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"average_sp_cost": average_sp_cost}})
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"variance_sp_cost": variance_sp_cost}})
#
#     @staticmethod
#     def update_k_shortest_path_cost(graph, collection, k):
#         """
#         Method to update the average and variance of k shortest paths costs of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :param k: number of k
#         :return: None
#         """
#         ksp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph[0], k=10 ,weighted=True)
#         ksp_cost = []
#         for paths in ksp:
#             for path in paths[1]:
#                 path_cost = 0
#                 for i in range(len(path)-1):
#                     path_cost += graph[0][path[i]][path[i+1]]["weight"]
#                 ksp_cost.append(path_cost)
#         average_ksp_cost = np.mean(ksp_cost)
#         variance_ksp_cost = np.var(ksp_cost)
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"average_ksp_cost": average_ksp_cost}})
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"variance_ksp_cost": variance_ksp_cost}})
#
#     @staticmethod
#     def weighted_spectral_density(graph, N, bins, _range=(0.0, 2.0)):
#         L = nx.normalized_laplacian_matrix(graph)
#         e = np.real(np.linalg.eigvals(L.A))
#         e = e[np.where(e>1e-8)]
#         frequency, edges = np.histogram(e, range=_range, bins=bins, density=False)
#         #print(frequency.sum())
#         #print(edges)
#
#         frequency = [frequency[i]/frequency.sum() for i in range(len(frequency))]
#        #print(frequency)
#         wsd_list = []
#         for ind, elem in enumerate(frequency):
#             #print(edges[ind+1])
#             wsd = np.asarray([(1-edges[ind+1])**N*elem])
#             #print(wsd)
#             wsd_list.append(wsd[0])
#         return wsd_list
#
#     @staticmethod
#     def update_weighted_spectrum_distribution(graph, collection,N,bins):
#         """
#         Method to update the weighted graph spectrum (eignvalues of the normalized Laplacian) of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         WSD = nt.Data.GraphProperties.weighted_spectral_density(graph[0],N,bins)
#         key = [str(x) for x in range(1,bins+1)]
#         WSD_dict = dict(zip(key,WSD))
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Weighted Spectrum Distribution": WSD_dict}})
#
#     @staticmethod
#     def update_graph_spectrum(graph, collection):
#         """
#         Method to update the graph spectrum (eignvalues of the normalized Laplacian) of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         L = np.asmatrix((nx.normalized_laplacian_matrix(graph[0]).toarray()))
#         eig =np.linalg.eig(L)[0]
#         eig_sorted = sorted(np.real(eig), key=float)
#         key = [str(x) for x in range(1,len(eig)+1)]
#         eignvalue = dict(zip(key,eig_sorted))
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Graph Spectrum": eignvalue}})
#
#     @staticmethod
#     def update_throughput_non_uniform(graph, collection,demand):
#         """
#         Method to update the throughput under arbitrary demand matrix of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         T_non_uniform = get_max_throughput_non_uniform(graph[0], demand, channel_spacing = 32e9, B_ch = 156)
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"throughput_non_uniform": T_non_uniform}})
#
#     @staticmethod
#     def update_edge_betweenness(graph, collection):
#         """
#         Method to update the maximum edge betweenness centrality of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         edge_betweenness = nx.edge_betweenness_centrality(graph[0])
#         m = max(edge_betweenness.values())
#         #nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"edge_betweenness": edge_betweenness}})
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"max edge betweenness": m}})
#         #print(max(edge_betweenness.values()))
#
#     @staticmethod
#     def update_clustering_coefficient(graph, collection):
#         """
#         Method to update the clustering coefficient of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         clustering_coefficient = nx.transitivity(graph[0])
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"clustering coefficient": clustering_coefficient}})
#     @staticmethod
#     def update_data_graph_list(graph_list, collection):
#         print("updating data...")
#         from sklearn.preprocessing import MinMaxScaler
#         minmax_scaler = MinMaxScaler()
#         for graph, _id in tqdm(graph_list):
#             # degree_dict = dict(graph.degree())
#             # clustering_dict = nx.clustering(graph)
#             # traffic_dict = {node:{"traffic":1/len(graph)} for node in graph.nodes}
#             # edge_betweeness = nx.edge_betweenness_centrality(graph)
#             # edge_betweeness_dict = {edge:{"edge betweeness":edge_betweeness[edge]} for edge in graph.edges}
#             #
#             # nx.set_node_attributes(graph, degree_dict, 'degree')
#             # nx.set_node_attributes(graph, clustering_dict, 'clustering coefficient')
#             # nx.set_node_attributes(graph, traffic_dict)
#             # nx.set_edge_attributes(graph, edge_betweeness_dict)
#             # topology_data = nt.Tools.graph_to_database_topology(graph)
#             # node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()))
#             comm = nt.Tools.get_communicability_distance(graph)
#             node_comm = []
#             for row in comm:
#                 node_comm.append(sum(row))
#             avg_node_comm = np.array(node_comm) / (len(graph.nodes) - 1)
#
#             communicability_dict = dict(zip(graph.nodes(), avg_node_comm))
#
#             node_comm_array = np.array(avg_node_comm).reshape(-1, 1)
#             normalized_node_comm = minmax_scaler.fit_transform(node_comm_array)
#             normalized_communicability_dict = dict(zip(graph.nodes(), normalized_node_comm))
#             nx.set_node_attributes(graph, communicability_dict, 'communicability distance')
#             nx.set_node_attributes(graph, normalized_communicability_dict, 'normalized communicability distance')
#             nx.set_edge_attributes(graph, nx.edge_current_flow_betweenness_centrality(graph), "current betweeness")
#             nx.set_node_attributes(graph, nx.current_flow_betweenness_centrality(graph), "current betweeness")
#             nx.set_edge_attributes(graph, nx.edge_betweenness_centrality(graph), "edge betweenness")
#             nx.set_node_attributes(graph, nx.clustering(graph), "clustering coefficient")
#             N = len(graph)
#             for node in graph.nodes:
#                 graph.nodes[node]["traffic"] = 1 / N
#                 graph.nodes[node]["degree"] = graph.degree[node]
#             graph_data = nt.Tools.graph_to_database_topology(graph)
#             node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()))
#             #     print(_id)
#             nt.Database.update_data_with_id("Topology_Data", "ILP-Regression", _id,
#                                             {"$set": {"topology data": graph_data, "node data": node_data}})
#             # print(topology_data)
#             # print(node_data)
#             # for key in topology_data.keys():
#             #     assert type(key) == str
#             #     print(type(key))
#             #     for _key in topology_data[key].keys():
#             #         assert type(_key) == str
#             #         print(type(_key))
#             # for key in node_data.keys():
#             #     assert type(key) == str
#             # nt.Database.update_data_with_id("Topology_Data", collection, _id, {"$set": {"node data":node_data,
#             #                                                                             "topology data":topology_data}})
#
#     @staticmethod
#     def update_graph_list(graph_list, _func, collection, *args, **kwargs):
#         """
#         Method to apply function to a graph_list updating all the members in the graph list
#         :param graph_list:      Graph list to apply functions to - list(graph, _id)
#         :param _func:           Function to apply to graph list members
#         :param db_name:         Name of database to update - string
#         :param collection:      Name of collection - string
#         :return: None
#         """
#         for graph in graph_list:
#             _func(graph, collection, *args, **kwargs)
#
#     @staticmethod
#     def update_uniform_capacity(dataframe):
#         pass
#
#     @staticmethod
#     def update_worst_SNR_node_pair(dataframe, collection):
#         for index, graph_data in tqdm(dataframe.iterrows()):
#             _id = graph_data["id"]
#             graph = nt.Tools.read_database_topology(graph_data["topology data"])
#             network = nt.Network.OpticalNetwork(graph)
#             SNR_list = network.estimate_worst_case_SNR()
#             SNR_list = sorted(SNR_list, key=lambda x: x[2])
#             nt.Database.update_data_with_id("Topology_Data", collection, _id,
#                                             {"$set": {"worst case SNR":SNR_list[0][2]}})
#
#     @staticmethod
#     def update_m(graph, collection):
#         """
#         Method to update the lower congestion bound of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         m = 2 * nt.Tools.get_lower_congestion_bound(graph[0])
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {"m": m}})
#
#     @staticmethod
#     def update_spanning_tree(graph, collection):
#         """
#         Method to update the number of spanning trees in a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         S = float(nt.Tools.calculate_number_of_spanning_trees(graph[0]))
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {"S": S}})
#
#     @staticmethod
#     def update_algebraic_connectivity(graph, collection):
#         """
#         Method to calculate and update the algebraic connectivity in a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         algebraic_connectivity = nx.linalg.algebraic_connectivity(graph[0])
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {
#                                             "algebraic connectivity": algebraic_connectivity}})
#
#     @staticmethod
#     def update_node_variance(graph, collection):
#         """
#         Method to update the node variance in a graph instance.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         degree_sequence = sorted([d for n, d in graph[0].degree()], reverse=True)
#         degree_variance = np.var(np.asarray(degree_sequence))
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {"degree variance": degree_variance}})
#
#     @staticmethod
#     def update_mean_internodal_distance(graph, collection):
#         """
#         Method to update the mean internodal distance in a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         edges = graph[0].edges()
#         sum_distance = 0
#         for edge in edges:
#             sum_distance += graph[0][edge[0]][edge[1]]["weight"] * 80
#         avg_dist = sum_distance / len(list(graph[0].edges()))
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {"mean internodal distance": avg_dist}})
#
#     @staticmethod
#     def update_T(graph, collection):
#         """
#         Method to update max throughput of a graph - worst case limiting cut case - not always exact...
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         T = nt.Tools.get_max_throughput_worst_case_exact(graph[0])
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {"capacity": T}})
#
#     @staticmethod
#     def update_communicability_index(graph, collection):
#         """
#         Method to update the communicability index of a graph.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         comm = nt.Tools.get_communicability_index(graph[0])
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {"communicability distance": comm}})
#     @staticmethod
#     def normalise_capacity(db="Topology_Data", collection="ILP-Regression"):
#         graph_df = nt.Database.read_data_into_pandas(db,
#                                                      collection,
#                                                      find_dic={})
#         _id = graph_df["_id"].to_list()
#         capacity = graph_df["ILP Capacity"].to_numpy()
#         max_capacity = capacity.max()
#         min_capacity = capacity.min()
#         capacity = (capacity-capacity.min())/(capacity.max()-capacity.min())
#
#         for i in range(len(_id)):
#             nt.Database.update_data_with_id(db, collection, _id[i],
#                                             {"$set": {
#                                                 "normalised ILP Capacity": capacity[i],
#                                                 "max ILP Capacity":max_capacity,
#                                                 "min ILP Capacity":min_capacity}})
#
#     @staticmethod
#     def update_comm_traff_ind(graph, collection):
#         """
#         Method to update communicability traffic index.
#         :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
#         :param collection:  Collection to update values in - string
#         :return: None
#         """
#         num_nodes = len(graph[0])
#         TM = np.ones((num_nodes, num_nodes)) * (1 / (num_nodes * (num_nodes - 1)))
#         np.fill_diagonal(TM, 0)
#         comm_traff_ind = nt.Tools.get_communicability_traffic_index(graph[0], TM)
#         nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
#                                         {"$set": {
#                                             "communicability traffic index": comm_traff_ind}})
#
#     @staticmethod
#     def create_demand_data(db_name, collection, graph_list, requests=1000, min_demand=10, max_demand=100):
#         """
#         Method to create poisson demand data.
#         :param db_name:     Name of database to save data to - string
#         :param collection:  Name of collection to save data to - string
#         :param graph_list:  List of graphs to create demand data for - list(graph, _id)
#         :param requests:    Amount of requests - int
#         :param min_demand:  Minimum amount for request - float (Gbps)
#         :param max_demand:  Maximum amount for request - float (Gbps)
#         :return:            None
#         """
#
#         for _graph in graph_list:
#             network = nt.Network.OpticalNetwork(_graph[0])
#             DM = network.demand.generate_random_demand_distribution()
#             demand_data = network.demand.create_poisson_process_demand(requests, min_demand, max_demand, 0.1,
#                                                                        1, DM)
#             data_dic = {"demand id": demand_data["id"].tolist(),
#                         "sn": demand_data["sn"].tolist(),
#                         "dn": demand_data["dn"].tolist(),
#                         "bandwidth": demand_data["bandwidth"].tolist(),
#                         "time": demand_data["time"].tolist(),
#                         "establish": demand_data["establish"].tolist(),
#                         "index": demand_data["index"].tolist(), "graph id": _graph[1],
#                         "graph type": "ER"}
#             nt.Database.insert_data(db_name, collection, data_dic)@ray.remote

# @ray.remote
class GraphProperties():

    def update_capacity_from_rwa_assignment(self, graph_data, collection_name):
        ind = 0
        for graph, _id in tqdm(zip(graph_data["topology data"].to_list(), graph_data["_id"].to_list())):
            try:
                graph = nt.Tools.read_database_topology(graph)
                new_rwa = {}
                for key in graph_data["ILP capacity RWA assignment"].to_list()[ind]:
                    new_rwa[ast.literal_eval(key)] = graph_data["ILP capacity RWA assignment"].to_list()[ind][key]

                network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
                network.physical_layer.add_uniform_launch_power_to_links(network.channels)
                network.physical_layer.add_wavelengths_to_links(new_rwa)
                network.physical_layer.add_non_linear_NSR_to_links()
                max_capacity = network.physical_layer.get_lightpath_capacities_PLI(new_rwa)
                # print(max_capacity)
                node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
                nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
                                                {"$set": {"ILP Capacity": max_capacity[0],
                                                          "ILP node pair capacities":node_pair_capacities}})
                ind += 1
            except:
                print("failed - pass")
                pass


    def update_Dmax_value(self, graph, collection, wavelength_num):
        """
        Method to update the Dmax value (the number of uniform traffic set which the network can accommodate) of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :param wavelength_num: Number of wavlength on each link
        :return: None
        """
        Dmax_value = np.floor(wavelength_num*nt.Tools.get_h_brute_force(graph[0]))
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Dmax value": Dmax_value}})


    def update_limiting_cut_value(self, graph, collection):
        """
        Method to update the limiting cut value (ceil(1/h)) of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        limiting_cut_value = nt.Tools.get_limiting_cut(graph[0])[3]
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Limiting cut value": limiting_cut_value}})


    def update_shortest_path_cost(self, graph, collection):
        """
        Method to update the average and variance of shortest paths costs of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        shortest_path = nt.Routing.get_shortest_dijikstra_all(graph[0])
        sp_cost = []
        for path in shortest_path:
            path_cost = 0
            for i in range(len(path[2])-1):
                path_cost += graph[0][path[2][i]][path[2][i+1]]["weight"]
            sp_cost.append(path_cost)
        average_sp_cost = np.mean(sp_cost)
        variance_sp_cost = np.var(sp_cost)
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"average_sp_cost": average_sp_cost}})
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"variance_sp_cost": variance_sp_cost}})


    def update_k_shortest_path_cost(self, graph, collection, k):
        """
        Method to update the average and variance of k shortest paths costs of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :param k: number of k
        :return: None
        """
        ksp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph[0], k=10 ,weighted=True)
        ksp_cost = []
        for paths in ksp:
            for path in paths[1]:
                path_cost = 0
                for i in range(len(path)-1):
                    path_cost += graph[0][path[i]][path[i+1]]["weight"]
                ksp_cost.append(path_cost)
        average_ksp_cost = np.mean(ksp_cost)
        variance_ksp_cost = np.var(ksp_cost)
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"average_ksp_cost": average_ksp_cost}})
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"variance_ksp_cost": variance_ksp_cost}})


    def weighted_spectral_density(self, graph, N, bins, _range=(0.0, 2.0)):
        L = nx.normalized_laplacian_matrix(graph)
        e = np.real(np.linalg.eigvals(L.A))
        e = e[np.where(e>1e-8)]
        frequency, edges = np.histogram(e, range=_range, bins=bins, density=False)
        #print(frequency.sum())
        #print(edges)

        frequency = [frequency[i]/frequency.sum() for i in range(len(frequency))]
       #print(frequency)
        wsd_list = []
        for ind, elem in enumerate(frequency):
            #print(edges[ind+1])
            wsd = np.asarray([(1-edges[ind+1])**N*elem])
            #print(wsd)
            wsd_list.append(wsd[0])
        return wsd_list


    def update_weighted_spectrum_distribution(self, graph, collection,N=4,bins=30):
        """
        Method to update the weighted graph spectrum (eignvalues of the normalized Laplacian) of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        WSD = self.weighted_spectral_density(graph[0],N,bins)
        key = [str(x) for x in range(1,bins+1)]
        WSD_dict = dict(zip(key,WSD))
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Weighted Spectrum Distribution": WSD_dict}})


    def update_graph_spectrum(self, graph, collection):
        """
        Method to update the graph spectrum (eignvalues of the normalized Laplacian) of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        L = np.asmatrix((nx.normalized_laplacian_matrix(graph[0]).toarray()))
        eig =np.linalg.eig(L)[0]
        eig_sorted = sorted(np.real(eig), key=float)
        key = [str(x) for x in range(1,len(eig)+1)]
        eignvalue = dict(zip(key,eig_sorted))
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"Graph Spectrum": eignvalue}})


    def update_throughput_non_uniform(self, graph, collection,demand):
        """
        Method to update the throughput under arbitrary demand matrix of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        T_non_uniform = get_max_throughput_non_uniform(graph[0], demand, channel_spacing = 32e9, B_ch = 156)
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"throughput_non_uniform": T_non_uniform}})


    def update_edge_betweenness(self, graph, collection):
        """
        Method to update the maximum edge betweenness centrality of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        edge_betweenness = nx.edge_betweenness_centrality(graph[0])
        m = max(edge_betweenness.values())
        #nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"edge_betweenness": edge_betweenness}})
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"max edge betweenness": m}})
        #print(max(edge_betweenness.values()))


    def update_clustering_coefficient(self, graph, collection):
        """
        Method to update the clustering coefficient of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        clustering_coefficient = nx.transitivity(graph[0])
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],{"$set": {"clustering coefficient": clustering_coefficient}})

    def update_data_graph_list(self, graph_list, collection):
        print("updating data...")
        from sklearn.preprocessing import MinMaxScaler
        minmax_scaler = MinMaxScaler()
        for graph, _id in tqdm(graph_list):
            # degree_dict = dict(graph.degree())
            # clustering_dict = nx.clustering(graph)
            # traffic_dict = {node:{"traffic":1/len(graph)} for node in graph.nodes}
            # edge_betweeness = nx.edge_betweenness_centrality(graph)
            # edge_betweeness_dict = {edge:{"edge betweeness":edge_betweeness[edge]} for edge in graph.edges}
            #
            # nx.set_node_attributes(graph, degree_dict, 'degree')
            # nx.set_node_attributes(graph, clustering_dict, 'clustering coefficient')
            # nx.set_node_attributes(graph, traffic_dict)
            # nx.set_edge_attributes(graph, edge_betweeness_dict)
            # topology_data = nt.Tools.graph_to_database_topology(graph)
            # node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()))
            comm = nt.Tools.get_communicability_distance(graph)
            node_comm = []
            for row in comm:
                node_comm.append(sum(row))
            avg_node_comm = np.array(node_comm) / (len(graph.nodes) - 1)

            communicability_dict = dict(zip(graph.nodes(), avg_node_comm))

            node_comm_array = np.array(avg_node_comm).reshape(-1, 1)
            normalized_node_comm = minmax_scaler.fit_transform(node_comm_array)
            normalized_communicability_dict = dict(zip(graph.nodes(), normalized_node_comm))
            nx.set_node_attributes(graph, communicability_dict, 'communicability distance')
            nx.set_node_attributes(graph, normalized_communicability_dict, 'normalized communicability distance')
            nx.set_edge_attributes(graph, nx.edge_current_flow_betweenness_centrality(graph), "current betweeness")
            nx.set_node_attributes(graph, nx.current_flow_betweenness_centrality(graph), "current betweeness")
            nx.set_edge_attributes(graph, nx.edge_betweenness_centrality(graph), "edge betweenness")
            nx.set_node_attributes(graph, nx.clustering(graph), "clustering coefficient")
            N = len(graph)
            for node in graph.nodes:
                graph.nodes[node]["traffic"] = 1 / N
                graph.nodes[node]["degree"] = graph.degree[node]
            graph_data = nt.Tools.graph_to_database_topology(graph)
            node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()))
            #     print(_id)
            nt.Database.update_data_with_id("Topology_Data", "ILP-Regression", _id,
                                            {"$set": {"topology data": graph_data, "node data": node_data}})
            # print(topology_data)
            # print(node_data)
            # for key in topology_data.keys():
            #     assert type(key) == str
            #     print(type(key))
            #     for _key in topology_data[key].keys():
            #         assert type(_key) == str
            #         print(type(_key))
            # for key in node_data.keys():
            #     assert type(key) == str
            # nt.Database.update_data_with_id("Topology_Data", collection, _id, {"$set": {"node data":node_data,
            #                                                                             "topology data":topology_data}})


    def update_graph_list(self, graph_list, _func, collection, *args, **kwargs):
        """
        Method to apply function to a graph_list updating all the members in the graph list
        :param graph_list:      Graph list to apply functions to - list(graph, _id)
        :param _func:           Function to apply to graph list members
        :param db_name:         Name of database to update - string
        :param collection:      Name of collection - string
        :return: None
        """
        for graph in graph_list:
            _func(graph, collection, *args, **kwargs)


    def update_uniform_capacity(self, dataframe):
        pass


    def update_worst_SNR_node_pair(self, dataframe, collection):
        for index, graph_data in tqdm(dataframe.iterrows()):
            _id = graph_data["id"]
            graph = nt.Tools.read_database_topology(graph_data["topology data"])
            network = nt.Network.OpticalNetwork(graph)
            SNR_list = network.estimate_worst_case_SNR()
            SNR_list = sorted(SNR_list, key=lambda x: x[2])
            nt.Database.update_data_with_id("Topology_Data", collection, _id,
                                            {"$set": {"worst case SNR":SNR_list[0][2]}})


    def update_m(self, graph, collection):
        """
        Method to update the lower congestion bound of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        m = 2 * nt.Tools.get_lower_congestion_bound(graph[0])
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {"m": m}})


    def update_spanning_tree(self, graph, collection):
        """
        Method to update the number of spanning trees in a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        S = float(nt.Tools.calculate_number_of_spanning_trees(graph[0]))
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {"S": S}})


    def update_algebraic_connectivity(self, graph, collection):
        """
        Method to calculate and update the algebraic connectivity in a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        algebraic_connectivity = nx.linalg.algebraic_connectivity(graph[0])
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {
                                            "algebraic connectivity": algebraic_connectivity}})


    def update_node_variance(self, graph, collection):
        """
        Method to update the node variance in a graph instance.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        degree_sequence = sorted([d for n, d in graph[0].degree()], reverse=True)
        degree_variance = np.var(np.asarray(degree_sequence))
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {"degree variance": degree_variance}})


    def update_mean_internodal_distance(self, graph, collection):
        """
        Method to update the mean internodal distance in a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        edges = graph[0].edges()
        sum_distance = 0
        for edge in edges:
            sum_distance += graph[0][edge[0]][edge[1]]["weight"] * 80
        avg_dist = sum_distance / len(list(graph[0].edges()))
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {"mean internodal distance": avg_dist}})


    def update_T(self, graph, collection):
        """
        Method to update max throughput of a graph - worst case limiting cut case - not always exact...
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        T = nt.Tools.get_max_throughput_worst_case_exact(graph[0])
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {"capacity": T}})


    def update_communicability_index(self, graph, collection):
        """
        Method to update the communicability index of a graph.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        comm = nt.Tools.get_communicability_index(graph[0])
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {"communicability distance": comm}})

    def normalise_capacity(self, db="Topology_Data", collection="ILP-Regression"):
        graph_df = nt.Database.read_data_into_pandas(db,
                                                     collection,
                                                     find_dic={})
        _id = graph_df["_id"].to_list()
        capacity = graph_df["ILP Capacity"].to_numpy()
        max_capacity = capacity.max()
        min_capacity = capacity.min()
        capacity = (capacity-capacity.min())/(capacity.max()-capacity.min())

        for i in range(len(_id)):
            nt.Database.update_data_with_id(db, collection, _id[i],
                                            {"$set": {
                                                "normalised ILP Capacity": capacity[i],
                                                "max ILP Capacity":max_capacity,
                                                "min ILP Capacity":min_capacity}})


    def update_comm_traff_ind(self, graph, collection):
        """
        Method to update communicability traffic index.
        :param graph:       (graph, _id) tuple from a graph list to then update in a collection. - tuple
        :param collection:  Collection to update values in - string
        :return: None
        """
        num_nodes = len(graph[0])
        TM = np.ones((num_nodes, num_nodes)) * (1 / (num_nodes * (num_nodes - 1)))
        np.fill_diagonal(TM, 0)
        comm_traff_ind = nt.Tools.get_communicability_traffic_index(graph[0], TM)
        nt.Database.update_data_with_id("Topology_Data", collection, graph[1],
                                        {"$set": {
                                            "communicability traffic index": comm_traff_ind}})


    def create_demand_data(self,db_name, collection, graph_list, requests=1000, min_demand=10, max_demand=100):
        """
        Method to create poisson demand data.
        :param db_name:     Name of database to save data to - string
        :param collection:  Name of collection to save data to - string
        :param graph_list:  List of graphs to create demand data for - list(graph, _id)
        :param requests:    Amount of requests - int
        :param min_demand:  Minimum amount for request - float (Gbps)
        :param max_demand:  Maximum amount for request - float (Gbps)
        :return:            None
        """

        for _graph in graph_list:
            network = nt.Network.OpticalNetwork(_graph[0])
            DM = network.demand.generate_random_demand_distribution()
            demand_data = network.demand.create_poisson_process_demand(requests, min_demand, max_demand, 0.1,
                                                                       1, DM)
            data_dic = {"demand id": demand_data["id"].tolist(),
                        "sn": demand_data["sn"].tolist(),
                        "dn": demand_data["dn"].tolist(),
                        "bandwidth": demand_data["bandwidth"].tolist(),
                        "time": demand_data["time"].tolist(),
                        "establish": demand_data["establish"].tolist(),
                        "index": demand_data["index"].tolist(), "graph id": _graph[1],
                        "graph type": "ER"}
            nt.Database.insert_data(db_name, collection, data_dic)


    def log_real_topologies(self):
        """
        Method to log SND and CONUS topologies.
        """
        nt.Database.delete_collection("Topology_Data", "real")
        nt.Topology.real_topology_logger_csv("/home/uceeatz/Code/RealTopologies/CONUS")
        nt.Topology.sdn_lib_topology_logger("/home/uceeatz/Code/RealTopologies/SDN")



class GNNData():
    @staticmethod
    def create_demand_data():

        nt.Database.delete_collection("Demand_Data", "ER")
        # Test topologies
        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ER", {"nodes":8, "mean k": 3})[:100]
    
        for graph in graph_list:
            network = nt.Network.OpticalNetwork(graph[0])
            DM = network.demand.generate_random_demand_distribution()
            demand_data = network.demand.create_poisson_process_demand(1000, 100, 1000, 0.1, 1, DM)
            data_dic = {"demand id":demand_data["id"].tolist(), "sn": demand_data["sn"].tolist(), "dn": demand_data["dn"].tolist(), "bandwidth":demand_data["bandwidth"].tolist(), "time": demand_data["time"].tolist(), "establish": demand_data["establish"].tolist(), "index":demand_data["index"].tolist(), "graph id": graph[1], "graph type":"ER"}
            nt.Database.insert_data("Demand_Data", "ER", data_dic)
    
    @staticmethod
    def convert_bandwidth_connection(demand_matrices):
        connection_requests_matrices = []
        for item in demand_matrices:
            connection_requests = network.convert_bandwidth_to_connection(item)
            connection_requests_matrices.append(connection_requests.tolist())
            bar.next()
        nt.Database.update_data_with_id("Demand_Data", "ER", df["_id"],
                                        {"$set": {"connection request matrices": connection_requests_matrices}})
    
    @staticmethod
    def create_ML_data():
        nt.Database.delete_collection("ML_Data", "ER")
        df = nt.Database.read_data_into_pandas("Demand_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("ML Data", max=100 * 2000)
        for i in range(len(df)):
            graph_id = df_dict["graph id"][i]
            graph = nt.Database.read_topology_dataset_list("Topology_Data", "ER", {"_id": graph_id})[0][0]
            network = nt.Network.OpticalNetwork(graph)
            demand_data = {"id": df_dict["demand id"][i], "sn": df_dict["sn"][i], "dn": df_dict["dn"][i],
                               "bandwidth": df_dict["bandwidth"][i], "time": df_dict["time"][i], "establish": df_dict["establish"][i],
                               "index": df_dict["index"][i]}
            demand_matrices = network.demand.construct_poisson_process_demand_matrix(demand_data)
            SNR_matrix = network.get_SNR_matrix()
    
            for item in demand_matrices:
                connection_requests = network.convert_bandwidth_to_connection(item.copy(), SNR_matrix)
                data_dic = {"graph id": graph_id, "connection request matrices": connection_requests.tolist(), "bandwidth matrix":item.tolist()}
                nt.Database.insert_data("ML_Data", "ER", data_dic)
                bar.next()
    
        bar.finish()
    
    @staticmethod
    def create_RWA_solutions():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("RWA Data", max=100 * 2000)
        for i in range(len(df)):
            graph_id = df_dict["graph id"][i]
            graph = nt.Database.read_topology_dataset_list("Topology_Data", "ER", {"_id": graph_id})[0][0]
            network = nt.Network.OpticalNetwork(graph)
            connection_requests = np.asarray(df_dict["connection request matrices"][i])
            network.rwa.FF_kSP(connection_requests, e=3)
    
            paths = network.rwa.get_paths_with_traffic(connection_requests, e=3)
    
    
            RWA_code = encode_RWA_path_state(network.rwa.wavelengths, paths)
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"RWA code": RWA_code[0], "wavelength code": RWA_code[1].tolist()}})
            bar.next()
            #print(network.rwa.wavelengths)
        bar.finish()
    
    @staticmethod
    def create_path_bandwidth_data():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("Path Bandwidth Data", max=100*2000)
        for i in range (len(df)):
            per_path_banwdith_matrix = np.asarray(df_dict["bandwidth matrix"][i])/np.asarray(df_dict["connection request matrices"][i])
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"per path bandwidth matrix": per_path_banwdith_matrix.tolist()}})
            bar.next()
        bar.finish()
    
    @staticmethod
    def copy_topology_to_ML_data():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("copy_topology_to_ML_data", max=100 * 2000)
        for i in range (len(df)):
            graph_id = df_dict["graph id"][i]
            graph = pd.DataFrame(list(nt.Database.read_data("Topology_Data", "ER", {"_id":graph_id})))["topology data"].to_list()
    
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"topology data": graph}})
            bar.next()
        bar.finish()
    
    @staticmethod
    def create_path_lists_1():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("create_path_lists", max=100 * 2000)
        for i in range (len(df)):
    
            graph= nt.Tools.read_database_topology(df["topology data"][i][0])
            #assert len(graph) <= 8
            network = nt.Network.OpticalNetwork(graph)
            connection_requests = np.asarray(df_dict["connection request matrices"][i])
            paths = network.rwa.get_paths_with_traffic(connection_requests, e=3)
            #paths = nt.Tools.convert_paths_to_SP(list(graph.edges), paths)
            #paths = nt.Tools.find_path_edges_set_all_to_all(graph)
            #graph = nt.Database.read_data("Topology_Data", "ER", {"_id":graph_id})["topology data"]
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"paths list": paths}})
            bar.next()
        bar.finish()
    
    @staticmethod
    def create_path_lists_2():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("create_path_lists", max=100 * 2000)
        for i in range (len(df)):
    
            graph= nt.Tools.read_database_topology(df["topology data"][i][0])
            assert len(graph) <= 8
            network = nt.Network.OpticalNetwork(graph)
            connection_requests = np.asarray(df_dict["connection request matrices"][i])
            paths = network.rwa.get_paths_with_traffic(connection_requests, e=3)
            paths = nt.Tools.convert_paths_to_SP(list(graph.edges), paths)
            #paths = nt.Tools.find_path_edges_set_all_to_all(graph)
            #graph = nt.Database.read_data("Topology_Data", "ER", {"_id":graph_id})["topology data"]
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"paths list": paths}})
            bar.next()
        bar.finish()
    
    @staticmethod
    def create_message_indeces():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("create_message_indeces", max=100 * 2000)
        for i in range(len(df)):
            graph = nt.Tools.read_database_topology(df["topology data"][i][0])
           # assert len(graph) <= 8
            network = nt.Network.OpticalNetwork(graph)
            connection_requests = np.asarray(df_dict["connection request matrices"][i])
            paths = network.rwa.get_paths_with_traffic(connection_requests, e=3)
            SP = nt.Tools.convert_paths_to_SP(list(graph.edges), paths)
            message_indeces = nt.Tools.find_indeces_paths_edges(list(graph.edges), SP)
            # paths = nt.Tools.find_path_edges_set_all_to_all(graph)
            # graph = nt.Database.read_data("Topology_Data", "ER", {"_id":graph_id})["topology data"]
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"message indeces": message_indeces, "paths list":SP}})
            bar.next()
        bar.finish()
    
    
    @staticmethod
    def create_bandwidth_features():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("create_bandwidth_features", max=100 * 2000)
        for i in range(len(df)):
            paths = df_dict["paths list"][i]
            per_path_bandwidth_matrix = df_dict["per path bandwidth matrix"][i]
            bandwidth_feature_paths = []
            for path in paths:
                #print(len(per_path_bandwidth_matrix))
                #print(path[0])
                #print(path[-1])
                bandwidth_feature_paths.append(per_path_bandwidth_matrix[path[0]-1][path[-1]-1])
    
    
            #print(len(per_path_bandwidth_matrix))
    
            #print(len(bandwidth_feature_paths))
            #print(len(paths))
            assert len(bandwidth_feature_paths) == len(paths)
    
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"bandwidth feature paths": bandwidth_feature_paths}})
            bar.next()
    
    @staticmethod
    def create_edge_list():
        df = nt.Database.read_data_into_pandas("ML_Data", "ER", {})
        df_dict = df.to_dict()
        bar = ShadyBar("create_edge_list", max=100 * 2000)
        for i in range(len(df)):
            graph = nt.Tools.read_database_topology(df["topology data"][i][0])
            edges = list(graph.edges)
            edge_num = len(edges)
            x_l = np.ones((edge_num,))*5000
            nt.Database.update_data_with_id("ML_Data", "ER", df_dict["_id"][i],
                                            {"$set": {"edges list": edges, "edges":edge_num, "edge features":x_l.tolist()}})
            bar.next()
        bar.finish()
    
    
    @staticmethod
    def encode_RWA_path_state(wavelengths, paths):
        wavelength_num = np.zeros((len(paths,)))
        for wavelength in wavelengths.keys():
            for path in wavelengths[wavelength]:
                wavelength_num[paths.index(path)] = wavelength + 1
                paths[paths.index(path)] = 1
        for ind, item in enumerate(paths):
            if type(item) != int:
                paths[ind] = 0
        return paths, wavelength_num
    
    @staticmethod
    def create_ML_data_total():
        create_demand_data()
        create_ML_data()
        create_RWA_solutions()
        create_path_bandwidth_data()
        copy_topology_to_ML_data()
        create_path_lists_1()
        create_bandwidth_features()
        create_edge_list()
        create_path_lists_2()
        create_message_indeces()
    
    @staticmethod
    def validate_ML_data():
        pass
    
    @staticmethod
    def analyse_data_out(sample_size=2000, batch_size=32):
        import graph_NN
        with tf.device("/CPU:0"):
            bar = ShadyBar("create_edge_list", max=sample_size)
            mps = graph_NN.MessagePasser()
            mps.init_propagation()
            dataset = nt.Database.read_data_into_pandas("ML_Data", "ER", {}, max_count=sample_size)
            dataset_RWA = dataset["RWA code"].to_list()
            _dataset = graph_NN.create_dataset(dataset, _range=(0, sample_size))
            batched_dataset = _dataset.shuffle(sample_size).batch(batch_size)
            new_dataset = batched_dataset.take(int(sample_size/batch_size))
            data_logit = []
            data_sigmoid = []
            data_RWA = []
            for item in new_dataset:
                out_logit = tf.reshape(mps.propagate(item, T=8), (-1,1))
                out_sigmoid = tf.reshape(mps.propagate(item, T=8, training=False), (-1,1))
                # print(tf.reshape(item[6], (-1,1)))
                rwa_code = tf.reshape(item[6], (-1,1))
                data_logit += out_logit.numpy().tolist()
                data_sigmoid += out_sigmoid.numpy().tolist()
                data_RWA += rwa_code.numpy().tolist()
                nt.Database.insert_data("ML", "training data analysis", {"data logit":out_logit.numpy().tolist(), "data sigmoid":out_sigmoid.numpy().tolist(), "data RWA":rwa_code.numpy().tolist()})
                bar.next()
            bar.finish()
            # print(data_logit)
            # print(data_RWA)
            # data_RWA_class_1_mask=np.where(np.asarray(data_RWA)==0, 1, 0)
            # data_RWA_class_2_mask = data_RWA
            # # print(np.ma.masked_array(data_logit, mask=data_RWA_class_1_mask))
            # print(np.ma.masked_array(data_logit, mask=data_RWA_class_2_mask))
            # plt.hist(np.ma.masked_array(data_logit, mask=data_RWA_class_1_mask))
            # plt.hist(np.ma.masked_array(data_logit, mask=data_RWA_class_2_mask))
            # plt.show()
            # plt.savefig("dataplot.png",bins=1000)
    
            # plt.hist(data_RWA)
            # plt.show()
            # plt.savefig("RWAdataplot.png")
    
    
            return data_logit, data_sigmoid
@ray.remote
class Topology():
    @staticmethod
    def create_real_based_topologies_ARP(amount=3000, alpha=1,
                                                             alpha_range=None,
                                                             max_degree=9, collection_name="ARP"):
        """
        Method to create ARP real based graphs.
        :param amount:         Amount of graphs to re-create - int
        :param alpha:           Alpha to use for the ARP - float
        :param alpha_range:     If creating multiple alpha values give a list - list
        :param max_degree:      Maximum degree to obey
        :param collection_name: Collection name to check save and cross-reference uniqueness
        """
        if alpha_range:
            for alpha in alpha_range:
                try:
                    create_real_based_topologies_ARP(amount=amount,
                                                     alpha=alpha, max_degree=max_degree)
                except:
                    pass
        else:
            real_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                                 find_dic={
                                                                     "name": {"$ne": "brain"},
                                                                     "source": "SNDlib",
                                                                     "connectivity": {
                                                                         "$lt": 0.5}},
                                                                     node_data=True)
            real_graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                                  "$source$ == CONUS",
                                                                  find_dic={}, node_data=True)
            bar = ShadyBar("ARP alpha: {}".format(alpha), max=len(
                real_graph_list*amount))
            top = nt.Topology.Topology()
            alphas = [0, 1, 10, 20, 30]
            choice_prob = top.choice_prob(SBAG=True, alpha=alpha)
            for ind, (grid_graph, _id) in enumerate(real_graph_list):
                for i in range(amount):
                    graph = top.create_real_based_grid_graph(grid_graph, len(list(
                        grid_graph.edges)), database_name="Topology_Data",
                                                             collection_name=collection_name,
                                                             ARP=True,
                                                             sequential_adding=True,
                                                             alpha=alpha, undershoot=True,
                                                             remove_C1_C2_edges=True,
                                                             max_degree=max_degree)
                    nt.Database.insert_graph(graph, db_name="Topology_Data",
                                            collection_name=collection_name,
                                             node_data=True,
                                             alpha=alpha, max_degree=max_degree)
                    bar.next()
            bar.finish()
            
    @staticmethod
    def create_real_based_topologies_SBAG(amount=3000, alpha=1, max_degree=9, alpha_range=None, 
                                                                                collection_name="SBAG-a0"):
        """
        Method to create real based topologies using SBAG.
        :param amount:         Amount of graphs to re-create - int
        :param alpha:           Alpha to use for the SBAG - float
        :param alpha_range:     If creating multiple alpha values give a list - list
        :param max_degree:      Maximum degree to obey
        :param collection_name: Collection name to check save and cross-reference uniqueness
        """
        if alpha_range:
            for alpha in alpha_range:
                try:
                    create_real_based_topologies_SBAG(amount=amount,alpha=alpha, max_degree=max_degree)
                except:
                    pass
        
        else:
            real_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real",find_dic={
                                                                                "name": {"$ne": "brain"},
                                                                                "source": "SNDlib",
                                                                                "connectivity": {"$lt": 0.5}},
                                                                                node_data=True)
        
            real_graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "real", "$source$ == CONUS",
                                                                        find_dic={}, node_data=True)
        
            bar = ShadyBar("SBAG - alpha: {}".format(alpha), max=len(real_graph_list * amount))
        
            top = nt.Topology.Topology()
        
        
        
            choice_prob = top.choice_prob(SBAG=True, alpha=alpha)
        
            for ind, (grid_graph, _id) in enumerate(real_graph_list):
                for i in range(amount):
                    graph = top.create_real_based_grid_graph(grid_graph, len(list(grid_graph.edges)),
                                                                database_name="Topology_Data",
                                                                collection_name=collection_name,
                                                                sequential_adding=True,
                                                                alpha=alpha, SBAG=True,
                                                                undershoot=True, remove_C1_C2_edges=True,
                                                                max_degree=max_degree)
        
                    nt.Database.insert_graph(graph, db_name="Topology_Data",
                                                    collection_name=collection_name, node_data=True,
                                                    alpha=alpha, max_degree=max_degree)
    
                    bar.next()
        
        bar.finish()
    
    @staticmethod
    def create_real_based_topologies_waxman(amount=3000, alpha=1, beta=1, max_degree=9,collection_name="waxman"):
        """
        Method to create real based topologies using the waxman graphs.
        :param amnount:         Amount of graphs to re-create - int
        :param alpha:           Alpha to use for the waxman - float
        :param beta:            Beta to use for the waxman - float
        :param max_degree:      Maximum degree to obey
        :param collection_name: Collection name to check save and cross-reference uniqueness
        """
        real_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                             find_dic={
                                                                 "name": {"$ne": "brain"},
                                                                 "source": "SNDlib",
                                                                 "connectivity": {
                                                                     "$lt": 0.5}},
                                                                 node_data=True)
        real_graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                              "$source$ == CONUS",
                                                              find_dic={}, node_data=True)
        bar = ShadyBar("waxman", max=len(real_graph_list * amount))
        top = nt.Topology.Topology()
    
        choice_prob = top.choice_prob(SBAG=True, alpha=alpha)
        for ind, (grid_graph, _id) in enumerate(real_graph_list):
            for i in range(amount):
                graph = top.create_real_based_grid_graph(grid_graph, len(list(
                    grid_graph.edges)),
                                                         database_name="Topology_Data",
                                                         collection_name=collection_name,
                                                         sequential_adding=True,
                                                         alpha=alpha, beta=beta,
                                                         waxman_graph=True,
                                                         undershoot=True,
                                                         remove_C1_C2_edges=True,
                                                         max_degree=max_degree)
                nt.Database.insert_graph(graph, db_name="Topology_Data",
                                         collection_name=collection_name, node_data=True,
                                         alpha=alpha, beta=beta, max_degree=max_degree)
                bar.next()
        bar.finish()
    
    @staticmethod
    def create_real_based_topologies_random(amount=3000, collection_name="random_real"):
        """
        Method to create real based topologies using random uniform probabilities.
        :param amount:          Amount of graphs to re-create - int
        :param collection_name: Collection name to check save and cross-reference uniqueness
        """
        real_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                             find_dic={
                                                                 "name": {"$ne": "brain"},
                                                                 "source": "SNDlib",
                                                                 "connectivity": {
                                                                     "$lt": 0.5}},
                                                                 node_data=True)
        real_graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                              "$source$ == CONUS",
                                                              find_dic={}, node_data=True)
        bar = ShadyBar("random", max=len(real_graph_list * amount))
        top = nt.Topology.Topology()
    
    
        for ind, (grid_graph, _id) in enumerate(real_graph_list):
            for i in range(amount):
                graph = top.create_real_based_grid_graph(grid_graph, len(list(
                    grid_graph.edges)),
                                                         database_name="Topology_Data",
                                                         collection_name=collection_name,
                                                         sequential_adding=True,
                                                         random=True, undershoot=True,
                                                         remove_C1_C2_edges=True)
                nt.Database.insert_graph(graph, db_name="Topology_Data",
                                         collection_name=collection_name, node_data=True)
                bar.next()
        bar.finish()
    
    @staticmethod
    def create_real_based_compare_data(amount=50, alpha_range=[1, 10, 100], beta=1):
        """
        Method to create all types of data including, BA, SBAG, ARP, Waxman, Random.
        :param amount:      Amount of graphs to re-create - int
        :param alpha_range: Range of alphas to re-create - list
        :param beta:        Beta to use for the waxman creation - float
        """
        try:
            nt.Database.delete_collection("Topology_Data", "SBAG-a0")
            create_real_based_topologies_SBAG(amount=amount, alpha=0)
        except:
            pass
        try:
            create_real_based_topologies_SBAG(amount=amount, alpha_range=alpha_range)
        except:
            pass
        try:
            nt.Database.delete_collection("Topology_Data", "ARP")    
            create_real_based_topologies_ARP(amount=amount, alpha_range=alpha_range)
        except:
            pass
        try:
            nt.Database.delete_collection("Topology_Data", "waxman")
            create_real_based_topologies_waxman(amount=amount)
        except:
            pass
        try:
            nt.Database.delete_collection("Topology_Data", "random_real")
            create_real_based_topologies_random(amount=amount)
        except:
            pass


    @staticmethod
    def create_real_based_topologies_dask(amount=50,n_workers=5, n_threads=4, node_range=None,
                                                                    nodes=None,
                                                                    grid_graph=None,
                                                                    db_name=None,
                                                                    edge_len=None,
                                                                    collection_name=None,
                                                                    show_progress=True,
                                                                    _mean=(0, 0),
                                                                    _std=(3, 16),
                                                                    descriptor=None,
                                                                    **kwargs):
        """
        Generalised method to create topologies based on a scattering of nodes on a 2d longitudce and lattitude grid
        using multiprocessing.
        :param amount:              Amount of graphs to re-create - int
        :param node_range:          Range of node sizes to re-create - list
        :param nodes:               Nodes to re-create - either use nodes or node_range, if both the node_range is chosen - int
        :param db_name:             Name of database to cross-refence uniqueness and save
        :param collection_name:     Name of collection to cross-reference uniqueness and save
        :param show_progress:       Whether to show progress bbar
        :param _mean:               mean to use for the scatter nodes - check doc of Topology.py
        :param _std:                std to use for the scatter nodes - check doc of Topology.py
        :param descriptor:          descriptor for progress bar - string
        :param **kwargs:            parameters for the create_real_based_topologies(**kwargs)

        """
        def create_save_graph(db, collection, node_graph, edges, **kwargs):
            graph = top.create_real_based_grid_graph(node_graph, edges, database_name=db,
                                                     collection_name=collection,
                                                     sequential_adding=True,
                                                     undershoot=True,
                                                     remove_C1_C2_edges=True,
                                                     **kwargs)
            nt.Database.insert_graph(graph, db_name=db_name, collection_name=collection_name,
                                     node_data=True, **kwargs)
        connectivity_ratio = 3.269
        top = nt.Topology.Topology()


        tasks = []
        if node_range:
            connectivity_list = [connectivity_ratio * (1 / (node - 1)) for node in node_range]
            links = [np.ceil((connectivity_list[node] * (node_range[node] ** 2 - node_range[node])) / 2)
                     for node in range(len(node_range))]
            # print(connectivity_list)

            for ind, nodes in enumerate(node_range):
                for i in range(amount):
                    node_graph = nt.Topology.scatter_nodes(nodes, _mean=_mean, _std=_std)
                    a = dask.delayed(create_save_graph)
                    tasks.append(a(db_name, collection_name, node_graph, int(links[ind]), **kwargs))

        elif nodes:

            for i in range(amount):
                node_graph = nt.Topology.scatter_nodes(nodes, _mean=_mean, _std=_std)
                edges_len = connectivity_ratio * (1 / (nodes - 1))
                if edge_len is not None:
                    edges_len = edge_len
                else:
                    edges_len = int(np.ceil((edges_len * (nodes ** 2 - nodes)) / 2))
                a = dask.delayed(create_save_graph)
                tasks.append(a(db_name, collection_name, node_graph, edges_len, **kwargs))

        elif grid_graph:
            for i in range(amount):
                a = dask.delayed(create_save_graph)
                tasks.append(a(db_name, collection_name, grid_graph, len(list(grid_graph.edges())), **kwargs))
        return tasks


    @staticmethod
    def create_real_based_topologies(amount=50, node_range=None, nodes=None, grid_graph=None, db_name=None,
                                                                    collection_name=None,
                                                                    edges_len=None,
                                                                    show_progress=True,
                                                                    _mean=(0, 0),
                                                                    _std=(3, 16),
                                                                    descriptor=None,
                                                                    **kwargs):
        """
        Generalised method to create topologies based on a scattering of nodes on a 2d longitudce and lattitude grid.
        :param amount:              Amount of graphs to re-create - int
        :param node_range:          Range of node sizes to re-create - list
        :param nodes:               Nodes to re-create - either use nodes or node_range, if both the node_range is chosen - int
        :param db_name:             Name of database to cross-refence uniqueness and save
        :param collection_name:     Name of collection to cross-reference uniqueness and save
        :param show_progress:       Whether to show progress bbar
        :param _mean:               mean to use for the scatter nodes - check doc of Topology.py
        :param _std:                std to use for the scatter nodes - check doc of Topology.py
        :param descriptor:          descriptor for progress bar - string
        :param **kwargs:            parameters for the create_real_based_topologies(**kwargs)
        
        """
        connectivity_ratio = 3.269
        top = nt.Topology.Topology()

        if node_range:
            print("using node range")
            connectivity_list = [connectivity_ratio*(1/(node-1)) for node in node_range]
            links = [np.ceil((connectivity_list[node]*(node_range[node]**2-node_range[node]))/2)
                                                                     for node in range(len(node_range))] 
            # print(connectivity_list)
            # if show_progress:
            #     if descriptor:
            #         bar = ShadyBar(descriptor, max=len(node_range) * amount)
            #     else:
            #         bar = ShadyBar("create_real_based_topologies", max=len(node_range) * amount)
            for ind, nodes in enumerate(node_range):
                for i in tqdm(range(amount)):
                    node_graph = nt.Topology.scatter_nodes(nodes, _mean=_mean, _std=_std)
                    if edges_len is None:
                        edges_len = int(links[ind])

                    graph = top.create_real_based_grid_graph(node_graph, edges_len, database_name=db_name,
                                                        collection_name=collection_name,
                                                        sequential_adding=True,
                                                        undershoot=True,
                                                            remove_C1_C2_edges=True,
                                                        **kwargs)
                                                                        
                    nt.Database.insert_graph(graph, db_name=db_name, collection_name=collection_name,
                                                    node_data=True)
                    # if show_progress:
                    #     bar.next()
        elif nodes:
            print("using nodes")
            
            # if show_progress:
            #     bar = ShadyBar("create_real_based_topologies", max=amount)
            for i in tqdm(range(amount)):
                    node_graph = nt.Topology.scatter_nodes(nodes, _mean=_mean, _std=_std)
                    if edges_len is None:
                        edges_len = connectivity_ratio * (1 / (nodes - 1))
                        edges_len = int(np.ceil((edges_len*(nodes**2-nodes))/2))

                    graph = top.create_real_based_grid_graph(node_graph, edges_len, database_name=db_name,
                                                        collection_name=collection_name,
                                                        sequential_adding=True,
                                                        undershoot=True,
                                                        remove_C1_C2_edges=True,
                                                        **kwargs)
                                                                        
                    nt.Database.insert_graph(graph, db_name=db_name, collection_name=collection_name,
                                                    node_data=True)
                    # if show_progress:
                    #     bar.next()
        elif grid_graph is not None:
            print("using grid graphs...")
            for i in tqdm(range(amount)):
                graph = top.create_real_based_grid_graph(grid_graph, len(list(grid_graph.edges())),
                                                                         database_name=db_name,
                                                         collection_name=collection_name,
                                                         sequential_adding= True,
                                                         undershoot=True,
                                                         remove_C1_C2_edges=True,
                                                         **kwargs)
                nt.Database.insert_graph(graph, db_name=db_name, collection_name=collection_name,
                                         node_data=True, **kwargs)

        # if show_progress:
        #     bar.finish()
    @staticmethod
    def  optimise_parameter(args, graph_args, graph_list, graph_len=10):
        top = nt.Topology.Topology()
        if "SBAG" in graph_args or "ARP" in graph_args:
            args = {"alpha": args[0]}
        elif "waxman_graph" in graph_args:
            args = {"beta":args[0]}

            # bar = ShadyBar("alpha: {} - {}".format(args["alpha"], graph_args), max=len(graph_list) * graph_len)
            # bar = ShadyBar("alpha: {} beta: {} - {}".format(args["alpha"], args["beta"], graph_args),
            #                max=len(graph_list) *graph_len)
        cost=[]

        #     print("creating graphs")
        # bar = ShadyBar("alpha: {} - {}".format(args["alpha"], graph_args), max=len(graph_list)*graph_len)
        for i in tqdm(range(graph_len)):
            tasks = []
            results = []
            for grid_graph, _id in graph_list:
            #         print(_id)

                # graphs.append((top.create_real_based_grid_graph(grid_graph, grid_graph.number_of_edges(),
                #                                                 sequential_adding=True,
                #                                                 undershoot=True,
                #                                                 remove_C1_C2_edges=True,
                #                                                 **graph_args,
                #                                                 **args), 2))
                np.random.seed(i)
                # _method = dask.delayed(top.create_real_based_grid_graph)
                results.append(top.create_real_based_grid_graph(grid_graph, grid_graph.number_of_edges(),
                                                                sequential_adding=True,
                                                                undershoot=True,
                                                                remove_C1_C2_edges=True,
                                                                **graph_args,
                                                                **args))
                # tasks.append(_method(grid_graph, grid_graph.number_of_edges(),
                #                                                 sequential_adding=True,
                #                                                 undershoot=True,
                #                                                 remove_C1_C2_edges=True,
                #                                                 **graph_args,
                #                                                 **args))
                # bar.next()
            # results = dask.compute(*tasks)
            # print(results)
            graphs = [(graph, 2) for graph in results]
        # bar.finish()
            cost.append(nt.Tools.weighted_spectral_density_distance_graph_list(graph_list, graphs, 4, 40))
        # print("cost: {}".format(cost))
        # print("alpha:{}".format(args["alpha"]))
        print("args: {} cost: {}".format(args, np.mean(cost)))
        cost = np.mean(cost)
        print(graph_args)

        return cost


        
                                                        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create or process data')
    parser.add_argument('-p', action='store', type=str)
    parser.add_argument('-n', nargs='+', type=int)
    parser.add_argument('-e', action='store', type=int,default=None)
    parser.add_argument('-c', action='store', type=str)
    parser.add_argument('-db', action='store', type=str, default="Topology_Data")
    parser.add_argument('-a', action='store', type=int)
    parser.add_argument('-tm', action="store", type=str)
    parser.add_argument('-alpha', action='store', type=int)
    parser.add_argument('-w', action='store', type=int, default=1)
    parser.add_argument('-cr', action='store', type=int, default=0)
    parser.add_argument('--name', action='store', type=str)
    args = parser.parse_args()
    create_real = vars(args)['cr']
    name = vars(args)['name']
    process = vars(args)['p']
    nodes = vars(args)['n']
    edges = vars(args)['e']
    database = vars(args)['db']
    collection = vars(args)['c']
    amount = vars(args)['a']
    topology_method = vars(args)['tm']
    workers = vars(args)['w']
    alpha = vars(args)['alpha']
    print(args)
    ray.init(address='auto', redis_password='5241590000000000')
    print("starting process")
    if process == "topology":
        print("topology")
        if create_real ==0:
            if topology_method == "SBAG":
                TopologyObjects = [Topology.remote() for i in range(workers)]
                data_len = amount
                print(data_len)
                print(workers)
                increment = np.floor(data_len / workers)
                print(increment)
                start_stop = list(range(0, data_len, int(increment)))
                if len(start_stop) == workers:
                    start_stop.append(data_len)
                else:
                    start_stop[-1] = data_len
                print(start_stop)
                results = ray.get([to.create_real_based_topologies.remote(start_stop[i+1] -start_stop[i],
                                                                          node_range=nodes,
                                                                          edges_len=edges,
                                                                          db_name=database,
                                                                          collection_name=collection,
                                                                          SBAG=True,
                                                                          alpha=alpha,
                                                                          max_degree=30
                                                                          ) for i, to in enumerate(TopologyObjects)])
        if create_real == 1:
            if topology_method == "SBAG":
                if name is not None:
                    graphs = nt.Database.read_topology_dataset_list("Topology_Data", "real", find_dic={"name":name},
                                                                    node_data=True)
                else:
                    graphs = nt.Database.read_topology_dataset_list("Topology_Data", "real", find_dic={},
                                                                    node_data=True)
                TopologyObjects = [Topology.remote() for i in range(workers)]
                data_len = amount
                print(data_len)
                print(workers)
                increment = np.floor(data_len / workers)
                print(increment)
                start_stop = list(range(0, data_len, int(increment)))
                if len(start_stop) == workers:
                    start_stop.append(data_len)
                else:
                    start_stop[-1] = data_len
                print(start_stop)
                print("graph lengths: {}".format(len(graphs)))
                for graph, _id in graphs:
                    results = ray.get([to.create_real_based_topologies.remote(start_stop[i+1] -start_stop[i],
                                                                              grid_graph=graph,
                                                                              db_name=database,
                                                                              collection_name=collection,
                                                                              SBAG=True,
                                                                              alpha=alpha,
                                                                              max_degree=30
                                                                              ) for i, to in enumerate(TopologyObjects)])

            elif topology_method=="BA":
                if name is not None:
                    graphs = nt.Database.read_topology_dataset_list("Topology_Data", "real", find_dic={"name":name},
                                                                    node_data=True)
                else:
                    graphs = nt.Database.read_topology_dataset_list("Topology_Data", "real", find_dic={},
                                                                    node_data=True)
                TopologyObjects = [Topology.remote() for i in range(workers)]
                data_len = amount
                print(data_len)
                print(workers)
                increment = np.floor(data_len / workers)
                print(increment)
                start_stop = list(range(0, data_len, int(increment)))
                if len(start_stop) == workers:
                    start_stop.append(data_len)
                else:
                    start_stop[-1] = data_len
                print(start_stop)
                print("graph lengths: {}".format(len(graphs)))
                for graph, _id in graphs:
                    results = ray.get([to.create_real_based_topologies.remote(start_stop[i+1] -start_stop[i],
                                                                              grid_graph=graph,
                                                                              db_name=database,
                                                                              collection_name=collection,
                                                                              BA_pure=True,
                                                                              # alpha=alpha,
                                                                              max_degree=30
                                                                              ) for i, to in enumerate(TopologyObjects)])


    elif process == "update-capacity":
        print("update-capacity")
        graph_data = nt.Database.read_data_into_pandas(db_name=database, collection_name=collection, find_dic={})
        GraphPropertiesObjects = [GraphProperties.remote() for i in range(workers)]
        data_len = len(graph_data)
        # print(data_len)
        # print(workers)
        increment = np.floor(data_len / workers)
        # print(increment)
        start_stop = list(range(0, data_len, int(increment)))
        if len(start_stop) == workers:
            start_stop.append(data_len)
        else:
            start_stop[-1] = data_len
        print(start_stop)

        results = ray.get([gp.update_capacity_from_rwa_assignment.remote(graph_data[start_stop[ind]:start_stop[ind+1]],
                                                                      collection) for ind, gp in enumerate(GraphPropertiesObjects)])
    elif process == "update-data":
        print("update-data started")
        print("reading data...")
        graph_list = nt.Database.read_topology_dataset_list(db_name=database,collection_name=collection, find_dic={})
        GraphPropertiesObjects = [GraphProperties.remote() for i in range(workers)]
        data_len = len(graph_list)
        print("data read...")
        increment = np.floor(data_len / workers)
        start_stop = list(range(0, data_len, int(increment)))
        if len(start_stop) == workers:
            start_stop.append(data_len)
        else:
            start_stop[-1] = data_len
        print(start_stop)
        results = ray.get(
            [gp.update_data_graph_list.remote(graph_list[start_stop[ind]:start_stop[ind + 1]],
                                              collection) for ind, gp in
             enumerate(GraphPropertiesObjects)])





