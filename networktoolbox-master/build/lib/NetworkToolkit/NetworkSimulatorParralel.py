# import NetworkToolkit as nt
# from progress.bar import ShadyBar
# import ray
# from tqdm import tqdm
# import numpy as np
# import os
# import socket
#
#
# # os.system('conda activate dgl')
# # print(socket.gethostname().split('.')[0])
# @ray.remote
# class NetworkSimulator:
#
#     def __init__(self):
#         pass
#
#     def incremental_non_uniform_demand_simulation(self, graph,
#                                                   traffic_matrix,
#                                                   routing_func="FF-kSP",
#                                                   channel_bandwidth=32e9,
#                                                   connections=False,
#                                                   start_bandwidth=0,
#                                                   incremental_bandwidth=1e9,
#                                                   e=0, max_count=10):
#         """
#
#         :param graph:
#         :param traffic_matrix:
#         :param routing_func:
#         :param channel_bandwidth:
#         :param connections:
#         :param start_bandwidth:
#         :param incremental_bandwidth:
#         :param e:
#         :param max_count:
#         :return:
#         """
#         network = nt.Network.OpticalNetwork(graph,
#                                             channel_bandwidth=channel_bandwidth, routing_func=routing_func)
#         SNR_list = network.get_SNR_matrix()
#         total_bandwidth = 0
#         if connections is not False:
#             total_bandwidth = 1
#         for i in range(max_count):
#             rwa_assignment, total_bandwidth = self.incremental_routing(
#                 total_bandwidth, incremental_bandwidth, SNR_list, network, traffic_matrix, e=e, connections=connections)
#             rwa_assignment, total_bandwidth = self.reverse_incremental_routing(total_bandwidth,
#                                                                                incremental_bandwidth,
#                                                                                SNR_list, network,
#                                                                                traffic_matrix,
#                                                                                connections=connections,
#                                                                                e=e)
#             incremental_bandwidth /= 2
#
#         network.physical_layer.add_uniform_launch_power_to_links(network.channels)
#         network.physical_layer.add_wavelengths_to_links(rwa_assignment)
#         network.physical_layer.add_non_linear_NSR_to_links()
#         # print("rwa assignment: {}".format(rwa_assignment))
#         capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
#
#         # capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
#         data_dic = {"rwa": rwa_assignment, "capacity": capacity}
#         return data_dic
#
#     def incremental_uniform_connections_simulation(self, graph, routing_func="FF-kSP",
#                                                    channel_bandwidth=32e9,
#                                                    start_bandwidth=0,
#                                                    k=1,
#                                                    incremental_bandwidth=1e9,
#                                                    e=0, max_count=10):
#         """
#
#         :param graph:
#         :param routing_func:
#         :param channel_bandwidth:
#         :param start_bandwidth:
#         :param incremental_bandwidth:
#         :param e:
#         :param max_count:
#         :return:
#         """
#         network = nt.Network.OpticalNetwork(graph,
#                                             channel_bandwidth=channel_bandwidth, routing_func=routing_func)
#         total_connections = 0
#         traffic_matrix = np.ones((len(graph), len(graph))) / len(graph) ** 2 - len(graph)
#         for i in range(max_count):
#             rwa_assignment, total_connections = self.incremental_routing(total_connections,
#                                                                          incremental_bandwidth,
#                                                                          network, traffic_matrix, e=e,k=k)
#             rwa_assignment, total_connections = self.reverse_incremental_routing(total_connections,
#                                                                                  incremental_bandwidth,
#                                                                                  network,traffic_matrix,
#                                                                                  connections=1,
#                                                                                  e=e,k=k)
#
#     def incremental_uniform_demand_simulation(self, graph, routing_func="FF-kSP",
#                                                 channel_bandwidth=32e9,
#                                                 start_bandwidth=0,
#                                                 incremental_bandwidth=1e9,
#                                                 e=0, k=1, max_count=10):
#         """
#         Method to simulate incremental uniform demand routing simulations.
#
#         :param graph:                   Graph to use for optical network simulation - nx.Graph()
#         :param channels:                Amount of channels in ON - int
#         :param channel_bandwidth:       Channel bandwidth - float
#         :param start_bandwidth:         Start Demand bandwidth - float
#         :param incremental_bandwidth:   Bandwidth to increment with each step - float
#         :return:                        Dictionary with simulation results
#         :rtype:                         dictionary
#         """
#         network = nt.Network.OpticalNetwork(graph,
#                                  channel_bandwidth=channel_bandwidth, routing_func=routing_func)
#         SNR_list = network.get_SNR_matrix()
#         if routing_func == "ILP-max-throughput":
#             k_SP = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, e=e)  # k shortest paths
#             network.physical_layer.add_wavelengths_full_occupation(channels_full=network.channels)
#             network.physical_layer.add_uniform_launch_power_to_links(network.channels)
#             network.physical_layer.add_non_linear_NSR_to_links(channels_full=network.channels,
#                                            channel_bandwidth=network.channel_bandwidth)
#             _SNR_list = network.physical_layer.get_SNR_k_SP(network.channels, k_SP)
#             network.rwa.SNR_list = _SNR_list
#         total_bandwidth = 0
#         traffic_matrix = np.ones((len(graph),len(graph)))/len(graph)**2-len(graph)
#         for i in range(max_count):
#             rwa_assignment, total_bandwidth = self.incremental_routing(total_bandwidth,
#                                                                        incremental_bandwidth,
#                                                                        network, traffic_matrix,
#                                                                        SNR_list=SNR_list, e=e,k=k)
#             rwa_assignment, total_bandwidth = self.reverse_incremental_routing(total_bandwidth,
#                                                                                incremental_bandwidth,
#                                                                                network,
#                                                                                traffic_matrix,
#                                                                                SNR_list=SNR_list,
#                                                                                e=e,k=k)
#             incremental_bandwidth /= 2
#
#         network.physical_layer.add_uniform_launch_power_to_links(network.channels)
#         network.physical_layer.add_wavelengths_to_links(rwa_assignment)
#         network.physical_layer.add_non_linear_NSR_to_links()
#         capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
#         data_dic = {"rwa": rwa_assignment, "capacity": capacity}
#         return data_dic
#
#
#     def reverse_incremental_routing(self, total_bandwidth,
#                                     incremental_bandwidth,
#                                     network, traffic_matrix,
#                                     SNR_list=None,
#                                     connections=False, e=0,k=1):
#         """
#
#         :param total_bandwidth:
#         :param incremental_bandwidth:
#         :param network:
#         :param traffic_matrix:
#         :param SNR_list:
#         :param connections:
#         :param e:
#         :return:
#         """
#         rwa_assignment = True
#
#         while rwa_assignment == True:
#
#             if not connections:
#                 total_bandwidth -= incremental_bandwidth
#                 demand_matrix_bandwidth = np.multiply(np.ones((len(network.graph), len(network.graph))),
#                                                       traffic_matrix) * total_bandwidth
#                 demand_matrix_connection = network.convert_bandwidth_to_connection(
#                     demand_matrix_bandwidth, SNR_list)
#             elif connections is not False:
#                 total_bandwidth -= connections
#                 demand_matrix_connection = np.floor(np.multiply(np.ones((len(network.graph),len(network.graph))),
#                                                        traffic_matrix)*(total_bandwidth))
#             rwa_assignment = network.route(demand_matrix_connection, e=e, k=k)
#         return rwa_assignment, total_bandwidth
#
#     def incremental_routing(self, total_bandwidth, incremental_bandwidth,
#                             network, traffic_matrix,SNR_list=None,
#                             e=0, k=1, connections=False):
#         """
#
#         :param total_bandwidth:
#         :param incremental_bandwidth:
#         :param network:
#         :param traffic_matrix:
#         :param SNR_list:
#         :param e:
#         :param connections:
#         :return:
#         """
#         rwa_assignment = False
#
#         while True:
#             if not connections:
#                 total_bandwidth += incremental_bandwidth
#                 demand_matrix_bandwidth = np.multiply(np.ones((len(network.graph), len(network.graph))),
#                                                       traffic_matrix) * total_bandwidth
#                 demand_matrix_connection = network.convert_bandwidth_to_connection(
#                     demand_matrix_bandwidth, SNR_list.copy())
#             elif connections is not False:
#                 total_bandwidth += connections
#                 demand_matrix_connection = np.floor(np.multiply(np.ones((len(network.graph),len(network.graph))),
#                                                        traffic_matrix)*(total_bandwidth))
#
#             rwa_assignment = network.route(demand_matrix_connection, e=e, k=k)
#             if rwa_assignment == True:
#                 break
#
#         return rwa_assignment, total_bandwidth
#
#
#
#     def connection_matrix_routing_graph_list(self,
#                                              routing_func="FF-kSP",
#                                              channel_bandwidth=32e9,
#                                              graph_data=None,
#                                              traffic_matrix_tag=None,
#                                              e=0, db_name=None,
#                                              collection_name=None,
#                                              start=None, stop=None):
#         """
#
#         :param routing_func:
#         :param channel_bandwidth:
#         :param graph_data:
#         :param traffic_matrix_tag:
#         :param e:
#         :param db_name:
#         :param collection_name:
#         :param start:
#         :param stop:
#         :return:
#         """
#         if start is not None:
#             graph_data = graph_data[start:stop]
#         topology_data = graph_data['topology data']
#         traffic_matrix_data = graph_data[traffic_matrix_tag]
#         _id_data = graph_data["_id"]
#         for topology, _id, traffic_matrix in tqdm(zip(topology_data,_id_data,
#                                                       traffic_matrix_data),
#                                                   desc="server: {} start: {} stop: {}".format(socket.gethostname().split('.')[0],
#                                                                                               start,
#                                                                                               stop)):
#             graph = nt.Tools.read_database_topology(topology)
#             network = nt.Network.OpticalNetwork(graph,
#                                                 channel_bandwidth=channel_bandwidth,
#                                                 routing_func=routing_func)
#             rwa_assignment = network.route(np.array(traffic_matrix), e=e)
#             network.physical_layer.add_uniform_launch_power_to_links(network.channels)
#             network.physical_layer.add_wavelengths_to_links(rwa_assignment)
#             network.physical_layer.add_non_linear_NSR_to_links()
#             capacities = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
#
#             rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
#             node_pair_capacities = {str(key): value for key, value in capacities[2].items()}
#             nt.Database.update_data_with_id(db_name, collection_name, _id,
#                                             {"$set":
#                                                 {
#                                                 "{} {} Capacity".format(routing_func, traffic_matrix_tag): capacities[0],
#                                                 "{} {} node pair capacities".format(routing_func, traffic_matrix_tag): node_pair_capacities,
#                                                 "{} {} RWA assignment".format(routing_func, traffic_matrix_tag): rwa_assignment
#                                                 }
#                                             })
#
#     def incremental_non_uniform_demand_simulation_graph_list(self, graph_data=None,
#                                                              routing_func="FF-kSP",
#                                                              channel_bandwidth=32e9,
#                                                              start_bandwidth=0,
#                                                              incremental_bandwidth=1e9,
#                                                              e=0, db_name=None,
#                                                              connections=False,
#                                                              traffic_matrix_tag=None,
#                                                              collection_name=None,
#                                                              start=None,
#                                                              stop=None):
#         """
#
#         :param graph_data:
#         :param routing_func:
#         :param channel_bandwidth:
#         :param start_bandwidth:
#         :param incremental_bandwidth:
#         :param e:
#         :param db_name:
#         :param connections:
#         :param traffic_matrix_tag:
#         :param collection_name:
#         :param start:
#         :param stop:
#         :return:
#         """
#         if start is not None:
#             graph_data = graph_data[start:stop]
#         topology_data = graph_data['topology data']
#         traffic_matrix_data = graph_data[traffic_matrix_tag]
#         _id_data = graph_data["_id"]
#         for topology, _id, traffic_matrix in tqdm(zip(topology_data,_id_data, traffic_matrix_data),
#                                                   desc="server: {} start: {} stop: {}".format(
#                 socket.gethostname().split('.')[0], start, stop)):
#             graph = nt.Tools.read_database_topology(topology)
#             data = self.incremental_non_uniform_demand_simulation(graph, traffic_matrix,
#                                                               routing_func=routing_func,
#                                                               channel_bandwidth=channel_bandwidth,
#                                                               start_bandwidth=start_bandwidth,
#                                                               incremental_bandwidth=incremental_bandwidth,
#                                                               connections=connections,
#                                                               e=e)
#             rwa_assignment = data["rwa"]
#             rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
#             node_pair_capacities = {str(key): value for key, value in data["capacity"][2].items()}
#             nt.Database.update_data_with_id(db_name, collection_name, _id,
#                                             {"$set": {
#                                                 "{} {} Capacity".format(routing_func, traffic_matrix_tag): data[
#                                                     "capacity"][0],
#                                                 "{} {} node pair capacities".format(routing_func, traffic_matrix_tag
#                                                                                     ): node_pair_capacities,
#                                                 "{} {} RWA assignment".format(
#                                                     routing_func, traffic_matrix_tag): rwa_assignment}})
#             print("capacity: {}".format(data["capacity"][0]))
#
#
#     def incremental_uniform_demand_simulation_graph_list(self,graph_list=None,
#                                                          routing_func="FF-kSP",
#                                                          channel_bandwidth=32e9,
#                                                          start_bandwidth=0,
#                                                          incremental_bandwidth=1e9,
#                                                          e=0, k=1, db_name=None,
#                                                          collection_name=None,
#                                                          start=None,
#                                                          stop=None):
#         """
#         Method to feed in graph_list and automate the incremental uniform demand
#         simulations.
#         :param routing_func:                Function to use for routing, check Network.py for details - string
#         :param channel_bandwidth:           Bandwidth of channels - float
#         :param start_bandwidth:             Initial demand bandwidth to start with - float
#         :param incremental_bandwidth:       Bandwidth to increment with - float
#         :param e:                           Maximum MNH+e length paths to allow in k-SP - int
#         :param db_name:                     Name of database to find graphs from - string
#         :param collection_name:             Name of collection name to find graphs - string from - string
#         :param find_dic:                    Dictionary containing database query - dict
#         :param descriptor:                  Description for progress bar - string
#         """
#
#         if start is not None:
#             graph_list = graph_list[start:stop]
#
#
#         for index, (graph, _id) in tqdm(enumerate(graph_list), desc="server: {} start: {} stop: {}".format(
#                 socket.gethostname().split('.')[0],start, stop)):
#             data = self.incremental_uniform_demand_simulation(graph,
#                                                               routing_func=routing_func,
#                                                               channel_bandwidth=channel_bandwidth,
#                                                               start_bandwidth=start_bandwidth,
#                                                               incremental_bandwidth=incremental_bandwidth,
#                                                               e=e, k=k)
#             rwa_assignment = data["rwa"]
#             rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
#             node_pair_capacities={str(key): value for key, value in data["capacity"][2].items()}
#             nt.Database.update_data_with_id(db_name, collection_name, _id,
#                                             {"$set": {
#                                                 "{} Capacity".format(routing_func):data[
#                                                     "capacity"][0],"{} node pair capacities".format(routing_func):node_pair_capacities, "{} RWA assignment".format(
#                                                     routing_func):rwa_assignment}})
#             print("capacity: {}".format(data["capacity"][0]))
#
#
#     def ILP_max_uniform_bandwidth(self, graph, max_time=1000, e=0, shortest_paths_only=False,T=None):
#         """
#
#         :param graph:
#         :param max_time:
#         :param e:
#         :param shortest_paths_only:
#         :param T:
#         :return:
#         """
#         network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
#         rwa_assignment, objective_value = network.rwa.maximise_uniform_bandwidth_demand(max_time=max_time, e=e,
#                                                                                         shortest_paths_only=shortest_paths_only,
#                                                                                         T=T)
#         network.physical_layer.add_uniform_launch_power_to_links(network.channels)
#         network.physical_layer.add_wavelengths_to_links(rwa_assignment)
#         network.physical_layer.add_non_linear_NSR_to_links()
#         max_capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
#         return max_capacity, rwa_assignment, objective_value
#
#     def ILP_chromatic_number(self, graph, max_time=1000):
#         """
#
#         :param graph:
#         :param max_time:
#         :return:
#         """
#         network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
#         rwa_assignment, min_wave = network.rwa.static_ILP(min_wave=True, max_seconds=max_time)
#         return rwa_assignment, min_wave
#
#     def ILP_chromatic_number_graph_list(self, descriptor="ILP chromatic number",
#                                         collection_name=None, db_name=None,
#                                         find_dic={}, start=None, stop=None,
#                                         graph_list=None, max_time=1000):
#         """
#
#         :param descriptor:
#         :param collection_name:
#         :param db_name:
#         :param find_dic:
#         :param start:
#         :param stop:
#         :param graph_list:
#         :param max_time:
#         :return:
#         """
#         from tqdm import tqdm
#         print(start)
#         if start is not None:
#             graph_list = graph_list[start:stop]
#             print("graph list length: {}".format(len(graph_list)))
#
#         for graph, _id in tqdm(graph_list):
#             rwa_assignment, chromatic_number = self.ILP_chromatic_number(graph, max_time=max_time)
#             if rwa_assignment is None or chromatic_number is None:
#                 nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
#                                                 {"$set": {"ILP chromatic": 156}})
#
#                 continue
#             print("chromatic number: {}".format(chromatic_number))
#             rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
#             nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
#                                             {"$set": {"ILP chromatic":chromatic_number,
#                                                       "ILP chromatic RWA assignment":rwa_assignment}})
#
#
#     def ILP_max_uniform_bandwidth_graph_list(self, descriptor="ILP uniform bandwidth",
#                                              collection_name=None, db_name=None,
#                                              find_dic={}, start=None, stop=None, graph_list=None, max_time=1000, e=0,
#                                              shortest_paths_only=False,T=None, scale=None):
#         """
#
#         :param descriptor:
#         :param collection_name:
#         :param db_name:
#         :param find_dic:
#         :param start:
#         :param stop:
#         :param graph_list:
#         :param max_time:
#         :param e:
#         :param shortest_paths_only:
#         :param T:
#         :return:
#         """
#
#
#
#         if graph_list is None:
#             graph_list = nt.Database.read_topology_dataset_list(db_name, collection_name,
#                                                             find_dic=find_dic)
#         if socket.gethostname() == "MacBook-Pro":
#             print("Laptop Run")
#             os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
#         else:
#             os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(socket.gethostname().split('.')[0])
#         if start is not None or start==0:
#             graph_list = graph_list[start:stop]
#         print(len(graph_list))
#         bar = ShadyBar(descriptor, max=len(graph_list))
#         for graph, _id in tqdm(graph_list,desc="server: {} start: {} stop: {}".format(
#                 socket.gethostname().split('.')[0],start, stop)):
#             if T ==1:
#                 demand = nt.Demand.Demand(graph)
#                 T = demand.generate_random_demand_distribution()
#                 T = [T[i, j] for i in range(len(graph)) for j in range(len(graph)) if i < j]
#
#                 print("length of T: \t{}".format(len(T)))
#                 print("sum of T: \t{}".format(sum(T)))
#             max_capacity, rwa_assignment, objective_value = self.ILP_max_uniform_bandwidth(graph, max_time=max_time,
#                                                                                            e=e,shortest_paths_only=shortest_paths_only,T=T)
#             print("max capacity: {}".format(max_capacity))
#             rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
#             node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
#             if scale is not None:
#                 nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
#                                                 {"$set": {"ILP Capacity {}".format(scale): max_capacity[0],
#                                                           "ILP capacity RWA assignment {}".format(scale):
#                                                               rwa_assignment,
#                                                           "ILP node pair capacities {}".format(scale):
#                                                               node_pair_capacities,
#                                                           "ILP objective value {}".format(scale): objective_value
#                                                           }})
#             else:
#                 nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
#                                                 {"$set": {"ILP Capacity":max_capacity[0],
#                                                           "ILP capacity RWA assignment":rwa_assignment,
#                                                           "ILP node pair capacities":node_pair_capacities,
#                                                           "ILP objective value":objective_value
#                                                           }})
#             bar.next()
#         bar.finish()
#
#
#
# if __name__ == "__main__":
#     # Command Line Interface
#     import argparse
#     parser = argparse.ArgumentParser(description='Process some integers.')
#     parser.add_argument('-start', action='store', type=int)
#     parser.add_argument('-stop',action='store', type=int,default=None)
#     parser.add_argument('--data', action='store', type=str)
#     parser.add_argument('-dstart',action='store', type=int)
#     parser.add_argument('-dstop', action='store', type=int)
#     parser.add_argument('-dalpha', action='store', type=int)
#     parser.add_argument('-m', action='store', type=str)
#     parser.add_argument('--name', action='store', type=str)
#     parser.add_argument('-t', action='store', type=int, default=1000)
#     parser.add_argument('-cb', action='store', type=int, default=16e9)
#     parser.add_argument('-ib', action='store', type=int, default=50e9)
#     parser.add_argument('-w', action='store', type=int, default=5)
#     parser.add_argument('-e', action='store', type=int, default=0)
#     parser.add_argument('-fde', action="store", type=str)
#     parser.add_argument('-spo', action="store", type=int, default=0)
#     parser.add_argument('-tmt', action="store", type=str, default=None)
#     parser.add_argument('--incremental', action='store', type=int, default=0)
#     parser.add_argument('--con', action='store', type=int, default=False)
#     parser.add_argument('-nut', action='store', type=int, default=0)
#     parser.add_argument('-fd', nargs="+", default=None)
#
#     # Args to variables
#     args = parser.parse_args()
#     connections=vars(args)['con']
#     workers=vars(args)['w']
#     incremental = vars(args)['incremental']
#     traffic_matrix_tag = vars(args)['tmt']
#     find_dic_exists = vars(args)['fde']
#     _start = vars(args)['start']
#     _stop = vars(args)['stop']
#     data = vars(args)['data']
#     dstart = vars(args)['dstart']
#     dstop = vars(args)['dstop']
#     alpha = vars(args)['dalpha']
#     method = vars(args)['m']
#     time = vars(args)['t']
#     e = vars(args)['e']
#     spo = vars(args)['spo']
#     find_dic_list = vars(args)['fd']
#     non_uniform_traffic = vars(args)['nut']
#     incremental_bandwidth = vars(args)["ib"]
#     channel_bandwidth=vars(args)['cb']
#
#     # Some initial info on the data
#     print("data: {}".format(data))
#     print("start: {} stop: {}".format(_start, _stop))
#     print("method: {}".format(method))
#     ray.init(address='auto', redis_password='5241590000000000')
#
#
#     # Reading graphs into graph_lists
#     if data == "real":
#         print("real topology")
#         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real",
#                                                        find_dic={"name": "30-Node-ONDPBook-Topology_nodes"}, node_data=True)
#     elif alpha is not None and _start is not None:
#         # DEPRICATED
#         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "SBAG", find_dic={"alpha": alpha},
#                                                             node_data=True)[_start:_stop]
#         print("finished reading")
#     elif traffic_matrix_tag is not None:
#         graph_data = nt.Database.read_data_into_pandas("Topology_Data", data, find_dic={})[_start:_stop]
#     elif _start is not None:
#         # DEPRICATED - start:stop are given to simulator functions
#         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", data, find_dic={},
#                                                             node_data=True)[_start:_stop]
#     elif find_dic_exists is not None:
#         print("find dict exists - if value doesnt exist")
#         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", data, find_dic={find_dic_exists: {
#             "$exists": False}},node_data=True)
#     elif find_dic_list is not None:
#         print("find dict exists - standard find dic")
#         find_dic ={}
#         for ind,item in enumerate(find_dic_list):
#             if ind * 2 == len(find_dic_list): break
#             if find_dic_list[ind*2+1][0:2]=="-s":
#                 find_dic[find_dic_list[ind * 2]] = str(find_dic_list[ind * 2 + 1][2:])
#             elif find_dic_list[ind*2+1][0:2]=="-i":
#                 find_dic[find_dic_list[ind * 2]] = int(find_dic_list[ind * 2 + 1][2:])
#             elif find_dic_list[ind*2+1][0:2]=="-f":
#                 find_dic[find_dic_list[ind * 2]] = float(find_dic_list[ind * 2 + 1][2:])
#         print("features: {}".format(find_dic))
#         graph_list = nt.Database.read_topology_dataset_list("Topology_Data",
#                                                             data,
#                                                             find_dic=find_dic,
#                                                             node_data=True)
#     else:
#         print("reading whole collection")
#         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", data, find_dic={},
#                                                             node_data=True)
#
#
#     # Methods
#
#     if method == "ILP_cr":
#         simulators = [NetworkSimulator.remote() for i in range(workers)]
#         data_len = len(graph_list)
#         start_stop = nt.Tools.create_start_stop_list(data_len, workers)
#         print(start_stop)
#         results = ray.get([s.ILP_chromatic_number_graph_list.remote(
#             db_name="Topology_Data",
#             collection_name=data,
#             graph_list=graph_list,
#             max_time=time,
#             start=start_stop[ind], stop=start_stop[ind + 1])
#             for ind, s in enumerate(simulators)])
#
#     elif method == "ILP_th":
#         if spo==1:
#             shortest_paths_only= True
#         else:
#             shortest_paths_only=False
#         simulators = [NetworkSimulator.remote() for i in range(workers)]
#         data_len = len(graph_list)
#         start_stop = nt.Tools.create_start_stop_list(data_len, workers)
#         print(start_stop)
#
#         results = ray.get([s.ILP_max_uniform_bandwidth_graph_list.remote(
#                                                        db_name="Topology_Data",
#                                                        collection_name=data,
#                                                        graph_list=graph_list,
#                                                        max_time=time,
#                                                        start=start_stop[ind], stop=start_stop[ind+1], e=e,
#                                                        shortest_paths_only=shortest_paths_only,
#                                                        T=non_uniform_traffic) for ind, s in enumerate(simulators)])
#
#     else:
#         simulators = [NetworkSimulator.remote() for i in range(workers)]
#         if traffic_matrix_tag is None:
#             data_len = len(graph_list)
#         else:
#             data_len = len(graph_data)
#         start_stop = nt.Tools.create_start_stop_list(data_len, workers)
#         print(start_stop)
#
#         if traffic_matrix_tag is None:
#
#             results = ray.get([s.incremental_uniform_demand_simulation_graph_list.remote(
#                                                                        graph_list=graph_list,
#                                                                        routing_func=method,
#                                                                        e=e,
#                                                                        channel_bandwidth=channel_bandwidth,
#                                                                        incremental_bandwidth=incremental_bandwidth,
#                                                                        collection_name=data,
#                                                                        db_name="Topology_Data",
#                                                                        start=start_stop[ind],
#                                                                        stop=start_stop[ind+1]) for ind, s in enumerate(simulators)])
#         elif traffic_matrix_tag is not None and incremental ==0:
#
#             results = ray.get([s.connection_matrix_routing_graph_list.remote(traffic_matrix_tag=traffic_matrix_tag,
#                                                                       graph_data=graph_data,
#                                                                       routing_func=method,
#                                                                       e=e,
#                                                                       channel_bandwidth=channel_bandwidth,
#                                                                       collection_name=data,
#                                                                       db_name="Topology_Data",
#                                                                       start=start_stop[ind],
#                                                                       stop=start_stop[ind+1]) for ind, s in enumerate(simulators)])
#         elif traffic_matrix_tag is not None and incremental ==1:
#             results = ray.get([s.incremental_non_uniform_demand_simulation_graph_list.remote(
#                                                                             traffic_matrix_tag=traffic_matrix_tag,
#                                                                              graph_data=graph_data,
#                                                                              routing_func=method,
#                                                                              e=e,
#                                                                              incremental_bandwidth=incremental_bandwidth,
#                                                                              channel_bandwidth=channel_bandwidth,
#                                                                              collection_name=data,
#                                                                              db_name="Topology_Data",
#                                                                              start=start_stop[ind],
#                                                                              connections=connections,
#                                                                              stop=start_stop[ind + 1]) for ind, s in enumerate(simulators)])
#
#
