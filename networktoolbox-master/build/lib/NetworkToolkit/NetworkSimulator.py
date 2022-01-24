import NetworkToolkit as nt
from progress.bar import ShadyBar
import ray
from tqdm import tqdm
import numpy as np
import os
import socket
import sys

class NetworkSimulator:

    def __init__(self):
        pass

    def incremental_non_uniform_demand_simulation(self, graph,
                                                  traffic_matrix,
                                                  routing_func="FF-kSP",
                                                  channel_bandwidth=32e9,
                                                  connections=False,
                                                  start_bandwidth=0,
                                                  incremental_bandwidth=1e9,
                                                  e=0, max_count=10):
        os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
            socket.gethostname().split('.')[0])
        network = nt.Network.OpticalNetwork(graph,
                                            channel_bandwidth=channel_bandwidth, routing_func=routing_func)
        SNR_list = network.get_SNR_matrix()
        total_bandwidth = start_bandwidth
        if connections is not False:
            total_bandwidth = 1
        for i in range(max_count):
            rwa_assignment, total_bandwidth = self.incremental_routing(total_bandwidth,
                                                                       incremental_bandwidth,
                                                                       network,
                                                                       traffic_matrix,
                                                                       SNR_list=SNR_list,
                                                                       e=e,
                                                                       connections=connections)
            rwa_assignment, total_bandwidth, demand_matrix_connection, demand_matrix_bandwidth = self.reverse_incremental_routing(total_bandwidth,
                                                                               incremental_bandwidth,
                                                                               network,
                                                                               traffic_matrix,
                                                                               SNR_list=SNR_list,
                                                                               connections=connections,
                                                                               e=e)
            incremental_bandwidth /= 2

        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(rwa_assignment)
        network.physical_layer.add_non_linear_NSR_to_links()
        # print("rwa assignment: {}".format(rwa_assignment))
        capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)

        # capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
        data_dic = {"rwa": rwa_assignment, "capacity": capacity}
        return data_dic

    def incremental_uniform_connections_simulation(self, graph, routing_func="FF-kSP",
                                                   channel_bandwidth=32e9,
                                                   start_bandwidth=0,
                                                   incremental_bandwidth=1e9,
                                                   e=0, max_count=10,PLI=True):
        network = nt.Network.OpticalNetwork(graph,
                                            channel_bandwidth=channel_bandwidth, routing_func=routing_func)
        total_connections = 0
        traffic_matrix = np.ones((len(graph), len(graph))) / (len(graph) ** 2 - len(graph))
        for i in tqdm(range(max_count)):
            rwa_assignment, total_connections = self.incremental_routing(total_connections,
                                                                         incremental_bandwidth,
                                                                         network, traffic_matrix, e=e,
                                                                         connections=1)
            rwa_assignment, total_connections = self.reverse_incremental_routing(total_connections,
                                                                                 incremental_bandwidth,
                                                                                 network, traffic_matrix,
                                                                                 connections=1,
                                                                                 e=e)
            network.physical_layer.add_uniform_launch_power_to_links(network.channels)
            network.physical_layer.add_wavelengths_to_links(rwa_assignment)
            network.physical_layer.add_non_linear_NSR_to_links()
            network.physical_layer.add_congestion_to_links(rwa_assignment)
            # print("rwa assignment: {}".format(rwa_assignment))
            if PLI: capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
            else: capacity = network.physical_layer.get_lightpath_capacities_no_PLI(rwa_assignment)
            data_dic = {"rwa": rwa_assignment, "capacity": capacity, "demand matrix":self.demand_matrix_connection }
            return data_dic

    def incremental_uniform_demand_simulation(self, graph, routing_func="FF-kSP",
                                              channel_bandwidth=32e9,
                                              start_bandwidth=0,
                                              incremental_bandwidth=1e9,
                                              e=0, max_count=10, k=1):
        """
        Method to simulate incremental uniform demand routing simulations.

        :param graph:                   Graph to use for optical network simulation - nx.Graph()
        :param channels:                Amount of channels in ON - int
        :param channel_bandwidth:       Channel bandwidth - float
        :param start_bandwidth:         Start Demand bandwidth - float
        :param incremental_bandwidth:   Bandwidth to increment with each step - float
        :return:                        Dictionary with simulation results
        :rtype:                         dictionary
        """
        network = nt.Network.OpticalNetwork(graph,
                                            channel_bandwidth=channel_bandwidth, routing_func=routing_func)
        SNR_list = network.get_SNR_matrix()
        if routing_func == "ILP-max-throughput":
            k_SP = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, e=e, k=k)  # k shortest paths
            network.physical_layer.add_wavelengths_full_occupation(channels_full=network.channels)
            network.physical_layer.add_uniform_launch_power_to_links(network.channels)
            network.physical_layer.add_non_linear_NSR_to_links(channels_full=network.channels,
                                                               channel_bandwidth=network.channel_bandwidth)
            _SNR_list = network.physical_layer.get_SNR_k_SP(network.channels, k_SP)
            network.rwa.SNR_list = _SNR_list
        total_bandwidth = start_bandwidth
        traffic_matrix = np.ones((len(graph), len(graph))) / (len(graph) ** 2 - len(graph))
        for i in range(max_count):
            rwa_assignment, total_bandwidth = self.incremental_routing(
                total_bandwidth, incremental_bandwidth, network, traffic_matrix, SNR_list=SNR_list, e=e, k=k)
            rwa_assignment, total_bandwidth, demand_matrix_connection, demand_matrix_bandwidth = \
                self.reverse_incremental_routing(total_bandwidth,
                                                 incremental_bandwidth,
                                                 network,
                                                 traffic_matrix,
                                                 SNR_list=SNR_list,e=e, k=k)
            incremental_bandwidth /= 2

        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(rwa_assignment)
        network.physical_layer.add_non_linear_NSR_to_links()
        # print("rwa assignment: {}".format(rwa_assignment))
        capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)

        # capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
        # print(demand_matrix_bandwidth)
        # print(demand_matrix_connection)
        data_dic = {"rwa": rwa_assignment, "capacity": capacity, "SNR":SNR_list,
                    "demand matrix connections":demand_matrix_connection,
                    "demand matrix bandwidth":demand_matrix_bandwidth}
        return data_dic
        # print("capacity: {} Tbps".format(capacity*1e-12))
        # print("capacity average: {} Gbps".format(capacity_avg*1e-9))
        # update_data_with_id("Topology_Data", "real", graph[1], {"$set":{"FF_kSP
        # Capacity Total":capacity, "FF_kSP Capacity Average":capacity_avg}})

    def reverse_incremental_routing(self, total_bandwidth,
                                    incremental_bandwidth,
                                    network, traffic_matrix,
                                    SNR_list=None,
                                    connections=False, e=0, k=1):
        rwa_assignment = True

        while rwa_assignment == True:

            if not connections:
                total_bandwidth -= incremental_bandwidth
                demand_matrix_bandwidth = np.multiply(np.ones((len(network.graph), len(network.graph))),
                                                      traffic_matrix) * total_bandwidth
                # print(demand_matrix_bandwidth)
                # print(traffic_matrix)
                self.demand_matrix_bandwidth = demand_matrix_bandwidth
                demand_matrix_connection = network.convert_bandwidth_to_connection(
                    demand_matrix_bandwidth, SNR_list)
                self.demand_matrix_connection = demand_matrix_connection

            elif connections is not False:
                total_bandwidth -= connections
                demand_matrix_connection = np.floor(np.multiply(np.ones((len(network.graph), len(network.graph))),
                                                                traffic_matrix) * (total_bandwidth))
                self.demand_matrix_connection = demand_matrix_connection
            rwa_assignment = network.route(demand_matrix_connection, e=e, k=k)
        return rwa_assignment, total_bandwidth, demand_matrix_connection, demand_matrix_bandwidth

    def incremental_routing(self, total_bandwidth, incremental_bandwidth,
                            network, traffic_matrix, SNR_list=None, e=0, connections=False, k=1):
        rwa_assignment = False
        print(type(network))
        while True:

            # print(demand_matrix_bandwidth)
            # print(demand_matrix_bandwidth[0])
            # print(SNR_list[0])
            if not connections:
                total_bandwidth += incremental_bandwidth
                demand_matrix_bandwidth = np.multiply(np.ones((len(network.graph), len(network.graph))),
                                                      traffic_matrix) * total_bandwidth
                demand_matrix_connection = network.convert_bandwidth_to_connection(
                    demand_matrix_bandwidth, SNR_list.copy())
            elif connections is not False:
                total_bandwidth += connections
                demand_matrix_connection = np.floor(np.multiply(np.ones((len(network.graph), len(network.graph))),
                                                                traffic_matrix) * (total_bandwidth))
            # print("bandwidth: {}".format(total_bandwidth))
            # print(demand_matrix_connection)

            # print(demand_matrix_connection)
            # print(traffic_matrix)
            # print(demand_matrix_connection)
            # print(demand_matrix_connection)
            rwa_assignment = network.route(demand_matrix_connection, e=e, k=k)
            # print(rwa_assignment)
            if rwa_assignment == True:
                break

        return rwa_assignment, total_bandwidth

    def connection_matrix_routing_graph_list(self,
                                             routing_func="FF-kSP",
                                             channel_bandwidth=32e9,
                                             graph_data=None,
                                             traffic_matrix_tag=None,
                                             e=0, db_name=None,
                                             collection_name=None,
                                             start=None, stop=None):
        if start is not None:
            # print("start: {} stop: {}".format(start, stop))
            graph_data = graph_data[start:stop]
        topology_data = graph_data['topology data']
        traffic_matrix_data = graph_data[traffic_matrix_tag]
        _id_data = graph_data["_id"]
        for topology, _id, traffic_matrix in tqdm(zip(topology_data, _id_data, traffic_matrix_data),
                                                  desc="server: {} start: {} stop: {}".format(
                                                      socket.gethostname().split('.')[0], start, stop)):
            graph = nt.Tools.read_database_topology(topology)
            network = nt.Network.OpticalNetwork(graph,
                                                channel_bandwidth=channel_bandwidth, routing_func=routing_func)
            rwa_assignment = network.route(np.array(traffic_matrix), e=e)
            network.physical_layer.add_uniform_launch_power_to_links(network.channels)
            network.physical_layer.add_wavelengths_to_links(rwa_assignment)
            network.physical_layer.add_non_linear_NSR_to_links()
            capacities = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)

            rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
            node_pair_capacities = {str(key): value for key, value in capacities[2].items()}
            nt.Database.update_data_with_id(db_name, collection_name, _id,
                                            {"$set": {
                                                "{} {} Capacity".format(routing_func, traffic_matrix_tag): capacities[
                                                    0],
                                                "{} {} node pair capacities".format(routing_func, traffic_matrix_tag
                                                                                    ): node_pair_capacities,
                                                "{} {} RWA assignment".format(
                                                    routing_func, traffic_matrix_tag): rwa_assignment}})

    def incremental_non_uniform_demand_simulation_graph_list(self, graph_data=None,
                                                             routing_func="FF-kSP",
                                                             channel_bandwidth=32e9,
                                                             start_bandwidth=0,
                                                             incremental_bandwidth=1e9,
                                                             e=0, db_name=None,
                                                             connections=False,
                                                             traffic_matrix_tag=None,
                                                             collection_name=None,
                                                             start=None,
                                                             stop=None):
        if start is not None:
            # print("start: {} stop: {}".format(start, stop))
            graph_data = graph_data[start:stop]
        topology_data = graph_data['topology data']
        traffic_matrix_data = graph_data[traffic_matrix_tag]
        _id_data = graph_data["_id"]
        for topology, _id, traffic_matrix in tqdm(zip(topology_data, _id_data, traffic_matrix_data),
                                                  desc="server: {} start: {} stop: {}".format(
                                                      socket.gethostname().split('.')[0], start, stop)):
            graph = nt.Tools.read_database_topology(topology)
            data = self.incremental_non_uniform_demand_simulation(graph, traffic_matrix,
                                                                  routing_func=routing_func,
                                                                  channel_bandwidth=channel_bandwidth,
                                                                  start_bandwidth=start_bandwidth,
                                                                  incremental_bandwidth=incremental_bandwidth,
                                                                  connections=connections,
                                                                  e=e)
            rwa_assignment = data["rwa"]
            rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
            node_pair_capacities = {str(key): value for key, value in data["capacity"][2].items()}
            nt.Database.update_data_with_id(db_name, collection_name, _id,
                                            {"$set": {
                                                "{} {} Capacity".format(routing_func, traffic_matrix_tag): data[
                                                    "capacity"][0],
                                                "{} {} node pair capacities".format(routing_func, traffic_matrix_tag
                                                                                    ): node_pair_capacities,
                                                "{} {} RWA assignment".format(
                                                    routing_func, traffic_matrix_tag): rwa_assignment}})
            print("capacity: {}".format(data["capacity"][0]))

    def incremental_uniform_demand_simulation_graph_list(self, graph_list=None,
                                                         routing_func="FF-kSP",
                                                         channel_bandwidth=32e9,
                                                         start_bandwidth=0,
                                                         incremental_bandwidth=1e9,
                                                         e=0, k=1, db_name=None,
                                                         collection_name=None,
                                                         find_dic={},
                                                         descriptor="incremental "
                                                                    "bandwidth",
                                                         start=None,
                                                         stop=None):
        """
        Method to feed in graph_list and automate the incremental uniform demand
        simulations.


        :param routing_func:                Function to use for routing,
        check Network.py for details - string
        string
        :param channel_bandwidth:           Bandwidth of channels - float
        :param start_bandwidth:             Initial demand bandwidth to start with -
        float
        :param incremental_bandwidth:       Bandwidth to increment with - float
        :param e:                           Maximum MNH+e length paths to allow in
        k-SP - int
        :param db_name:                     Name of database to find graphs from -
        string
        :param collection_name:             Name of collection name to find graphs -
        string
        from - string
        :param find_dic:                    Dictionary containing database query - dict
        :param descriptor:                  Description for progress bar - string
        """

        if start is not None:
            # print("start: {} stop: {}".format(start, stop))
            graph_list = graph_list[start:stop]

        # print("starting {}".format(routing_func))

        for index, (graph, _id) in tqdm(enumerate(graph_list), desc="server: {} start: {} stop: {}".format(
                socket.gethostname().split('.')[0], start, stop)):
            data = self.incremental_uniform_demand_simulation(graph,
                                                              routing_func=routing_func,
                                                              channel_bandwidth=channel_bandwidth,
                                                              start_bandwidth=start_bandwidth,
                                                              incremental_bandwidth=incremental_bandwidth,
                                                              e=e,k=k)
            rwa_assignment = data["rwa"]
            rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
            node_pair_capacities = {str(key): value for key, value in data["capacity"][2].items()}
            nt.Database.update_data_with_id(db_name, collection_name, _id,
                                            {"$set": {
                                                "{} Capacity".format(routing_func): data[
                                                    "capacity"][0],
                                                "{} node pair capacities".format(routing_func): node_pair_capacities,
                                                "{} RWA assignment".format(
                                                    routing_func): rwa_assignment,
                                                "demand matrix connections":data["demand matrix connections"].tolist(),
                                                "demand matrix bandwidth":data["demand matrix bandwidth"].tolist()}})
            print("capacity: {}".format(data["capacity"][0]))

            # bar.next()

    def ILP_max_uniform_bandwidth(self, graph, **kwargs):

        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        rwa_assignment, objective_value = network.rwa.maximise_uniform_bandwidth_demand(**kwargs)
        if objective_value == 0:
            return None, None, None
        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(rwa_assignment)
        network.physical_layer.add_non_linear_NSR_to_links()
        max_capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
        return max_capacity, rwa_assignment, objective_value

    def ILP_chromatic_number(self, graph, max_time=1000):
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        rwa_assignment, min_wave = network.rwa.static_ILP(min_wave=True, max_time=max_time)
        return rwa_assignment, min_wave

    def ILP_chromatic_number_graph_list(self, descriptor="ILP chromatic number",
                                        collection_name=None, db_name=None,
                                        find_dic={}, start=None, stop=None,
                                        graph_list=None, max_time=1000):
        from tqdm import tqdm
        print(start)
        if start is not None:
            graph_list = graph_list[start:stop]
            print("graph list length: {}".format(len(graph_list)))

        for graph, _id in tqdm(graph_list):
            rwa_assignment, chromatic_number = self.ILP_chromatic_number(graph, max_time=max_time)
            if rwa_assignment is None or chromatic_number is None:
                nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
                                                {"$set": {"ILP chromatic": 156}})

                continue
            print("chromatic number: {}".format(chromatic_number))
            rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
            nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
                                            {"$set": {"ILP chromatic": chromatic_number,
                                                      "ILP chromatic RWA assignment": rwa_assignment}})

    def ILP_max_uniform_bandwidth_graph_list(self, descriptor="ILP uniform bandwidth",
                                             collection_name=None, db_name=None,T=0,
                                             find_dic={}, start=None, stop=None,write_dic=None,
                                             graph_list=None, scale=None,
                                             **kwargs):

        if graph_list is None:
            graph_list = nt.Database.read_topology_dataset_list(db_name, collection_name,
                                                                find_dic=find_dic)
        if socket.gethostname() == "MacBook-Pro":
            print("Laptop Run")
            os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
        else:
            os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
                socket.gethostname().split('.')[0])
        if start is not None or start == 0:
            graph_list = graph_list[start:stop]
        print(len(graph_list))
        bar = ShadyBar(descriptor, max=len(graph_list))
        for graph, _id in tqdm(graph_list, desc="server: {} start: {} stop: {}".format(
                socket.gethostname().split('.')[0], start, stop)):
            if T == 1:
                demand = nt.Demand.Demand(graph)
                T = demand.generate_random_demand_distribution()
                T = [T[i, j] for i in range(len(graph)) for j in range(len(graph)) if i < j]
                # print(T)
                print("length of T: \t{}".format(len(T)))
                print("sum of T: \t{}".format(sum(T)))
            max_capacity, rwa_assignment, objective_value = self.ILP_max_uniform_bandwidth(graph,T=T, _id=_id,**kwargs)
            print("max capacity: {}".format(max_capacity))
            if max_capacity is not None:
                rwa_assignment = {str(key): value for key, value in rwa_assignment.items()}
                node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
            if scale is not None:
                nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
                                                {"$set": {"ILP Capacity {}".format(scale): max_capacity[0],
                                                          "ILP capacity RWA assignment {}".format(scale):
                                                              rwa_assignment,
                                                          "ILP node pair capacities {}".format(scale):
                                                              node_pair_capacities,
                                                          "ILP objective value {}".format(scale): objective_value
                                                          }})
            else:
                if write_dic is not None:
                    write_name = write_dic
                    print(write_name)
                else:
                    write_name = ""
                nt.Database.update_data_with_id("Topology_Data", collection_name, _id,
                                                {"$set": {"ILP Capacity{}".format(write_name): max_capacity[0] if
                                                max_capacity is not None else 0,
                                                          "ILP capacity RWA assignment{}".format(write_name):
                                                              rwa_assignment if max_capacity is not None else 0,
                                                          "ILP node pair capacities{}".format(write_name):
                                                              node_pair_capacities if max_capacity is not None else 0,
                                                          "ILP objective value{}".format(write_name): objective_value
                                                          }})
            bar.next()
        bar.finish()