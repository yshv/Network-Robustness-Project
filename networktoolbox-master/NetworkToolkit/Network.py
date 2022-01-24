# import NetworkToolkit.Topology as Topology
import NetworkToolkit.Routing.Router as Router
import NetworkToolkit.PhysicalLayer as PhysicalLayer
import NetworkToolkit.Routing as Routing
import NetworkToolkit as nt

import logging
import numpy as np
# print("Network version 1.0.3")
# import NetworkToolkit.Tools as Tools
# import NetworkToolkit.Demand as Demand

# TODO: Update the documentation - 20/05/2020
# TODO: Re-structure data structure to using graph attributes - 20/05/2020


class OpticalNetwork():
    """

    """

    def __init__(self, graph, B_o=5e12, channel_bandwidth = 32e9,
                 mimic_topology="nsf", routing_func="FF-kSP", fibre_num=1):
        """

        :param mimic_topology:
        """
        #super.__init__(super, graph)
        self.channel_bandwidth = channel_bandwidth
        self.channels = np.int(np.floor(B_o/channel_bandwidth))
        self.routing_channels = int(fibre_num*self.channels)

        # print("channels: {}".format(self.channels))
        #         # print("channels: {}".format(self.channel_bandwidth))
        self.graph = graph
        self.rwa = Router.RWA(self.graph, self.routing_channels, channel_bandwidth)
        self.physical_layer = PhysicalLayer.PhysicalLayer(self.graph, self.channels,
                                                          channel_bandwidth)
        self.demand = nt.Demand.Demand(self.graph)
        self.LA = []
        self.WA = []
        self.edge_num = len(list(graph))
        self.node_num = len(graph)
        if routing_func == "FF-kSP":
            self.route = self.rwa.FF_kSP
        elif routing_func == "kSP-FF":
            self.route = self.rwa.k_SP_FF_revised
        elif routing_func == "kSP-CA-FF":
            self.route = self.rwa.k_SP_CA_FF
        elif routing_func == "kSP-baroni-FF":
            self.route = self.rwa.k_SP_baroni_FF
        elif routing_func == "ILP-min-wave":
            self.route = self.rwa.minimise_wavelengths_used
        elif routing_func == "ILP-max-throughput":
            self.route = self.rwa.maximise_throughput
        elif routing_func == "ILP-min-congestion":
            self.route = self.rwa.minimise_congestion




    def create_dynamic_demand(self, n_t, min_bandwidth, max_bandwidth, _lambda, _mu):
        """
        Method to create a set of demand matrices with poisson process generated bandwidth demands, given a normalised demand distribution over nodes.
        It then uses worst case SNR esetimation to tell us how many connections of given bandwidth to setup and return the demand matrices in terms of their connection
        request demands.

        :param n_t: The amount of connections to establish (amount of traffic)
        :param min_bandwidth: min bandwdth demands (Gbps)
        :param max_bandwidth: max bandwidth demands (Gbps)
        :param _lambda: distribution mean for exponentional distribution to use to generate the time between connetions to be established
        :param _mu: distrinution mean for exponentional distribution to use to generate the time between establishing connections and tearing them down.
        :return: array of demand matrices for connection requests shape - (2*n_t, nodes, nodes)
        """
        demand_distribution = self.demand.generate_random_demand_distribution()
        demand_data = self.demand.create_poisson_process_demand(n_t, min_bandwidth, max_bandwidth, _lambda, _mu, demand_distribution)
        demand_matrix_list = self.demand.construct_poisson_process_demand_matrix(demand_data)
        # print(demand_matrix_list)
        SNR_list = self.estimate_worst_case_SNR()
        SNR_matrix = np.zeros((self.node_num, self.node_num))
        for item in SNR_list:
            source = item[0]
            destination = item[1]
            SNR = item[2]
            SNR_matrix[source-1, destination-1] = SNR
            SNR_matrix[destination-1, source-1] = SNR
        for k in np.arange(len(demand_data["sn"])):
            for i in np.arange(self.node_num):
                for j in np.arange(self.node_num):
                    if demand_matrix_list[k, i, j] > 1e-10:
                        demand_matrix_list[k, i, j] = np.ceil(demand_matrix_list[k, i, j]/SNR_matrix[i,j])
                    else: demand_matrix_list[k , i, j] = 0
        # print(demand_matrix_list)
        return demand_matrix_list
    def get_SNR_matrix(self):
        """

        """
        SNR_list = self.estimate_worst_case_SNR()
        SNR_matrix = np.zeros((self.node_num, self.node_num))
        for item in SNR_list:
            source = item[0]
            destination = item[1]
            SNR = item[2]
            SNR_matrix[source - 1, destination - 1] = SNR
            SNR_matrix[destination - 1, source - 1] = SNR
        return SNR_matrix

    def convert_bandwidth_to_connection(self, bandwidth_demand, SNR_matrix):
        """
        Method to convert bandwidth demands into WDM fixed grid connection requests with SNR worst case estimation.

        :param bandwidth_demand: TM with banwidth demands
        :return: TM with connection requests
        """


        for i in np.arange(self.node_num):
            for j in np.arange(self.node_num):
                if bandwidth_demand[i, j] > 1e-3:
                    bandwidth_demand[i, j] = np.ceil((bandwidth_demand[i, j]) / (self.channel_bandwidth*np.log2(1+SNR_matrix[i, j])))
                elif bandwidth_demand[i, j] < 0:
                    bandwidth_demand[i, j] = 0
                else:
                    bandwidth_demand[i, j] = 0

        return bandwidth_demand

    def route_bandwidth(self, incremental_bandwidth, e, i4):
        demand_matrix_bandwidth = \
            network.demand.create_uniform_bandwidth_requests(incremental_bandwidth * i)
        # print(demand_matrix_bandwidth[0])
        # print(SNR_list[0])
        demand_matrix_connection = network.convert_bandwidth_to_connection(
            demand_matrix_bandwidth, SNR_list.copy())
        print("bandwidth: {}".format(incremental_bandwidth * i))
        print(demand_matrix_connection)
        rwa_assignment = network.route(demand_matrix_connection, e=e)

    def estimate_worst_case_SNR(self):
        """
        Method that estimates the worst case SNR for each node pair with estimating full occupancy.

        :return: SNR_list - (s, d, SNR)
        """
        self.physical_layer.add_wavelengths_full_occupation(self.channels)
        self.physical_layer.add_uniform_launch_power_to_links(self.channels)
        self.physical_layer.add_non_linear_NSR_to_links()
        unique_paths = nt.Routing.get_shortest_dijikstra_all(self.graph)
        SNR_list = self.physical_layer.get_SNR_shortest_path_node_pair(self.channels, unique_paths)
        return SNR_list

if __name__ == "__main__":
    import NetworkToolkit as nt
    from progress.bar import ShadyBar

    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                        find_dic={"nodes":{"$lt": 16}})
    print("hello")
    bar = ShadyBar("kSP FF Progress", max=len(graph_list))
    
    for graph in graph_list:
        # print(graph)
        graph_copy = nt.Tools.assign_congestion(graph[0].copy())
        network = nt.Network.OpticalNetwork(graph_copy, channel_bandwidth=32e9)
        print("calculating SNR list")
        SNR_list = network.get_SNR_matrix()
        print("done...")
        blocked=False
        i=0
        bandwidth_step = 0.5
        while True:

            #print("i:{}".format(i))
            demand_matrix_bandwidth = network.demand.create_uniform_bandwidth_requests(bandwidth_step*i)
            # print(demand_matrix_bandwidth[0])
            # print(SNR_list[0])
            demand_matrix_connection = network.convert_bandwidth_to_connection(demand_matrix_bandwidth, SNR_list.copy())
            print(demand_matrix_connection)
            blocked = network.rwa.FF_kSP(demand_matrix_connection)
            print(blocked)
            i+=1
            if blocked == True:
                i -=1
                demand_matrix_bandwidth = network.demand.create_uniform_bandwidth_requests(bandwidth_step*(i-1))
                demand_matrix_connection=network.convert_bandwidth_to_connection(demand_matrix_bandwidth, SNR_list)
                #network.rwa.FF_kSP(demand_matrix_connection)
                #network.rwa.k_SP_FF(demand_matrix_connection)
                network.rwa.k_SP_CA_FF(demand_matrix_connection)




                break


        # network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        # network.physical_layer.add_wavelengths_to_links(network.rwa.wavelengths)
        # network.physical_layer.add_non_linear_NSR_to_links()
        # capacity = network.physical_layer.get_lightpath_capacities_PLI(network.rwa.wavelengths)
        capacity = len(list(graph[0].nodes))*(len(list(graph[0].nodes))-1)*bandwidth_step*(i-1)*1e9
        capacity_avg = bandwidth_step*(i-1)*1e9
        #print("capacity: {} Tbps".format(capacity*1e-12))
        #print("capacity average: {} Gbps".format(capacity_avg*1e-9))
        #nt.Database.update_data_with_id("Topology_Data", "real", graph[1], {"$set":{
        # "FF_kSP Capacity Total":capacity, "FF_kSP Capacity Average":capacity_avg}})
        bar.next()
    bar.finish()

    # Analysis of results
    # df = nt.Database.read_data_into_pandas("Topology_Data", "grid", {})
    # df_capacity = df["kSP_FF Capacity Total"].to_list()
    # df_capacity_avg = df["kSP_FF Capacity Average"].to_list()
    #
    # mean_df = df.groupby("scaling factor").mean()
    # scaling_factor = mean_df["scaling factor"].to_list()
    # plt.plot(scaling_factor, mean_df["kSP_FF Capacity Total"].to_list())
    # plt.savefig("")
""" for i in range(156):
    demand_matrix = network.demand.create_uniform_connection_requests(i)
    blocked = network.rwa.k_SP_FF(demand_matrix)
    if blocked:
        demand_matrix = network.demand.create_uniform_connection_requests(i-1)
        print(demand_matrix)
        network.rwa.k_SP_FF(demand_matrix)
        break

print(network.rwa.wavelengths)
network.physical_layer.add_uniform_launch_power_to_links(network.channels)
network.physical_layer.add_wavelengths_to_links(network.rwa.wavelengths)
network.physical_layer.add_non_linear_NSR_to_links()
print(network.physical_layer.get_lightpath_capacities_PLI(network.rwa.wavelengths))

"""






