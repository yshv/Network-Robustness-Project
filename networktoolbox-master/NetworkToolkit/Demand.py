import numpy as np
from .Topology import *

class Demand():
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = len(list(graph.nodes()))
        self.num_edges = len(list(graph.edges()))

    @staticmethod
    def create_skewed_demand(graph, alpha=0):
        """
        Method to create skewed demand according to gibbens 1993 paper. Takes a graph
        :param graph:
        :param alpha:
        :return:
        """
        N = len(graph)
        D = np.ones((N,N))
        np.fill_diagonal(D, 0)
        D = D*(1/(N**2-N))
        node_pairs = [(i, j) for i in range(N) for j in range(N) if (i > j)]
        indeces = list(range(len(node_pairs)))
        node_choices = np.random.choice(indeces, len(indeces), replace=False)
        node_pair_choices = np.array(node_pairs)[node_choices]
        for i in range(0, len(node_pair_choices)-1, 2):
            s = np.random.uniform(low=0, high=D[node_pair_choices[i][0], node_pair_choices[i][1]] * alpha)
            D[node_pair_choices[i + 1][0], node_pair_choices[i + 1][1]] += s
            D[node_pair_choices[i + 1][1], node_pair_choices[i + 1][0]] += s
            D[node_pair_choices[i][0], node_pair_choices[i][1]] -= s
            D[node_pair_choices[i][1], node_pair_choices[i][0]] -= s

        assert abs(np.sum(D)-1)<0.0001
        return D

    @staticmethod
    def create_locally_skewed_demand(graph, gamma=0):
        """
        Method to create locally skewed demand data.
        :param graph:
        :param gamma:
        :return:
        """
        top = Topology()
        node_data = graph.nodes.data()
        N = len(graph)

        distances = np.array([np.array([top.calculate_harvesine_distance(node_data[i+1]["Latitude"],
                                                        node_data[j+1]["Latitude"],
                                                        node_data[i+1]["Longitude"],
                                                        node_data[j+1]["Longitude"]) if i != j else 0 for i in range(N) ]) for j in range(N)])
        # np.fill_diagonal(sum_distances, 0)
        # sum_distances = sum_distances.sum(axis=0)
        traffic_local = np.array([[(1 / (distances[i, j] / distances[i].sum())) ** gamma if i > j else (
            (1 / (distances[i, j] / distances[j].sum())) ** gamma if i < j else 0) for i in range(N)] for j in
                                  range(N)])
        traffic_local = np.array([[traffic_local[i, j] / traffic_local.sum() for i in range(N)] for j in range(N)])
        assert abs(np.sum(traffic_local) - 1) < 0.0001
        return traffic_local



    def bandwidth_2_connections(self, SNR_matrix, T_b):
        """
        Method to convert a normalised bandwidth demand to a normalised connection demand.
        Where T_z^c = \frac{T_z^B/(1/log_2(1+SNR_z))}{\sum_z (T_z^c/(log_2(1+SNR_z))}
        :param SNR_matrix: SNR matrix with estimated SNR values
        :param T_b:        Normalised bandwidth demand matrix
        :return:           T_c^z Normalised connections demand matrix
        """
        a = np.multiply(T_b, (1 / np.log2(1 + SNR_matrix)))
        b = np.divide(T_b, np.log2(1 + SNR_matrix))
        np.fill_diagonal(a, 0)
        np.fill_diagonal(b, 0)
        T_c = np.divide(a, b.sum())
        assert np.abs(T_c.sum() - 1.0) < 0.0001
        return T_c

    def create_uniform_bandwidth_normalised(self):
        """

        :return:
        """
        N = len(self.graph)
        T_b = np.ones((N, N)) / (N ** 2 - N)
        np.fill_diagonal(T_b, 0)
        return T_b


    def create_uniform_connection_requests(self, set):
        """
        Method to create a uniform connections requests matrix.

        :param set: amount of connection requests per node pair
        :return: demand matrix
        """
        demand_matrix = np.ones((self.num_nodes, self.num_nodes)) * set
        return demand_matrix

    def create_uniform_bandwidth_requests(self, bandwidth):
        """
        Method to create a uniform bandwidth demand matrix.

        :param bandwidth: uniform bandwidth between each node pair
        :return: demand matrix
        """
        demand_matrix = np.ones((self.num_nodes, self.num_nodes)) * bandwidth
        return demand_matrix

    def construct_poisson_process_demand_matrix(self, demand_data):
        """
        Method to construct a demand matrix for poisson process demand (dynamic) time based.

        :param demand_data: dict of demand data - {"id":id, "sn":source node, "dn":destination node, "bandwidth":bandwidth requirement for connections,
                                                   "time": time, "establish":establish - 1 takedown -0, "index":index}
        :return: list of demand matrices
        """
        demand_matrix = np.zeros(
            (self.num_nodes, self.num_nodes))  # array to hold the active state of the demand matrix
        demand_matrix_list = np.zeros((np.int(len(demand_data["sn"])), self.num_nodes,
                                       self.num_nodes))  # array to hold all time states of the demand matrix
        for i in np.arange(len(demand_data["sn"])):
            if demand_data["establish"][i] == 0:
                demand_matrix[demand_data["sn"][i] - 1, demand_data["dn"][i] - 1] -= demand_data["bandwidth"][
                    i]  # if take down connection reduce the bandwidth
                demand_matrix[demand_data["dn"][i] - 1, demand_data["sn"][i] - 1] -= demand_data["bandwidth"][i]
            elif demand_data["establish"][i] == 1:
                demand_matrix[demand_data["sn"][i] - 1, demand_data["dn"][i] - 1] += demand_data["bandwidth"][
                    i]  # if establish connection increate the bandwidth
                demand_matrix[demand_data["dn"][i] - 1, demand_data["sn"][i] - 1] += demand_data["bandwidth"][
                    i]
            else:
                print("help¬!¬¬¬")
            demand_matrix_list[
                i] = demand_matrix  # add active state of demand matrix to the array of other active states
        return demand_matrix_list

    def create_poisson_process_demand(self, n_t, min_bandwidth, max_bandwidth, _lambda, _mu, timeout_limit, bandwidth_per_wavelength,
                                      demand_distribution=np.asarray([]),batch_time = 1):
        """
        Method to create poisson demand scheme for dynamic demand for time series analysis.

        :param n_t: number of traffic
        :param min_bandwidth: minimum bandwidth demand
        :param max_bandwidth: maximum bandwidth demand
        :param _lambda: mean for exponential distribution of start times
        :param _mu: mean for exponential distribution of end times of connections
        :param demand_distribution:the normalised distribution over nodes
        :param timeout_limit: waiting time limit for requests
        :param bandwidth_per_wavelength: bandwidth per wavelength connection
        :param batch_time: batch time for RWA
        :return: dict of demand data - {"id":id, "sn":source node, "dn":destination node, "bandwidth":bandwidth requirement for connections,
                                        "wavelength_num": wavelength number required, "time": time, 
                                        "establish":establish - 1 takedown -0, "index":index}
        """
        traffic_id = np.arange(1, 2 * n_t+1, 1)
        sn = np.zeros((2 * n_t,))
        dn = np.zeros((2 * n_t,))
        time = np.zeros((2 * n_t,))
        low = 1 # lower number of nodes
        high = self.num_nodes #higher number of nodes
        bandwidth = np.zeros((2 * n_t,)) # array for bandwidth requests
        establish = np.concatenate((np.ones((n_t,)), np.zeros((n_t,)))) # array of establish connections (1 if to establish, 0 to take down)
#         assert np.sum(demand_distribution) == 1
        timeout = np.ones((2 * n_t,))*timeout_limit # array for timeout
        connection = np.zeros((2 * n_t,)) # array for connection requests

        if len(demand_distribution) == 0: # source creation
            sn[:n_t] = np.random.choice(a=np.arange(low, high + 1), size=(len(sn[0:n_t]),))
        else:
            assert np.sum(demand_distribution) == 1
            sn[:n_t] = np.random.choice(a=np.arange(low, high + 1), size=(len(sn[0:n_t]),),
                                        p=np.sum(demand_distribution, axis=0))

        sn[n_t:2 * n_t] = sn[:n_t] # duplicating of source nodes
        if len(demand_distribution) == 0: # destination creation
            dn[:n_t] = np.random.choice(a=np.arange(low, high + 1), size=(len(sn[:n_t]),))
        else:
#             dn[:n_t] = np.random.choice(a=np.arange(low, high + 1), size=(n_t,), p=np.true_divide(demand_distribution[sn[:n_t]-1],np.sum(demand_distribution[sn[:n_t]-1], axis=0)))
            for i in np.arange(n_t):
                dn[i] = np.random.choice(a=np.arange(low, high + 1), size=1,
                      p=np.true_divide(demand_distribution[np.int(sn[i]-1)],np.sum(demand_distribution[np.int(sn[i]-1)], axis=0)))
        
        
        
        for i in np.arange(n_t):
            while sn[i] == dn[i]:
                if len(demand_distribution) == 0:
                    dn[i] = np.random.choice(a=np.arange(low, high + 1), size=1)
                else:
                    dn[i] = np.random.choice(a=np.arange(low, high + 1), size=1,
                          p=np.true_divide(demand_distribution[np.int(sn[i]-1)],np.sum(demand_distribution[np.int(sn[i]-1)], axis=0)))

        dn[n_t:2 * n_t] = dn[0:n_t] # destination multiplication for teardown connections
        bandwidth[:n_t] = np.random.uniform(min_bandwidth, max_bandwidth, size=(n_t,)) # bandwidth creation
        bandwidth[n_t:] = bandwidth[:n_t] # bandwidth duplication for teadown connections
        
        connection[:n_t] = np.ceil(bandwidth[:n_t]/bandwidth_per_wavelength)
        connection[n_t:] = connection[:n_t]
        

        for i in np.arange(1, n_t):
            time[i] = time[i - 1] + np.random.exponential(_lambda, size=1) # time creation for establish connections
        for i in np.arange(n_t , 2 * n_t):
            time[i] = time[i - n_t] + np.random.exponential(_mu, size=1) # time creation for teardown connections
            while time[i] - time[i - n_t] <= batch_time:
                time[i] = time[i - n_t] + np.random.exponential(_mu, size=1)                   
            
        index, time_sorted = np.argsort(time), np.sort(time) # sorting of times to give the sorted arrays
        demand_data = {"id": traffic_id[index], "sn": sn[index].astype(int), "dn": dn[index].astype(int),
                   "bandwidth": bandwidth[index],"wavelength_num":connection[index], "time": time[index], 
                   "establish": establish[index].astype(int),"index": index, 'timeout':timeout}
        return demand_data

    def generate_random_demand_distribution(self):
        """
        Method to generate a random normalalised symmetric distribution for a graph.

        :return: random normalised demand distribution in nd array of shape (nodes,nodes)
        """
        normal_demand_matrix = np.zeros((self.num_nodes, self.num_nodes))
        sequence = np.random.multinomial(500, np.ones((np.int((self.num_nodes**2-self.num_nodes)/2),))/(np.int((self.num_nodes**2-self.num_nodes)/2)), size=1)[0]*1e-3
        iter = np.nditer(sequence)
        for i in range(0, self.num_nodes):
            for j in range(0, self.num_nodes):
                if i == j:
                    continue
                elif i > j:
                    continue
                else:
                    normal_prob = next(iter)
                    normal_demand_matrix[i, j] = normal_prob
                    normal_demand_matrix[j, i] = normal_prob

        return normal_demand_matrix
if __name__ == "__main__":
    import NetworkToolkit as nt

    graph = nt.Tools.load_graph("ACMN_4_node")
    graph = nt.Tools.assign_congestion(graph)
    demand = Demand(graph)
    demand_distribution = demand.generate_random_demand_distribution()
    print(demand_distribution)
    demand_data = demand.create_poisson_process_demand(30000, 10, 100, 0.1, 1, demand_distribution=demand_distribution,batch_time = 1)
    demand_matrix_list = demand.construct_poisson_process_demand_matrix(demand_data)
    #demand_matrix_list = demand.create_uniform_bandwidth_requests(demand_data)


    #demand.generate_random_demand_distribution()