import pickle
import json
import networkx as nx
import numpy as np
import pandas as pd
import ast
import NetworkToolkit.Network as Network
import NetworkToolkit as nt
from asyncio import Event
from typing import Tuple
from time import sleep
import ray
import random
import scipy
from ray.actor import ActorHandle
from tqdm import tqdm


def assert_python_types(write_dictionary):
    """
    Method to assert that all types in write dictionary are either in or float, not numpy arrays, so that mongodb does not
    complain.
    :param write_dictionary: The dictionary that is going to be written to mongodb.
    :return: None
    """
    for key, value in write_dictionary.items():
        # print(key)
        # print(type(value))
        if isinstance(value, np.int64):
            write_dictionary[key] = int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            write_dictionary[key] = float(value)
        elif isinstance(value, list):
            if isinstance(value[0], list):
                for idx_c, column in enumerate(write_dictionary[key]):
                    for idx_r, row in enumerate(column):
                        if type(row) == np.int64:
                            write_dictionary[key][idx_c][idx_r] = int(row)
                        elif type(row) == np.float64 or type(row) == np.float32:
                            write_dictionary[key][idx_c][idx_r] = float(row)
                        assert isinstance(write_dictionary[key][idx_c][idx_r], int) or isinstance(
                            write_dictionary[key][idx_c][idx_r], float), \
                            "Trying to write an invalid value in {}:{}".format(key, type(row))
            else:
                for idx, item in enumerate(write_dictionary[key]):
                    if type(item) == np.int64:
                        write_dictionary[key][idx] = int(item)
                    elif type(item) == np.float64 or type(item) == np.float32:
                        write_dictionary[key][idx] = float(item)
                    assert isinstance(write_dictionary[key][idx], int) or isinstance(write_dictionary[key][idx], float), \
                        "Trying to write an invalid value in {}:{}".format(key, type(item))
        elif isinstance(value, np.ndarray):
            value = value.tolist()
            write_dictionary[key] = value
            assert_python_types(write_dictionary)


def total_fibre_length(graph):
    """
    Method to calculate the total fibre length for a optical network.
    :param graph: network to be processed
    :return: total fibre length
    """
    total_fibre = 0
    for s,d in graph.edges:
        total_fibre += graph[s][d]["weight"]*80
    return total_fibre

def create_spatial_ER_graph(grid_graph, E):
    N=len(grid_graph)
    p = (E / scipy.special.binom(N, 2))
    grid_graph = nx.convert_node_labels_to_integers(grid_graph)
    er_graph = nx.Graph()
    er_graph.add_nodes_from([1,2])

    while nx.is_connected(er_graph) == False:
        er_graph = nx.erdos_renyi_graph(len(grid_graph), p)
        # if nx.is_connected(er_graph):
            # print("graph is connected!!")
        nx.set_node_attributes(er_graph, dict(grid_graph.nodes.data()))
        p+=0.01
    topology = nt.Topology.Topology()
    er_graph = topology.assign_distances_grid(er_graph, harvesine=True)
    return er_graph

def prufer_sequence_ptd(N, E, grid_graph):
    '''
    Method to design graph with prufer sequence
    :param N: node number
    :param E: edge number
    :param T_C: traffic matrix
    :param grid_graph: node positions
    :return graph: The designed graph
    '''

    #     print(len(Prufer_sq))

    graph = nx.Graph()
    graph.add_nodes_from(list(np.arange(1, N + 1)))
    All_node = np.arange(1, N + 1)

    while True:
        Prufer_sq = list(np.random.randint(1, N + 1, size=N - 2))
        eligible_nodes = list(set(All_node).difference(set(Prufer_sq)))
        if len(eligible_nodes) == E - (N - 1):
            break

    sorted(eligible_nodes)

    leaf_nodes = eligible_nodes.copy()

    temp_sq = Prufer_sq.copy()

    for i in range(len(Prufer_sq)):

        graph.add_edge(eligible_nodes[0], temp_sq[0])
        eligible_nodes.pop(0)

        if len(temp_sq) >= 2:
            if temp_sq[0] not in temp_sq[1:]:
                eligible_nodes.append(temp_sq[0])
                sorted(eligible_nodes)
        elif len(temp_sq) == 1:
            eligible_nodes.append(temp_sq[0])

        temp_sq.pop(0)

    #     print(temp_sq)
    #     print(eligible_nodes)

    graph.add_edge(eligible_nodes[0], eligible_nodes[1])

    #     print(leaf_nodes)
    for i in range(len(leaf_nodes) - 1):
        graph.add_edge(leaf_nodes[i], leaf_nodes[i + 1])

    graph.add_edge(leaf_nodes[0], leaf_nodes[-1])

    #     print(len(graph.edges))

    nx.set_node_attributes(graph, dict(grid_graph.nodes.data()))
    top = nt.Topology.Topology()
    graph = top.assign_distances_grid(graph, pythagorus=False,
                                      harvesine=True)

    #     print(top.check_bridges(graph))

    #     DWC = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], [T_C], alpha)[0]

    return graph

@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


def pairs_list_to_mat(graph, pairs_list):
    """
    Method to convert a connection request set in form [[s,d], ...] to a matrix.
    :param pairs_list: conn req list in form [[s,d], ...]
    :return: matrix [[amount of requests], ...]
    """
    conn_matrix = np.zeros((len(graph), len(graph)))
    for s, d in pairs_list:
        # assert [d,s] not in pairs_list
        conn_matrix[s - 1, d - 1] += 1
        conn_matrix[d - 1, s - 1] += 1
    return conn_matrix

def mat_to_pairs_list(conn_matrix):
    """

    :param conn_matrix:
    :return:
    """
    connection_pairs = []
    for i in range(len(conn_matrix)):
        for j in range(len(conn_matrix)):
            if i > j:
                for k in range(int(conn_matrix[i,j])):
                    connection_pairs.append((i+1,j+1))
            else:
                pass
    return connection_pairs




def rwa_dict_to_list(rwa_dict):
    """
    Method to convert an rwa dictionary to a list
    :param rwa_dict: dictionary {wavelength:[[path],[path],...]}
    :return: rwa_list: list ([[[path],[path]], [[path], [path]],...]
    """
    rwa_list = []
    for key in rwa_dict.keys():
        rwa_list.append(rwa_dict[key])

    return rwa_list


def rwa_list_to_dict(rwa_list):
    """
    Method to convert an rwa list to a dictionary
    :param rwa_list: list ([[[path],[path]], [[path], [path]],...]
    :return: rwa_dict: dictionary {wavelength:[[path],[path],...]}
    """
    rwa_dict = {}
    for ind, paths in enumerate(rwa_list):
        rwa_dict[ind] = paths
    return rwa_dict


def single_to_multi_fibre_rwa(rwa, routing_channels, channels_per_link):
    rwa_len = int(routing_channels / channels_per_link)
    rwa_multi = [{wavelength: [] for wavelength in range(channels_per_link)} for i in range(rwa_len)]
    for i in tqdm(range(channels_per_link)):
        for j in range(rwa_len):
            rwa_multi[j][i] = rwa[int(i * j)]
    return rwa_multi

def get_demand_weighted_cost_combined(graph_list,T_c,Alpha,penalty_num = 1000):
    """
    Method to get demand weighted cost in a topology dataset.
    :param graph_list: networkx graph and id list to process
    :param T_C: Traffic demand in terms of connections
    :param Alpha: Weight of different K shortest path
    :penalty_num: cost when there is no paths between a certain node pair
    :return: list of demand weighted cost
    """
    DWC = []
    
    
    for graph,ind in graph_list:
        if 0 in graph.nodes():
            graph = nx.relabel_nodes(graph, lambda x: x+1)
        
        E = len(graph.edges())
        alpha = E/N/(N-1)*2
        DWC_structure = nt.Tools.get_demand_weighted_cost([[graph,0]], [T_c], Alpha,penalty_num)[0]
        DWC_distance = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], [T_c], Alpha,penalty_num)[0]
        dwc = alpha*DWC_distance + (1-alpha)* DWC_structure
        DWC.append(dwc)
    
    return DWC
    
    
    
def get_demand_weighted_cost_distance(graph_list,T_c,Alpha,penalty_num = 1000):
    """
    Method to get demand weighted cost in a topology dataset.
    :param graph_list: networkx graph and id list to process
    :param T_C: Traffic demand in terms of connections
    :param Alpha: Weight of different K shortest path
    :penalty_num: cost when there is no paths between a certain node pair
    :return: list of demand weighted cost
    """

    DWC = []
    T_c_ind = 0
    K = len(Alpha)

    for graph,ind in graph_list:
        if 0 in graph.nodes():
            graph = nx.relabel_nodes(graph, lambda x: x+1)

            
        node_num = len(graph.nodes())
        edge_num = len(graph.edges())

        ksp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, e=10000, k=K, weighted= 'weight')
#         print(ksp)

        SP = np.zeros((K,node_num,node_num))

        for sd,paths in ksp:
            for ind,path in enumerate(paths):
#                 print('sd:',sd)
#                 print('paths:',len(paths))
                SP[ind][sd[0]-1][sd[1]-1]=  nt.Routing.Tools.path_cost(graph, paths[ind], weight= 'weight')
                SP[ind][sd[1]-1][sd[0]-1]= nt.Routing.Tools.path_cost(graph, paths[ind], weight= 'weight')
        
        SP[np.where(SP==0)] = penalty_num

#         print(SP[0])
#         print(SP[1])
#         print(T_b[T_b_ind])
        dwc = 0
        for k,alpha in enumerate(Alpha):
            for i in range(node_num):
                    for j in range(i+1,node_num):
                        dwc += alpha*T_c[T_c_ind][i][j]*SP[k][i][j]
        DWC.append(dwc/edge_num)
        T_c_ind += 1
        
    return DWC

def get_demand_weighted_cost(graph_list, T_C, Alpha, penalty_num=100):
    """
    Method to get demand weighted cost in a topology dataset.
    :param graph_list: networkx graph and id list to process
    :param T_C: Traffic demand in terms of connections
    :param Alpha: Weight of different K shortest path
    :penalty_num: cost when there is no paths between a certain node pair
    :return: list of demand weighted cost
    """

    DWC = []
    T_C_ind = 0
    K = len(Alpha)

    for graph, ind in graph_list:
        if 0 in graph.nodes():
            graph = nx.relabel_nodes(graph, lambda x: x+1)

        node_num = len(graph.nodes())
        edge_num = len(graph.edges())

        ksp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, e=None, k=K, weighted=None)
        #         print(ksp)

        SP = np.zeros((K, node_num, node_num))
        #         print(node_num)

        for sd, paths in ksp:
            for ind, path in enumerate(paths):
                SP[ind][sd[0] - 1][sd[1] - 1] = len(path) - 1
                SP[ind][sd[1] - 1][sd[0] - 1] = len(path) - 1

        SP[np.where(SP == 0)] = penalty_num
        # print(SP[0])
        # print(SP[1])
        # print(T_C[T_C_ind])
        dwc = 0
        for k, alpha in enumerate(Alpha):
            for i in range(node_num):
                for j in range(i + 1, node_num):
                    dwc += alpha * T_C[T_C_ind][i][j] * SP[k][i][j]
        DWC.append(dwc / edge_num)
        T_C_ind += 1

    return DWC


def accumulate_raw_degree_data(graph, degree_list):
    """
    Method to accumulate the raw data of each degree of a node and append to degree_list
    :param graph: graph from which to loop over nodes for
    :param degree_list: list of degrees of nodes
    :return: degree_list - list of degrees
    """
    for node, degree in graph.degree():
        degree_list.append(degree)
    return degree_list


def convert_degree_dict_to_list(degree_dict):
    """
    Method to convert a dictionary of degrees to a list, for plotting and processing.
    :param degree_dict: input dictionary to convert to a list
    :return: cnt of degree list - np.array()
    """
    cnt_tot = []
    for key in degree_dict.keys():
        cnt_tot.append(degree_dict[key])
    cnt_tot = np.asarray(cnt_tot)
    return cnt_tot


def accumulate_degree_sequence(graph, degree_dict=None):
    """
    Method to accumulate the degree sequence of a graph in a dictionary.
    :param graph: networkx graph to process
    :param degree_dict: degree dictionary to use (zero initialised) otherwised derived
    :return: dictionary keyed by degree and count
    """
    import collections
    if degree_dict == None:
        _max_range = np.max(graph.degree) + 1
        degree_dict = {i: 0 for i in range(_max_range)}

    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    degree, cnt = zip(*degree_count.items())
    for ind, degr in enumerate(degree):
        degree_dict[degr] += cnt[ind]
    return degree_dict


def kl_divergence(p, q):
    return np.sum(np.where(q * p != 0, p * np.log(p / q), 0))


def normalise_func_out(func):
    def normalised(*args, **kwargs):
        outputs = func(*args, **kwargs)
        normalised_outputs = np.divide(outputs, outputs.sum())
        return normalised_outputs

    return normalised


def save_dictionary(dictionary, name, location):
    _json = json.dumps(dictionary)
    f = open("{}/{}.json".format(location, name), "w")
    f.write(_json)
    f.close()


def load_graph(name):
    """
    Method to load graphs from weighted edge lists stored in Topology

    :param name: Name of graph to be loaded
    :return: returns graph with congestion and NSR assignments
    :rtype: nx.Graph()
    """
    graph = nx.read_weighted_edgelist(
        path="/home/uceeatz/Code/Optical-Networks/Topology/{}".format(name + ".weighted.edgelist"),
        create_using=nx.Graph(), nodetype=int)
    # graph = self.assign_congestion(graph)
    # graph = self.assign_NSR(graph)
    return graph


def load_dictionary(name, location):
    with open('{}.json', 'r') as f:
        dictionary = json.load(f)
    return dictionary


def save_data(data, name, location):
    data_df = pd.DataFrame.from_dict(data)
    print(data_df)
    data_df.to_json(path_or_buf="{}/{}.json".format(location, name))


def load_data(name, location):
    data_df = pd.read_json(path_or_buf="{}/{}.json".format(location, name))
    return data_df


def load_data_csv(name, location):
    data_df = pd.read_csv(path_or_buf="{}/{}.json".format(location, name))


def load_network_data(general_name, location, range, parameter):
    network_data = []
    for i in range:
        print(general_name)
        data_df = load_data(general_name.format(i), location)
        print(data_df)
        network_data.append(data_df[parameter].tolist()[0])
    return network_data


def get_vol(graph):
    E = len(list(graph.edges()))
    node_sum = 0
    for node in graph.nodes():
        node_sum += graph.degree(node)
    # print("node sum: {}".format(node_sum))
    return 2 * E


def load_network_data_continuous(name, location, parameter):
    data_df = load_data(name=name, location=location)
    return data_df[parameter].tolist()


def load_network_data_over_range(general_name, location, range_1, range_2, parameter):
    network_data = [[0 for i in range_2] for j in range_1]
    for i in range_1:
        for j in range_2:
            data_df = load_data(general_name.format(i, j), location)
            network_data[range_1.index(i)][j] = data_df[parameter].tolist()[0]
    return network_data


def get_spectral_gap_laplacian(graph):
    # get largest eigenvalue
    # get second largest eigenvalue
    # calculate difference
    L = np.asmatrix((nx.normalized_laplacian_matrix(graph).toarray()))
    eig = np.linalg.eig(L)
    eig_sorted = sorted(eig[0], key=float)
    spectral_gap = abs(eig_sorted[-1]) - abs(eig_sorted[-2])
    return spectral_gap


def get_largest_eig_laplacian(graph):
    L = np.asmatrix((nx.normalized_laplacian_matrix(graph).toarray()))
    eig = np.linalg.eig(L)
    print(eig[0])
    eig_sorted = sorted(eig[0], key=float)
    lambda_max = eig_sorted[-1]
    print(eig_sorted)
    print("lambda_max: {}".format(lambda_max))
    return lambda_max


def get_cheeger_lower_bound(graph):
    spectral_gap = get_spectral_gap_laplacian(graph)
    cheeger_lower = spectral_gap / 2
    return cheeger_lower


def get_cheeger_upper_bound(graph):
    degree = []
    for node in graph.nodes():
        degree.append(graph.degree(node))
    max_degree = max(degree)
    spectral_gap = get_spectral_gap_laplacian(graph)
    cheeger_upper = np.sqrt(2 * spectral_gap * max_degree)
    return cheeger_upper


def get_cheeger_diff(graph):
    cheeger_upper = get_cheeger_upper_bound(graph)
    cheeger_lower = get_cheeger_lower_bound(graph)
    cheeger_diff = cheeger_upper - cheeger_lower
    return cheeger_diff


def find_max_length_path(equal_cost_paths):
    path_length = []
    for node_pair in equal_cost_paths:
        for path in node_pair[1]:
            path_length.append(len(path))
    return max(path_length)


def get_communicability_traffic_index(graph, TM):
    for edge in graph.edges():
        graph[edge[0]][edge[1]]["weight"] = 1
    max_TM = np.max(TM)
    xi = get_communicability_distance(graph)
    TM_inv = np.reciprocal(TM)
    np.fill_diagonal(TM_inv, 0)
    TM_inv = TM_inv * max_TM
    comm_traff_index = xi - TM_inv
    comm_traff_index = np.linalg.norm(comm_traff_index)

    return comm_traff_index


def get_communicability_distance(graph):
    A = nx.to_numpy_array(graph)
    eig = np.linalg.eigh(A)
    ind_sort = np.argsort(eig[0])
    Q = eig[1][:, ind_sort]
    eig_val = eig[0][ind_sort]
    e_diag = np.diagflat(np.exp(eig_val))
    G = Q @ e_diag @ Q.T
    comm_dist = np.zeros((len(G[:, 0]), len(G[:, 0])))
    for i in range(len(G[:, 0])):
        for j in range(len(G[:, 0])):
            comm_dist[i, j] = np.sqrt((G[i, i] + G[j, j] - 2 * G[i, j]))
    return comm_dist


def get_communicability_index(graph):
    comm_dist = get_communicability_distance(graph)
    sum_dist = 0
    for row in comm_dist:
        sum_dist += sum(row)
    return sum_dist / 2


def communicability(graph):
    import NetworkToolkit as nt
    L = np.asmatrix((nx.laplacian_matrix(graph).toarray()))
    # L = np.asmatrix(nx.normalized_laplacian_matrix(graph).toarray())
    # print(type(L))
    eig = np.linalg.eig(L)
    # print(eig[0])

    Q = eig[1]
    Q_t = np.transpose(eig[1])
    # print(Q)
    # print(Q_t)
    # print(Q * Q_t)
    diag = np.diagflat(eig[0])
    rwa = nt.Router.RWA(graph, 156, 32e9)
    rwa.get_k_shortest_paths_MNH(e=3)
    max_path_length = find_max_length_path(rwa.equal_cost_paths)
    # print(max_path_length)
    diag_sum = 0
    # for k in range(max_path_length):
    #    diag_sum += diag**k/(math.factorial(k))
    k_diag = np.asmatrix(diag ** max_path_length)
    # print(k_diag)
    route_communicability = Q * k_diag * Q_t
    # print(route_communicability)
    # print(L ** 4)
    G = route_communicability
    comm_dist = [[0 for j in range(len(G[:, 0]))] for i in range(len(G[:, 0]))]
    for i in range(len(G[:, 0])):
        for j in range(len(G[:, 0])):
            # print(G[i,i] + G[j, j] - 2*G[i, j])
            comm_dist[i][j] = np.sqrt(G[i, i] + G[j, j] - 2 * G[i, j])
    # print(comm_dist)
    sum_dist = 0
    for row in comm_dist:
        sum_dist += sum(row)
    # print(sum_dist / 2)
    return sum_dist / 2


def get_longest_routing_path(graph):
    import NetworkToolkit as nt
    def sort_path_len(path):
        return len(path)

    # find all equal cost paths
    # sort them in terms of length
    # find the length of the longest path given
    # rwa = nt.Routing.Router.RWA(graph, 156, 32e9)
    equal_cost_paths = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, e=3)

    P = []
    for dest in equal_cost_paths:
        for path in dest[1]:
            P.append(path)
    P = sorted(P, key=sort_path_len)
    # print("l[-1]: {}".format(len(P[-1])))
    # print("l[0]: {}".format(len(P[0])))
    # print("P:{}".format(P))

    return len(P[-1])


def get_lower_congestion_bound(graph):
    import math
    vol_G = get_vol(graph)
    # print("Volg: {}".format(vol_G))
    l = get_longest_routing_path(graph)
    lambda_1 = get_lambda_1_laplacian(graph)
    # print(lambda_1)
    # print("l: {}".format(l))
    # print("kambda_1: {}".format(lambda_1))
    m_lower = vol_G / ((l * lambda_1))
    # print("m_lower: {}".format(m_lower))
    return math.ceil(m_lower / 2)


def get_chromatic_lower(graph):
    lambda_max = get_largest_eig_laplacian(graph)
    chromatic_lower = 1 + (1) / (lambda_max - 1)
    return np.ceil(chromatic_lower)


def get_lambda_1_laplacian(graph):
    L = np.asmatrix((nx.normalized_laplacian_matrix(graph).toarray()))
    # print(L)
    eig = np.linalg.eig(L)
    # print(eig[0])
    lambda_1 = sorted(eig[0], key=float)[1]
    # print(eig[0])
    # print("lambda 1: {}".format(lambda_1))
    return lambda_1


def get_max_throughput_worst_case(graph, channel_spacing, B_ch, D_b_int):
    import Routing
    rwa = Routing.RWA(graph, B_ch, channel_spacing)
    rwa.add_wavelengths_full_occupation(B_ch)
    rwa.add_uniform_launch_power_to_links(B_ch)
    rwa.add_non_linear_NSR_to_links(channels_full=B_ch, channel_bandwidth=channel_spacing)
    R_s = []
    SNR_shortest_path = rwa.get_SNR_shortest_path_node_pair(B_ch)
    for item in SNR_shortest_path:
        R_s.append(100e9 / np.log2(1 + item[2]))
    R_s = np.asarray(R_s)
    D_ij = np.ceil(R_s / channel_spacing)
    # create demand matrix from D_ij
    node_num = len(list(graph.nodes))
    demand_matrix = np.zeros((node_num, node_num))
    for s_d in SNR_shortest_path:
        demand_matrix[s_d[0] - 1, s_d[1] - 1] = D_ij[SNR_shortest_path.index(s_d)]
        demand_matrix[s_d[1] - 1, s_d[0] - 1] = D_ij[SNR_shortest_path.index(s_d)]

    h = get_h_brute_force_non_uniform(rwa.RWA_graph, demand_matrix)
    D_max = np.floor(B_ch * h)
    max_throughput = D_b_int * D_max * node_num * (node_num - 1)
    return max_throughput


def assign_congestion(graph):
    congestion = list(map(lambda x: (x[0], x[1], {"congestion": 0}), graph.edges))
    graph.add_edges_from(congestion)
    return graph


def get_max_throughput_worst_case_exact(graph, channel_spacing=32e9, B_ch=156, D_b=32e9, T=10e9):
    network = Network.OpticalNetwork(graph)
    network.physical_layer.add_wavelengths_full_occupation(network.channels)
    network.physical_layer.add_uniform_launch_power_to_links(network.channels)
    network.physical_layer.add_non_linear_NSR_to_links()
    shortest_paths = network.rwa.get_shortest_dijikstra_all()
    shortest_paths = nt.Routing.Tools.get_shortest_dijikstra_all(graph)
    SNR_shortest_path = network.physical_layer.get_SNR_shortest_path_node_pair(network.channels, shortest_paths)
    R_s = []
    node_num = len(list(graph.nodes))
    R_ij = np.zeros((node_num, node_num))
    for item in SNR_shortest_path:
        R_s.append(T / np.log2(1 + item[2]))
    R_s = np.asarray(R_s)
    for s_d in SNR_shortest_path:
        R_ij[s_d[0] - 1, s_d[1] - 1] = R_s[SNR_shortest_path.index(s_d)]
        R_ij[s_d[1] - 1, s_d[0] - 1] = R_s[SNR_shortest_path.index(s_d)]
    # create demand matrix from D_ij
    node_num = len(list(graph.nodes))
    demand_matrix = np.zeros((node_num, node_num))
    for s_d in SNR_shortest_path:
        demand_matrix[s_d[0] - 1, s_d[1] - 1] = D_b
        demand_matrix[s_d[1] - 1, s_d[0] - 1] = D_b
    D_max = get_D_brute_force_non_uniform_bandwidth(graph, R_ij)
    sum_T = 0
    for row in np.nditer(demand_matrix):
        sum_T += row
    max_throughput = (sum_T * D_max) / 2
    # print("max throughput: {} Tbps ".format(max_throughput*1e-12))
    return max_throughput


def get_max_throughput_non_uniform(graph, demand, channel_spacing, B_ch, D_b=100e9, SNR_shortest_path=None):
    if SNR_shortest_path == None:
        import Routing
        rwa = Routing.RWA(graph, B_ch, channel_spacing)

        rwa.add_wavelengths_full_occupation(B_ch)
        rwa.add_uniform_launch_power_to_links(B_ch)
        rwa.add_non_linear_NSR_to_links(channels_full=B_ch, channel_bandwidth=channel_spacing)
        R_s = []
        node_num = len(list(graph.nodes))
        R_ij = np.zeros((node_num, node_num))
        SNR_shortest_path = rwa.get_SNR_shortest_path_node_pair(B_ch)
        for item in SNR_shortest_path:
            R_s.append(100e9 / np.log2(1 + item[2]))
        R_s = np.asarray(R_s)
        for s_d in SNR_shortest_path:
            R_ij[s_d[0] - 1, s_d[1] - 1] = R_s[SNR_shortest_path.index(s_d)]
            R_ij[s_d[1] - 1, s_d[0] - 1] = R_s[SNR_shortest_path.index(s_d)]
        D_ij = np.ceil(R_s / channel_spacing)

        # create demand matrix from D_ij
        node_num = len(list(graph.nodes))
        demand_matrix = np.zeros((node_num, node_num))
        for i, s_d in enumerate(SNR_shortest_path):
            demand_matrix[s_d[0] - 1, s_d[1] - 1] = demand[i]
            demand_matrix[s_d[1] - 1, s_d[0] - 1] = demand[i]

        D_max = get_D_brute_force_non_uniform_bandwidth(graph, R_ij)
        sum_T = 0
        for row in np.nditer(demand_matrix):
            sum_T += row
        max_throughput = (sum_T * D_max) / 2
        print("max throughput: {} Tbps ".format(max_throughput * 1e-12))
        return max_throughput
    else:
        import Routing
        R_s = []
        node_num = len(list(graph.nodes))
        R_ij = np.zeros((node_num, node_num))
        for item in SNR_shortest_path:
            R_s.append(100e9 / np.log2(1 + item[2]))
        R_s = np.asarray(R_s)
        for s_d in SNR_shortest_path:
            R_ij[s_d[0] - 1, s_d[1] - 1] = R_s[SNR_shortest_path.index(s_d)]
            R_ij[s_d[1] - 1, s_d[0] - 1] = R_s[SNR_shortest_path.index(s_d)]
        D_ij = np.ceil(R_s / channel_spacing)

        # create demand matrix from D_ij
        node_num = len(list(graph.nodes))
        demand_matrix = np.zeros((node_num, node_num))
        for i, s_d in enumerate(SNR_shortest_path):
            demand_matrix[s_d[0] - 1, s_d[1] - 1] = demand[i]
            demand_matrix[s_d[1] - 1, s_d[0] - 1] = demand[i]

        D_max = get_D_brute_force_non_uniform_bandwidth(graph, R_ij)
        sum_T = 0
        for row in np.nditer(demand_matrix):
            sum_T += row
        max_throughput = (sum_T * D_max) / 2
        # print("max throughput: {} Tbps ".format(max_throughput*1e-12))
        return max_throughput


def get_D_brute_force_non_uniform_bandwidth(graph, bandwidth_matrix, B_o=5e12):
    """
    This method calculates the max(D) value for a graph, given the bandwidth matrix. This can calcuate non-uniform demands.
    :param graph: graph to use for calculation
    :param bandwidth_matrix: bandwidth that has to be allocated to all node-pair matrices
    :param B_o: Total optical bandwidth to be used (standard C-band - 5THz)
    :return: max(D)
    """
    # for edge in graph.edges():
    #    graph[edge[0]][edge[1]]["weight"] = 1
    A = nx.to_numpy_matrix(graph, weight=None)

    D = np.sum(A, axis=1)  # axis =1 for rows, axis = 0 for columns
    demand_matrix_sum = np.sum(D, axis=1)
    N = len(list(graph.nodes()))
    bound = (np.sum(D))
    sorted_D_ind = np.argsort(D, axis=0).flatten() + np.ones(np.shape(np.argsort(D, axis=0).flatten()))
    # sorted_D = D[sorted_D_ind]
    # print("sorted D ind: {}".format(sorted_D_ind))
    cuts = np.array(list(np.arange(1, np.uint64(2 ** N), dtype=np.uint64)), dtype=np.uint64)

    cuts = np.reshape(cuts, (len(cuts), 1))
    cuts = np.unpackbits(cuts.view(np.uint8), axis=1, bitorder="little")

    cuts = cuts[:, :N]
    cuts = cuts.T
    # print(cuts)
    # binary_repr = np.vectorize(np.binary_repr)
    # cuts = binary_repr(cuts)
    # print(cuts)
    # str_bin_to_list = lambda x: [int(y) for y in x]
    # str_bin_to_list = np.vectorize(str_bin_to_list)
    # cuts_vectors = np.zeros((2**N, N))
    # print(cuts_vectors)
    # print(str_bin_to_list(cuts))
    # print(str_bin_to_list("111"))
    # print(np.fromstring("11111111", sep="", dtype=int))
    # print(sorted_D_ind)
    # print(cuts)
    bandwidth_sum = np.zeros(np.shape(cuts[0, :]))
    nodes_A = np.multiply(sorted_D_ind.T, cuts)
    nodes_B = np.multiply(sorted_D_ind.T, np.logical_not(nodes_A))
    for column in range(len(cuts[0, :])):
        nodes_A_ind = nodes_A[:, column]
        nodes_A_ind = nodes_A_ind[nodes_A_ind != 0]
        nodes_A_ind = nodes_A_ind - np.ones(np.shape(nodes_A_ind))
        nodes_B_ind = nodes_B[:, column]
        nodes_B_ind = nodes_B_ind[nodes_B_ind != 0]
        nodes_B_ind = nodes_B_ind - np.ones(np.shape(nodes_B_ind))
        # print("A:\n{}".format(nodes_A_ind))
        # print("B:\n{}".format(nodes_B_ind))
        # print("cartesian:\n{}".format(np.asarray(np.meshgrid(nodes_A_ind, nodes_B_ind)).T.reshape(-1,2)))
        cartesian = np.asarray(np.meshgrid(nodes_A_ind, nodes_B_ind)).T.reshape(-1, 2)
        sum = 0
        for row in range(len(cartesian[:, 0])):
            sum += bandwidth_matrix[np.int(cartesian[row, 0]), np.int(cartesian[row, 1])]
        # print("sum: {}".format(sum))

        bandwidth_sum[column] = sum

        # for row in range(len(cuts[:,0])):
        #   np.insert(Demand_sum, column)
        # np.insert(Demand_sum, column, np.sum(demand_matrix[]))
    num_edges = np.einsum('ij,ij->j', np.logical_not(cuts), A * cuts)
    D_array = (num_edges * B_o) / bandwidth_sum
    D_array = D_array[~np.isnan(D_array)]
    D = np.floor(np.amin(D_array))
    return D


def get_h_brute_force_non_uniform(graph, demand_matrix):
    A = nx.to_numpy_matrix(graph, weight=None)
    N = len(list(graph.nodes()))
    demand_matrix[np.isinf(demand_matrix)] = 0

    cuts = np.array(list(range(1, 2 ** N - 1)), dtype=np.uint64)
    #     print(cuts)
    cuts = np.reshape(cuts, (len(cuts), 1))
    cuts = np.unpackbits(cuts.view(np.uint8), axis=1, bitorder="little")
    cuts = cuts[:, :N]
    cuts = cuts.T
    #     print(cuts)
    cuts = np.array(cuts)

    whole = np.ones([len(cuts), len(cuts[0])])
    cuts_inverse = whole - cuts

    h_array = []

    for i in range(cuts.shape[1]):
        cut_num = np.dot(cuts_inverse[:, i].T, np.dot(A, cuts[:, i]).T)
        demand_sum = np.dot(cuts_inverse[:, i].T, np.dot(demand_matrix, cuts[:, i]).T)
        #         print(cut_num)
        #         print(demand_sum)
        h_array.append(cut_num / demand_sum)
    #         print(h_array)

    h = np.min(h_array)
    ind = np.argmin(h_array)
    sub_A = cuts[:, ind] * range(1, N + 1)
    sub_B = range(1, N + 1) - cuts[:, ind] * range(1, N + 1)
    return h, sub_A, sub_B


def get_h_brute_force(graph):
    # for edge in graph.edges():
    #    graph[edge[0]][edge[1]]["weight"] = 1
    A = nx.to_numpy_matrix(graph, weight=None)

    D = np.sum(A, axis=1)  # axis =1 for rows, axis = 0 for columns

    N = len(list(graph.nodes()))
    bound = (np.sum(D))
    sorted_D_ind = np.argsort(D)
    sorted_D = D[sorted_D_ind]
    # print("sorted D ind: {}".format(sorted_D_ind))

    cuts = np.array(list(range(1, 2 ** N)), dtype=np.uint64)

    cuts = np.reshape(cuts, (len(cuts), 1))

    # print(cuts)
    # binary_repr = np.vectorize(np.binary_repr)
    # cuts = binary_repr(cuts)
    # print(cuts)
    # str_bin_to_list = lambda x: [int(y) for y in x]
    # str_bin_to_list = np.vectorize(str_bin_to_list)
    # cuts_vectors = np.zeros((2**N, N))
    # print(cuts_vectors)
    # print(str_bin_to_list(cuts))
    # print(str_bin_to_list("111"))
    # print(np.fromstring("11111111", sep="", dtype=int))
    cuts = np.unpackbits(cuts.view(np.uint8), axis=1, bitorder="little")

    cuts = cuts[:, :N]
    cuts = cuts.T

    num_edges = np.einsum('ij,ij->j', np.logical_not(cuts), A * cuts)

    h_array = num_edges / (np.sum(np.logical_not(cuts), axis=0) * np.sum(cuts, axis=0))
    h_array = h_array[~np.isnan(h_array)]
    h = np.amin(h_array)
    h_index = np.where(h_array == h)
    # print("sorted_D:\n{}".format(np.argsort(D, axis=0).flatten()+np.ones(np.shape(np.argsort(D, axis=0).flatten()))))
    # print(cuts)
    # print(h_index[0][0])
    # print(cuts[:, h_index[0][0]])
    subgraph_A = np.multiply(np.argsort(D, axis=0).flatten() + np.ones(np.shape(np.argsort(D, axis=0).flatten())),
                             cuts[:, h_index[0][0]])
    subgraph_B = np.multiply(np.argsort(D, axis=0).flatten() + np.ones(np.shape(np.argsort(D, axis=0).flatten())),
                             np.logical_not(cuts[:, h_index[0][0]]))
    # print(subgraph_A[subgraph_A != 0])
    # print(subgraph_B[subgraph_B != 0])
    return h


def get_limiting_cut(graph):
    A = nx.to_numpy_matrix(graph, weight=None)
    D = np.sum(A, axis=1)  # axis =1 for rows, axis = 0 for columns
    N = len(list(graph.nodes()))

    sorted_D_ind = np.argsort(D, axis=0).flatten()  # +np.ones(np.shape(np.argsort(D, axis=0).flatten()))

    sorted_D_ind = np.take_along_axis(np.array(list(graph.nodes), ndmin=2), sorted_D_ind, axis=1)

    cuts = np.array(list(range(1, 2 ** N)), dtype=np.uint64)
    cuts = np.reshape(cuts, (len(cuts), 1))
    cuts = np.unpackbits(cuts.view(np.uint8), axis=1, bitorder="little")

    cuts = cuts[:, :N]
    cuts = cuts.T

    num_edges = np.einsum('ij,ij->j', np.logical_not(cuts), A * cuts)
    h_array = num_edges / (np.sum(np.logical_not(cuts), axis=0) * np.sum(cuts, axis=0))
    h_array = h_array[~np.isnan(h_array)]
    # h_array = h_array[h_array != 0]
    h = np.amin(h_array)
    h_index = np.where(h_array == h)
    subgraph_A = np.multiply(sorted_D_ind, cuts[:, h_index[0][0]])
    subgraph_B = np.multiply(sorted_D_ind,
                             np.logical_not(cuts[:, h_index[0][0]]))
    subgraph_A = np.expand_dims(subgraph_A[subgraph_A != 0].astype(int), axis=0)
    subgraph_B = np.expand_dims(subgraph_B[subgraph_B != 0].astype(int), axis=0)

    cut_edges = []
    for edge in graph.edges():
        # print("edge:{}".format(edge))
        if edge[0] in subgraph_A and edge[1] in subgraph_B or edge[1] in subgraph_A and edge[0] in subgraph_B:
            cut_edges.append(edge)
    if h == 0:
        print("h is zero")
        print(list(graph.nodes))
        print(list(graph.edges))
    return subgraph_A, subgraph_B, cut_edges, np.ceil(1 / h)


def check_graphset_totally_disconnected(graph_set):
    for graph in graph_set:
        if len(list(graph.nodes)) > 1:
            return False
        elif len(list(graph.nodes)) <= 1:
            continue

    return True


def get_uniform_occupation_distribution(graph):
    graph_set = [graph]
    edge_occupation = []
    n = 0
    while check_graphset_totally_disconnected(graph_set) == False:
        n = 0
        new_graph_set = []  # create a new graph set
        for graph in graph_set:
            n += 1
            graph = remove_disconnected_nodes(graph)
            if not graph:
                # graph_set.remove(graph)
                continue
            (S, T, E, lambda_LL) = get_limiting_cut(graph)  # get limiting cut for graph

            new_graph_set.append(
                graph.subgraph(S.tolist()[0]))  # append the new subgraph of the previous processed graph
            new_graph_set.append(
                graph.subgraph(T.tolist()[0]))  # append the new subgraph of the previous processed graph
            for edge in E:
                edge_occupation.append((edge, lambda_LL))  # append occupation values for all the cut edges

        graph_set = new_graph_set  # set new graph set after all the graphs have been processed
        graph_set = list(filter(lambda x: len(list(x.nodes())) > 1, graph_set))
    return edge_occupation


def remove_disconnected_nodes(graph):
    connected_components = nx.connected_components(graph)
    # print("connected components: {}".format(connected_components))
    for components in connected_components:
        if len(list(graph.edges())) == 0:
            return False
        if len(components) == 1:
            if nx.is_frozen(graph):
                graph = nx.Graph(graph)
            for component in components:
                graph.remove_node(component)

    return graph


def get_vol(graph):
    E = len(list(graph.edges()))
    node_sum = 0
    for node in graph.nodes():
        node_sum += graph.degree(node)
    # print("node sum: {}".format(node_sum))
    return 2 * E


def read_database_dict(dict):
    new_dict = {int(key): dict[key] for key in dict.keys()}
    return new_dict


def write_database_dict(dict):
    new_dict = {str(key): value for key, value in dict.items()}
    return new_dict


def read_database_topology(graph_data, node_data=None, use_pickle=False):
    """
    Method to read database topology and convert to nx.Graph()
    :param graph_data: result["topology data"] from pymongo query object
    :param node_data:  result["node data"]
    :return: nx.graph()
    """
    # print(graph_data)
    if node_data is not None:
        try:
            if type(graph_data) is dict:
                graph_data = {int(y): {int(z): ast.literal_eval(graph_data[y][z]) for z in graph_data[y].keys()} for y
                              in graph_data.keys()}
            else:
                graph_data = pickle.loads(graph_data)
            # if use_pickle:
            #     graph_data = pickle.loads(graph_data)
            # else:
            #     graph_data = {int(y): {int(z): ast.literal_eval(graph_data[y][z]) for z in graph_data[y].keys()} for y
            #                   in graph_data.keys()}
        except:
            if use_pickle:
                graph_data = pickle.loads(graph_data)
            else:
                graph_data = {y: {z: ast.literal_eval(graph_data[y][z]) for z in graph_data[y].keys()} for y
                              in graph_data.keys()}
        graph = nx.from_dict_of_dicts(graph_data)
        nx.set_node_attributes(graph, read_database_node_data(node_data, use_pickle=use_pickle))
    else:
        try:
            if type(graph_data) is dict:
                graph_data = {int(y): {int(z): ast.literal_eval(graph_data[y][z]) for z in graph_data[y].keys()} for y
                              in graph_data.keys()}
            else:
                graph_data = pickle.loads(graph_data)
            # if use_pickle:
            #     graph_data = pickle.loads(graph_data)
            # else:
            #     graph_data = {int(y): {int(z): ast.literal_eval(graph_data[y][z]) for z in graph_data[y].keys()} for y
            #                   in graph_data.keys()}
        except:
            if use_pickle:
                graph_data = pickle.loads(graph_data)
            else:
                graph_data = {y: {z: ast.literal_eval(graph_data[y][z]) for z in graph_data[y].keys()} for y
                              in graph_data.keys()}
        graph = nx.from_dict_of_dicts(graph_data)
    # print(len(list(graph.edges)))
    # print(graph.nodes.data())
    return graph


def graph_to_database_topology(graph, use_pickle=False):
    graph_data = nx.to_dict_of_dicts(graph)
    if use_pickle:
        graph_data = pickle.dumps(graph_data)
    else:
        graph_data = {str(y): {str(z): str(graph_data[y][z]) for z in graph_data[y].keys()} for y in graph_data.keys()}
    return graph_data


def node_data_to_database(node_data, use_pickle=False):
    if use_pickle:
        node_data = pickle.dumps(node_data)
    else:
        node_data = {str(y): {str(z): str(node_data[y][z]) for z in node_data[y].keys()} for y in node_data.keys()}
    return node_data


def read_database_node_data(node_data, use_pickle=False):
    try:
        if type(node_data) is dict:
            node_data = {int(y): {str(z): ast.literal_eval(node_data[y][z]) for z in node_data[y].keys()} for y in
                         node_data.keys()}
        else:
            node_data = pickle.loads(node_data)
        # if use_pickle:
        #     node_data = pickle.loads(node_data)
        # else:
        #     node_data = {int(y): {str(z): ast.literal_eval(node_data[y][z]) for z in node_data[y].keys()} for y in
        #               node_data.keys()}
    except:
        if use_pickle:
            node_data = pickle.loads(node_data)
        else:
            # Internet Topology Zoo topologies take a different form and need another node_data expression...
            node_data = {y: {z: node_data[y][z] for z in node_data[y].keys()} for y in
                         node_data.keys()}
    # print(node_data)
    return node_data


def create_binary_topology_vector(graph):
    for edge in graph.edges():
        graph[edge[0]][edge[1]]["weight"] = 1
    A = nx.to_numpy_matrix(graph)
    N = len(list(graph.nodes()))
    topology_vector = np.ndarray([])
    for i in range(1, N):
        topology_vector = np.append(topology_vector, np.diagonal(A, i))
    topology_vector.flatten()

    return topology_vector


def calculate_number_of_spanning_trees(graph):
    """

    :param graph:
    :return:
    """
    G = graph.copy()
    for edge in G.edges():
        G[edge[0]][edge[1]]["weight"] = 1
    A = nx.to_numpy_array(G)

    L = nx.laplacian_matrix(G).toarray()

    L = np.delete(L, 0, axis=0)
    L = np.delete(L, 0, axis=1)
    S = np.linalg.det(L)

    return int(S)


def nodes_to_edges(nodes):
    """
    Method to convert a path into a list of edges.
    :param nodes: path to convert - [path]
    :return: list of edges
    :rtype: [(edge), ..., (edge)]
    """
    edges = []
    for i in range(0, len(nodes) - 1):
        edges.append((nodes[i], nodes[i + 1]))
    return edges


def find_path_edges_set_all_to_all(graph, e=3):
    nodes = list(graph.nodes())
    edge_num = len(list(graph.edges()))
    edge_set = set(range(1, edge_num + 1))
    edges = list(graph.edges())
    rwa = nt.Router.RWA(graph, 156, 32e9)
    SP = rwa.get_k_shortest_paths_MNH(e=e)
    SP = rwa.k_SP_to_list_of_paths(SP)
    path_edges = []
    for item in SP:
        path_edges.append(nt.Tools.nodes_to_edges(item))
    path_edges_set = []
    for path in path_edges:
        path_indeces = []
        for edge in path:
            if edge in edges:  #
                edge_index = edges.index(edge)
                path_indeces.append(edge_index)
            else:
                edge_index = edges.index((edge[1], edge[0]))
                path_indeces.append(edge_index)
        path_edges_set.append(path_indeces)
    # print(path_edges_set)
    return path_edges_set


def convert_paths_to_SP(edges, SP):
    path_edges = []
    for item in SP:
        path_edges.append(nt.Tools.nodes_to_edges(item))
    path_edges_set = []
    for path in path_edges:
        path_indeces = []
        for edge in path:
            if edge in edges:  #
                edge_index = edges.index(edge)
                path_indeces.append(edge_index + 1)
            else:
                edge_index = edges.index((edge[1], edge[0]))
                path_indeces.append(edge_index + 1)
        path_edges_set.append(path_indeces)
    # print(path_edges_set)
    return path_edges_set


def find_indeces_paths_edges(edges, path_edges_set):
    # print(edges)
    # SP = list(map(lambda x: [i -1 if i > 0 else i for i in x], path_edges_set.copy()))
    # print(SP)
    # print(len(edges))
    edges_message_indeces = [[] for i in range(len(edges))]
    # print(edges_message_indeces)
    for ind_p, path in enumerate(path_edges_set):
        for ind_e, edge in enumerate(path):

            # print(edge)
            if edge == 0:
                pass
            else:

                edges_message_indeces[edge - 1].append([ind_p + 1, ind_e + 1])
    # print(edges_message_indeces)
    return edges_message_indeces


def weighted_spectral_density_distance_graph_list(graph_list1, graph_list2, N, bins):
    """
    Method to calculate the distance between two different graph lists. Calculates the spectral distribution using
    all eigenvalues of each list of graphs.
    :param graph_list1:     First list of graphs [(graph, _id), ...,]
    :param graph_list2:     Second list of graphs [(graph, _id), ...,]
    :param N:               N to use for the weighted spectral distance calculation
    :param bins:            Number of bins to use of the pdf of spectral distribution
    :return:                WSD distance between graph lists spectral values
    """
    e1 = np.empty((1,))
    e2 = np.empty((1,))
    for graph1, _id in graph_list1:
        L1 = nx.normalized_laplacian_matrix(graph1)
        e1 = np.concatenate((e1, np.linalg.eigvals(L1.A)), axis=None)
    for graph2, _id in graph_list2:
        L2 = nx.normalized_laplacian_matrix(graph2)
        e2 = np.concatenate((e2, np.linalg.eigvals(L2.A)), axis=None)
    e1 = np.delete(e1, 0)
    e2 = np.delete(e2, 0)
    frequency1, edges1 = np.histogram(e1, range=(0.0, 2.0), bins=bins, density=False)
    frequency2, edges2 = np.histogram(e2, range=(0.0, 2.0), bins=bins, density=False)
    #     print(frequency1)
    #     print(edges1)
    prob1 = np.array([frequency1[i] / frequency1.sum() for i in range(len(frequency1))])
    prob2 = np.array([frequency2[i] / frequency2.sum() for i in range(len(frequency2))])

    diff = np.power(prob1 - prob2, 2)
    ones = np.ones(len(frequency1))
    bin_diff = np.power(ones - edges1[1:], N)
    wsd_diff = np.multiply(bin_diff, diff)
    wsd_sum = wsd_diff.sum()
    return wsd_sum


def create_start_stop_list(data_len, workers):
    """
    Method to create list of indeces that represent data splits to process evenely in parralel.
    :param graph_list: List of graphs to process
    :param workers:    Length of data to split
    :return:           return list of indeces to use as graph_list[indeces[i]:indeces[i+1]]
    """
    if workers > data_len:
        workers = data_len
    increment = np.floor(data_len / workers)
    start_stop = list(range(0, data_len, int(increment)))
    if len(start_stop) == workers:
        start_stop.append(data_len)
    else:
        start_stop[-1] = data_len
    return start_stop


def create_find_dic(find_dic_list):
    """
    Method to create find_dic from command line arguments
    :param find_dic_list:
    :return:
    """
    find_dic = {}
    if find_dic_list is None:
        return find_dic

    for ind, item in enumerate(find_dic_list):
        if ind * 2 == len(find_dic_list): break
        if find_dic_list[ind * 2 + 1][0:2] == "-s":
            find_dic[find_dic_list[ind * 2]] = str(find_dic_list[ind * 2 + 1][2:])
        elif find_dic_list[ind * 2 + 1][0:2] == "-i":
            find_dic[find_dic_list[ind * 2]] = int(find_dic_list[ind * 2 + 1][2:])
        elif find_dic_list[ind * 2 + 1][0:2] == "-f":
            find_dic[find_dic_list[ind * 2]] = float(find_dic_list[ind * 2 + 1][2:])
    return find_dic


if __name__ == "__main__":
    graph = load_graph("ACMN_8_node")

    # num_nodes = len(graph)
    # L = [[3, -1, -1, -1], [-1, 3, -1, -1], [-1, -1, 2, 0], [-1, -1, 0, 2]]
    # TM = np.ones((num_nodes, num_nodes)) * (1 / (num_nodes * (num_nodes - 1)))
    # comm_traff_ind = get_communicability_traffic_index(graph, TM)
    # print(comm_traff_ind)
    # print(calculate_number_of_spanning_trees(graph))
    T = get_max_throughput_worst_case_exact(graph)

    # pr.print_stats(sort='time')
