import NetworkToolkit as nt
from tqdm import tqdm
import numpy as np
import ray
import ast
import socket
import os
import networkx as nx
import time

@ray.remote
class ScaledThroughput:
    def calculate_T_UL_scaled(self,graph_data, scale=1):
        """

        :param graph_data:
        :param scale:
        :return:
        """
        graph = nt.Tools.read_database_topology(graph_data["topology data"])
        for s, d in graph.edges:
            graph[s][d]["weight"] = graph[s][d]["weight"] * scale
        new_rwa = {}
        # print(graph_data["ILP RWA assignment"])
        for key in graph_data["ILP capacity RWA assignment"]:
            new_rwa[ast.literal_eval(key)] = graph_data["ILP capacity RWA assignment"][key]

        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(new_rwa)
        network.physical_layer.add_non_linear_NSR_to_links()
        max_capacity = network.physical_layer.get_lightpath_capacities_PLI(new_rwa)
        return max_capacity


    def calculate_scaled_throughput_data(self,dataframe, start=0.01, end=1, step=0.02, graph_type="SBAG",
                                         save_dic={}):
        """

        :param dataframe:
        :param start:
        :param end:
        :param step:
        :return:
        """

        for i in tqdm(np.arange(start, end, step)):
            for index, graph_data in dataframe.iterrows():
                throughput = self.calculate_T_UL_scaled(graph_data, scale=i)[0]
                alpha = graph_data["alpha"]
                # print(throughput)
                dict1 = {"graph type": graph_type, "scale": i, "throughput data": throughput, "data type":"scaled",
                         "alpha":alpha}

                write_dic = {**dict1, **save_dic}
                nt.Database.insert_data("Topology_Data", "ECOC", write_dic)


def ILP_chromatic_number(graph, max_time=1000):
    network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
    rwa_assignment, min_wave = network.rwa.static_ILP(min_wave=True, max_seconds=max_time)
    return rwa_assignment, min_wave
@ray.remote(num_cpus=5)
def ILP_uniform_connections(graph_list=None, max_time=1000, collection=None, db="Topology_Data"):
    os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
        socket.gethostname().split('.')[0])
    for graph, _id in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        rwa_assignment, objective_value = network.rwa.maximise_uniform_connection_demand(max_time=max_time, e=10, k=10)
        rwa_assignment = nt.Tools.write_database_dict(rwa_assignment)
        nt.Database.update_data_with_id(db, collection, _id, newvals={"$set":{"ILP-uniform-connections "
                                                                              "RWA":rwa_assignment,
                                                                              "ILP-uniform-connections": objective_value}})

def parralel_ILP_uniform_connections(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data",
                                     workers=50):
    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000')
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_uniform_connections.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind+1]],
        max_time=max_time) for ind in range(workers)])

@ray.remote
def create_SBAG_top(amount, grid_graph, alpha, db_name="Topology_Data",
                    collection_name="ECOC"):
    for i in tqdm(range(amount)):
        graph = top.create_real_based_grid_graph(grid_graph, len(list(grid_graph.edges())),
                                                 database_name=db_name,
                                                 collection_name=collection_name,
                                                 sequential_adding=True,
                                                 undershoot=True,
                                                 remove_C1_C2_edges=True,
                                                 SBAG=True,
                                                 alpha=alpha
                                                 )
        nt.Database.insert_graph(graph, db_name=db_name, collection_name=collection_name,
                                 node_data=True, alpha=alpha, type="SBAG")

def create_graphs():
    grid_graph = nt.Database.read_topology_dataset_list("Topology_Data", "ECOC", find_dic={"name":
                                                                                               "NSFNET"},
                                                        node_data=True)[0][0]

    top = nt.Topology.Topology()
    ray.init(address='auto', redis_password='5241590000000000')
    alpha_range = list(np.arange(0, 5.5, 0.5))
    amount = 200
    data_len = len(alpha_range) * amount
    workers = 100
    workers_alpha = int(workers / len(alpha_range))
    data = []

    for _alpha in alpha_range:
        for i in range(workers_alpha):
            data.append((_alpha, int(data_len / workers)))

    rest = data_len - len(data) * int(data_len / workers)
    print(int(rest / len(alpha_range)))
    for _alpha in alpha_range:
        data.append((_alpha, int(rest / len(alpha_range))))
    print(data)
    print(len(data))

    results = ray.get([create_SBAG_top.remote(_amount, grid_graph, _alpha) for _alpha, _amount in data])
def ILP_chromatic(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data"):
    
    workers = 75
    print(len(graph_list))
    ray.init(address='auto', redis_password='5241590000000000')
    NetworkSimulator = (ray.remote(num_cpus=5, num_gpus=0))(nt.NetworkSimulator.NetworkSimulator)
    simulators = [NetworkSimulator.remote() for i in range(workers)]
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([s.ILP_chromatic_number_graph_list.remote(
        db_name=db,
        collection_name=collection,
        graph_list=graph_list,
        max_time=max_time,
        start=indeces[ind], stop=indeces[ind + 1]) for ind, s in enumerate(simulators)])
def ILP_throughput(graph_list, max_time=1000, collection=None, db="Topology_Data",workers=50):
    # {"ILP Capacity":{"$exists":False}
    new_graph_list = []
    for graph, _id in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        new_graph_list.append((graph, _id))
    ray.init(address='auto', _redis_password='5241590000000000')
    NetworkSimulator = (ray.remote(num_cpus=7))(nt.NetworkSimulator.NetworkSimulator)
    simulators = [NetworkSimulator.remote() for i in range(workers)]
    indeces = nt.Tools.create_start_stop_list(len(new_graph_list), workers)
    print(indeces)

    # results = ray.get([s.ILP_chromatic_number_graph_list.remote(collection_name="ECOC", db_name="Topology_Data",
    #                                start=indeces[i],stop=indeces[i+1], graph_list=graph_list) for i,s in enumerate(
    #     simulators)])
    # results = ray.get([s.ILP_max_uniform_bandwidth_graph_list.remote(collection_name="ECOC", db_name="Topology_Data",
    #                                                             start=indeces[i], stop=indeces[i + 1],
    #                                                             graph_list=graph_list) for i, s in enumerate(
    #     simulators)])
    results = ray.get([s.ILP_max_uniform_bandwidth_graph_list.remote(
        db_name=db,
        collection_name=collection,
        write_dic="cc",
        graph_list=new_graph_list,T=0,
        start=indeces[ind], stop=indeces[ind + 1], max_time=max_time,e=10, k=10, capacity_constraint=True) for ind,
                                                                                                             s in enumerate(
        simulators)])

def ILP_throughput_scaled(alpha, scale):
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ECOC", find_dic={"ILP Capacity":{
        "$exists":True},"alpha":alpha},
                                                        node_data=True)
    # {"ILP Capacity":{"$exists":False}

    new_graph_list = []
    for graph, _id in graph_list:
        for s,d in graph.edges:
            graph[s][d]["weight"] = np.ceil(scale*graph[s][d]["weight"])

        new_graph_list.append((graph, _id))
    workers = 50
    ray.init(address='auto', redis_password='5241590000000000')
    NetworkSimulator = ray.remote(nt.NetworkSimulator.NetworkSimulator)
    simulators = [NetworkSimulator.remote() for i in range(workers)]
    indeces = nt.Tools.create_start_stop_list(len(new_graph_list), workers)
    print(indeces)

    # results = ray.get([s.ILP_chromatic_number_graph_list.remote(collection_name="ECOC", db_name="Topology_Data",
    #                                start=indeces[i],stop=indeces[i+1], graph_list=graph_list) for i,s in enumerate(
    #     simulators)])
    # results = ray.get([s.ILP_max_uniform_bandwidth_graph_list.remote(collection_name="ECOC", db_name="Topology_Data",
    #                                                             start=indeces[i], stop=indeces[i + 1],
    #                                                             graph_list=graph_list) for i, s in enumerate(
    #     simulators)])
    results = ray.get([s.ILP_max_uniform_bandwidth_graph_list.remote(
        db_name="Topology_Data",
        collection_name="ECOC",
        graph_list=new_graph_list,
        max_time=3000,
        T=0,
        start=indeces[ind], stop=indeces[ind + 1], e=0, scale=scale) for ind, s in enumerate(simulators)])

def scale_throughput():
    workers = 350
    ray.init(address='auto', redis_password='5241590000000000')
    graph_data = nt.Database.read_data_into_pandas("Topology_Data", "ECOC",
                                                   find_dic={"ILP Capacity": {"$exists": True}})
    simulators = [ScaledThroughput.remote() for i in range(workers)]
    data_len = len(graph_data)
    indeces = nt.Tools.create_start_stop_list(data_len, workers)
    results = ray.get([s.calculate_scaled_throughput_data.remote(graph_data.iloc[indeces[ind]:indeces[ind + 1]],
                                                                 graph_type="SBAG", save_dic={}) for ind,
                                                                                                     s in
                       enumerate(simulators)])
def test_1(*args, **kwargs):
    print("working 1")
    time.sleep(5)
    print(args)
    print(kwargs)

def test_2(*args, **kwargs):
    print("working 2")
    print(args)
    print(kwargs)

def add_tasks(task_queue, number_of_tasks=1):
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "BA"},
                                                        node_data=True)[4200:4202]
    # graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "ER"},
    #                                                      node_data=True)[4200:4400]
    # graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "SBAG"},
    #                                                      node_data=True)[4200:4400]
    for num in range(number_of_tasks):
        # task_queue.put({"function":parralel_ILP_uniform_connections, "args":graph_list, "kwargs":{"max_time":6*3600,
        #                                                                                           "workers":600}})
        # task_queue.put({"function":ILP_throughput, "args":graph_list, "kwargs":{"max_time":48*3600,
        #                                                                         "collection":"topology-paper",
        #                                                                         "workers":600}})
        task_queue.put(
            {"function": test_1, "args": graph_list, "kwargs": {"max_time": 6 * 3600,"workers": 600}})
        task_queue.put(
            {"function": test_2, "args": graph_list, "kwargs":{"max_time":48*3600,
                                                                                "collection":"topology-paper",
                                                                                "workers":600}})
    return task_queue

def process_tasks(task_queue):
    while not task_queue.empty():
        task = task_queue.get()
        print("executing")
        func = task["function"]
        args = task["args"]
        kwargs = task["kwargs"]
        func(*args, **kwargs)
    return True
def run():
    import multiprocessing
    empty_task_queue = multiprocessing.Queue()
    tasks = add_tasks(empty_task_queue, 1)
    processes = []
    process_tasks(tasks)
    for n in range(1):
        p = multiprocessing.Process(target=process_tasks, args=(tasks,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":

    # scale_throughput()
    # graph, _id = nt.Database.read_topology_dataset_list("Topology_Data","real", find_dic={"name":
    #                                                                                  "30-Node-ONDPBook-Topology_nodes"})[0]
    # print('read graph, calculating chromatic number')
    # rwa, chrom_num = ILP_chromatic_number(graph,max_time=48*3600)
    # nt.Database.update_data_with_id("Topology_Data", "real",_id, newvals={"$set":{"wavelength requirement":chrom_num}})
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"BA",
                                                                                                     "nodes": 30
                                                                                                     # "ILP-uniform-connections":{"$exists":False}
                                                                                                     },
node_data=True)
    graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "ER",
                                                                                                      "nodes":30
                                                                                                      # "ILP-uniform-connections":{"$exists":False}
                                                                                                      },
                                                        node_data=True)
    graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "SBAG",
                                                                                                      "nodes": 30
                                                                                                      # "ILP-uniform-connections":{"$exists":False}
                                                                                                      },
                                                        node_data=True)
    print(len(graph_list))
    # parralel_ILP_uniform_connections(graph_list, max_time=6*3600, workers=len(graph_list))
    # print("waiting to start")
    # for i in tqdm(range(12*3600), desc="starting in"):
    #     time.sleep(1)

    # print("starting now")
    ILP_throughput(graph_list, max_time=48*3600, collection="topology-paper", workers=len(graph_list))
    # ILP_chromatic(graph_list, max_time=6*3600)
    # ILP_throughput_scaled(5, 0.1)

