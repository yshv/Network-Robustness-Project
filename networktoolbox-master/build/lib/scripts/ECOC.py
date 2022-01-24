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
    rwa_assignment, min_wave = network.rwa.static_ILP(min_wave=True, max_time=max_time)
    return rwa_assignment, min_wave
@ray.remote(num_cpus=1)
def ILP_uniform_connections(graph_list=None, T_c=None,max_time=1000, collection=None, db="Topology_Data", threads=10):
    os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
        socket.gethostname().split('.')[0])
    for item in graph_list:
        if T_c is not None:
            graph, _id, T_c = item
        else:
            graph, _id = item
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        if T_c is not None:
            rwa_assignment, objective_value = network.rwa.maximise_connection_demand(T_c=T_c, max_time=max_time, e=10,
                                                                                             k=10, _id=_id,
                                                                                             threads=threads)
        else:
            rwa_assignment, objective_value = network.rwa.maximise_uniform_connection_demand(max_time=max_time, e=10,
                                                                                             k=10, _id=_id, threads=threads)
        rwa_assignment = nt.Tools.write_database_dict(rwa_assignment)
        if T_c is not None:
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-connections "
                                                                                   "RWA": rwa_assignment,
                                                                                   "ILP-connections": objective_value}})
        else:
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set":{"ILP-uniform-connections "
                                                                                  "RWA":rwa_assignment,
                                                                                  "ILP-uniform-connections": objective_value}})
#, memory=5000*1024*1024
@ray.remote(num_cpus=1, memory=5000*1024*1024)
def ILP_connections(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                            threads=10, throughput=False):
    if socket.gethostname() == "MacBook-Pro":
        print("Laptop Run")
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
        node_file_dir ="/Users/robin/OneDrive/OneDrive - University College London/Code/PycharmProjects/networktoolbox/scripts/nodefiles"
    else:
        os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
            socket.gethostname().split('.')[0])
        node_file_dir = "/scratch/datasets/gurobi/nodefiles"
    for graph, _id, T_c in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)


        try:
            time_start = time.perf_counter()
            data = network.rwa.maximise_connection_demand(T_c=T_c, max_time=max_time,
                                                          e=20,
                                                          k=20, _id=_id,
                                                          threads=threads,
                                                          node_file_dir=node_file_dir,
                                                          node_file_start=0.001,
                                                          emphasis=0,
                                                          max_solutions=100)
            time_taken = time.perf_counter() - time_start
            rwa_assignment = nt.Tools.write_database_dict(data["rwa"])

            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-connections RWA": rwa_assignment,
                                                                                   "ILP-connections": data["objective"],
                                                                                   "ILP-connections gap": data["gap"],
                                                                                   "ILP-connections data written": 1,
                                                                                   "ILP-connections time": time_taken,
                                                                                   "ILP-connections status": data[
                                                                                       "status"].value}})


            if throughput:

                network.physical_layer.add_uniform_launch_power_to_links(network.channels)
                network.physical_layer.add_wavelengths_to_links(data["rwa"])
                network.physical_layer.add_non_linear_NSR_to_links()
                max_capacity = network.physical_layer.get_lightpath_capacities_PLI(data["rwa"])
                node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}

                nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-connections RWA": rwa_assignment,
                                                                                   "ILP-connections": data["objective"],
                                                                                   "ILP-connections gap":data["gap"],
                                                                                   "ILP-connections throughput written":1,
                                                                                   "ILP-connections status":data["status"].value,
                                                                                   "ILP-connections node pair capacities": node_pair_capacities,
                                                                                   "ILP-connections Capacity":max_capacity[0]}})
        except Exception as error:
            print(error)
            nt.Database.update_data_with_id(db, collection, _id,
                                            newvals={"$set": {"error occurred": 1,
                                                              "error": str(error),
                                                              "error host": str(socket.gethostname())}})
            continue


@ray.remote(num_cpus=1, memory=3000*1024*1024)
def ILP_min_path(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                            threads=10, throughput=False):
    if socket.gethostname() == "MacBook-Pro":
        print("Laptop Run")
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
        node_file_dir ="/Users/robin/OneDrive/OneDrive - University College London/Code/PycharmProjects/networktoolbox/scripts/nodefiles"
    else:
        os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
            socket.gethostname().split('.')[0])
        node_file_dir = "/scratch/datasets/gurobi/nodefiles"
    for graph, _id, T_c in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)


        try:
            time_start = time.perf_counter()
            data = network.rwa.minimise_total_path_length(T_c=T_c, max_time=max_time,
                                                          e=20,
                                                          k=20, _id=_id,
                                                          threads=threads,
                                                          node_file_dir=node_file_dir,
                                                          node_file_start=0.001,
                                                          emphasis=0,
                                                          max_solutions=100)
            time_taken = time.perf_counter() - time_start
            rwa_assignment = nt.Tools.write_database_dict(data["rwa"])

            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-connections RWA": rwa_assignment,
                                                                                   "ILP-connections": data["objective"],
                                                                                   "ILP-connections gap": data["gap"],
                                                                                   "ILP-connections data written": 1,
                                                                                   "ILP-connections time": time_taken,
                                                                                   "ILP-connections status": data[
                                                                                       "status"].value}})


            if throughput:

                network.physical_layer.add_uniform_launch_power_to_links(network.channels)
                network.physical_layer.add_wavelengths_to_links(data["rwa"])
                network.physical_layer.add_non_linear_NSR_to_links()
                max_capacity = network.physical_layer.get_lightpath_capacities_PLI(data["rwa"])
                node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}

                nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-connections RWA": rwa_assignment,
                                                                                   "ILP-connections": data["objective"],
                                                                                   "ILP-connections gap":data["gap"],
                                                                                   "ILP-connections throughput written":1,
                                                                                   "ILP-connections status":data["status"].value,
                                                                                   "ILP-connections node pair capacities": node_pair_capacities,
                                                                                   "ILP-connections Capacity":max_capacity[0]}})
        except Exception as error:
            print(error)
            nt.Database.update_data_with_id(db, collection, _id,
                                            newvals={"$set": {"error occurred": 1,
                                                              "error": str(error),
                                                              "error host": str(socket.gethostname())}})
            continue

def parralel_ILP_uniform_connections(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data",
                                     workers=50, threads=10):
    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000')
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_uniform_connections.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind+1]],
        max_time=max_time, threads=threads) for ind in range(workers)])

def parralel_ILP_connections(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data",
                                     workers=50, threads=10):
    print(len(graph_list))
    ray.shutdown()
    ray.init(address='auto', _redis_password='5241590000000000', dashboard_port=8265)
    # ray.init()
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_connections.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
        max_time=max_time, threads=threads, throughput=True) for ind in tqdm(range(workers))])
    # ray.shutdown()

def parralel_ILP_min_path(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data",
                                     workers=50, threads=10):
    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000', dashboard_port=8265)
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_min_path.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
        max_time=max_time, threads=threads, throughput=True) for ind in range(workers)])


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

@ray.remote(num_cpus=2, memory=6000*1024*1024)
def ILP_wave_req(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                            threads=10):
    os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
        socket.gethostname().split('.')[0])
    for graph, _id in graph_list:
        try:
            graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
            assert type(graph) == nx.classes.graph.Graph
            network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
            time_start = time.perf_counter()
            data = network.rwa.static_ILP(min_wave=True, max_time=max_time,e=0,k=20,threads=threads,
                                          node_file_start=0.01)
            time_taken = time.perf_counter()-time_start


            rwa_assignment = nt.Tools.write_database_dict(data["rwa"])
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-chromatic RWA": rwa_assignment,
                                                                                   "wavelength requirement": data["objective"],
                                                                                   "ILP-chromatic status":data["status"].value,
                                                                                   "ILP-chromatic gap":data["gap"],
                                                                                   "ILP-chromatic time":time_taken,
                                                                                   "ILP-chromatic data written":1}})
        except Exception as err:
            print(err)
def ILP_chromatic(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data", workers=10, threads=10):
    

    print(len(graph_list))
    ray.init(address='auto', _redis_password='5241590000000000')
    # NetworkSimulator = (ray.remote(num_cpus=1, num_gpus=0))(nt.NetworkSimulator.NetworkSimulator)
    # simulators = [NetworkSimulator.remote() for i in range(workers)]
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_wave_req.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind+1]],
        max_time=max_time, threads=threads) for ind in range(workers)])

@ray.remote(num_cpus=1, memory=4000*1024*1024)
def ILP_throughput(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                            threads=10):
    if socket.gethostname() == "MacBook-Pro":
        print("Laptop Run")
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
        node_file_dir ="/Users/robin/OneDrive/OneDrive - University College London/Code/PycharmProjects/networktoolbox/scripts/nodefiles"
    else:
        os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
            socket.gethostname().split('.')[0])
        node_file_dir = "/scratch/datasets/gurobi/nodefiles"
    for graph, _id in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        data = network.rwa.maximise_uniform_bandwidth_demand(max_time=max_time,
                                                                            e=0,
                                                                            k=20, _id=_id,
                                                                            threads=threads,
                                                                            node_file_dir=node_file_dir,
                                                                            node_file_start=0.01,
                                                                            c_type="I",
                                                                            capacity_constraint=False,
                                                                            verbose=0,
                                                                            emphasis=2)
        try:
            if data["objective"]==0:
                max_capacity=[0]
                node_pair_capacities=0
                rwa_assignment=None
            else:
                network.physical_layer.add_uniform_launch_power_to_links(network.channels)
                network.physical_layer.add_wavelengths_to_links(data["rwa"])
                network.physical_layer.add_non_linear_NSR_to_links()
                max_capacity = network.physical_layer.get_lightpath_capacities_PLI(data["rwa"])
                node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
                rwa_assignment = nt.Tools.write_database_dict(data["rwa"])
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-throughput RWA": rwa_assignment,
                                                                                   "ILP-throughput": data["objective"],
                                                                                   "ILP-capacity":max_capacity[0],
                                                                                   "ILP node pair capacities":node_pair_capacities,
                                                                                   "data written":1, "ILP-throughput status":data["status"].value,
                                                                                   "ILP-throughput gap":data["gap"]}})
        except Exception as err:
            print(err)

def parralel_ILP_throughput(graph_list, max_time=1000, collection=None, db="Topology_Data",workers=50, num_cpus=2, threads=4):
    # ray.init(
    ray.init()
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_throughput.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
        max_time=max_time, threads=threads) for ind in range(workers)])
    ray.shutdown()
@ray.remote
def heuristic_throughput(graph_list=None, collection=None, db="Topology_Data", e=10, k=10, route_function="FF-kSP",
                         m_step=100, max_count=10, channel_bandwidth=16e9):
    for graph, _id, T_c in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=channel_bandwidth, routing_func=route_function)
        rwa_assignment = False
        time_start = time.perf_counter()
        M = 0
        # print("starting routing {}".format(route_function))
        for i in range(max_count):
            while rwa_assignment != True:
                M += m_step

                demand_matrix = np.ceil(np.array(T_c)*M)
                rwa_assignment = network.route(demand_matrix, e=e, k=k)
            # print("M: {}".format(M))
            if int(M) > 1:
                M -= m_step

            demand_matrix = np.ceil(np.array(T_c) * M)
            rwa_assignment = network.route(demand_matrix, e=e, k=k)
            if int(M) == 1 and rwa_assignment == False:
                break
            elif int(M) ==1 and rwa_assignment == True:
                print("Cant route base demand")
                print("nodes: {}".format(len(graph)))
                print("edges: {}".format(len(list(graph.edges))))
                print("graph is connected? :{}".format(nx.is_connected(graph)))
                print("min degree: {}".format(min([degree for node,degree in nx.degree(graph)])))
            m_step /= 2
            m_step = np.ceil(m_step)

        assert rwa_assignment != True
        print("objective value: {}".format(M))
        time_taken = time.perf_counter() - time_start
        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(rwa_assignment)
        network.physical_layer.add_non_linear_NSR_to_links()
        max_capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)
        node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
        rwa_assignment = nt.Tools.write_database_dict(rwa_assignment)
        nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"{} RWA".format(route_function): rwa_assignment,
                                                                               "{}-connections".format(route_function): M,
                                                                               "{} throughput written".format(route_function): 1,
                                                                               "{} node pair capacities".format(route_function): node_pair_capacities,
                                                                               "{} time".format(route_function): time_taken,
                                                                               "{} Capacity".format(route_function): max_capacity[
                                                                                   0]}})




def parralel_heuristic_throughput(graph_list, collection=None, db="Topology_Data",workers=50, route_function="FF-kSP", e=10, k=10, m_step=100,
                                  channel_bandwidth=16e9):

    ray.shutdown()
    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True)
    indeces  = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([heuristic_throughput.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]], route_function=route_function, e=e, k=k, m_step=m_step, channel_bandwidth=channel_bandwidth) for ind in tqdm(range(workers))])
    # ray.shutdown()


@ray.remote
def EA_run(N, E, T_C, alpha, topology_num):
    Solutions = []
    for i in range(topology_num):
        ptd = nt.Topology.PTD(N, E, T_C, alpha)
        solution = ptd.run_ga()
        Solutions.append(solution)
        graph = solution['graph']
        nt.Database.insert_graph(graph, "Topology_Data", "ta", node_data=True, use_pickle=True, type="ga")

    return Solutions

def parralel_EA_run(N,E, T_C,alpha,topology_num, workers=1):
    # get indeces [0,1,2] for example for data 0-2

    indeces = nt.Tools.create_start_stop_list(topology_num, workers)
    print(indeces)
    ray.init()
    # Run all the ray instances
    results = ray.get([EA_run.remote(N, E, T_C, alpha, indeces[i + 1] - indeces[i]) for i in range(workers)])





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
    # print(indeces)

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
def main():
    # try:
    graph_list = [1]
    ind=0
    try:
        while len(graph_list) != 0:

            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform-test","T_c",
            #                                                        find_dic={"FF-kSP Capacity":{"$exists":False}, "test data":{"$exists":False}})
            # graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
            #                                                     find_dic={"node_base": "nsfnet"}, node_data=True)
            # graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-scratch", "T_c", find_dic={"ILP-connections":0},
            #                                                      use_pickle=True,
            #                                                      max_count=20000)
            # # print(len(graph_list))
            # # print(np.shape(graph_list))
            # # T_c = [np.ones((len(graph), len(graph))) for graph, _id in graph_list]
            # # [np.fill_diagonal(item, 0) for item in T_c]
            #
            # # graph_list = list(map(lambda x: (x[0], x[1], T_c[graph_list.index(x)]), graph_list))
            # ILP_chromatic(graph_list, collection="topology-paper", max_time=24*3600, workers=int(len(graph_list)), threads=1)

            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
            #                                                      find_dic={"node_base": "nsfnet"},
            #                                                      node_data=True)
            # graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
            #                                                     find_dic={"beta": 8.5, "nodes": 30},
            #                                                     node_data=True, skip=0)
            # parralel_ILP_throughput(graph_list, collection="topology-paper", max_time=24*3600, workers=len(graph_list), threads=1)

            # parralel_heuristic_throughput(graph_list[:], collection="MPNN-uniform-test", workers=int(len(graph_list[:])))
            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform-test", "T_c",
            #                                                     find_dic={"kSP-FF Capacity": {"$exists": False},
            #                                                               "test data": {"$exists": False}})
            # parralel_heuristic_throughput(graph_list[:], collection="MPNN-uniform-test", workers=int(len(graph_list[:])), route_function="kSP-FF")
            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ta",
            #                                                     find_dic={"type": "ga", "ILP-connections":{"$exists":False}})
            # # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real",find_dic={"name":"CONUS"})
            # # parralel_ILP_throughput(graph_list, max_time=48*3600, collection="real", threads=80, workers=1)
            # dataset = nt.Database.read_data_into_pandas("Topology_Data", "ta",
            #                                             find_dic={"node order": "numeric", "gamma": 0.2, "alpha": 2,
            #                                                       "data written": 1})
            # T_C = dataset["traffic_matrix"][0]
            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ta","T_c", find_dic={"type":"ga", "ILP-connections":{"$exists":False}})
            # graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "ta", "T_c", find_dic={"type": "ga",
            #                                                                                             "ILP-connections": 0})
            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform-test", "T_c",find_dic={"ILP-connections Capacity":{"$exists":False}, "test data":{"$exists":False}, "nodes":{"$gte":16}})
            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform-test", "T_c",find_dic={'FF-kSP time': {'$exists': False}, 'test data': {'$exists': False}, 'nodes': {'$gte': 15, '$lte': 30}})
            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform-test", "T_c",
            #                                                     find_dic={'kSP-FF time': {'$exists': False}, 'test data': {'$exists': False}, 'nodes': {'$gte': 15, '$lte': 30}})
            ray.shutdown()
            graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform", "T_c",
                                                                find_dic={
                                                                    # 'FF-kSP Capacity': {'$exists': False},
                                                                          'nodes': {'$gte': 35, '$lte': 100}}, parralel=True, max_count=1000)
            # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ta", "T_c",
            #                                                     find_dic={'ILP-connections': {'$exists': False},
            #                                                               'T_c': {"$exists":True},
            #                                                               "purpose":"scaling"},
            #                                                     parralel=True, max_count=20000)

            # network = nt.Network.OpticalNetwork(graph_list[0][0])
            # T_b = network.demand.create_uniform_bandwidth_normalised()
            # SNR_matrix = network.get_SNR_matrix()
            # T_C = network.demand.bandwidth_2_connections(SNR_matrix,T_b)
            # graph_list = [(graph, _id, T_C) for graph,_id in graph_list]
            # parralel_ILP_connections(graph_list[:], collection="ta", max_time= 4*3600, workers=int(len(
            #     graph_list[:]) / 1), threads=1)
            parralel_heuristic_throughput(graph_list, collection="MPNN-uniform", workers=len(graph_list), route_function="FF-kSP",
                                          e=100,k=100, m_step=500, channel_bandwidth=10e9)

            # exit()
            # from geneticalgorithm import geneticalgorithm as ga

            # parralel_ILP_min_path(graph_list[:],n collection="MPNN-uniform", max_time=2*3600, workers=int(len(
            #     graph_list[:]) / 1), threads=1)
            # exit()

    except Exception as err:
        if err is KeyboardInterrupt:
            exit()
        else:
            print(err)
            main()


    # except Exception as err:
    #     print(err)
    #     if err == MemoryError:
    #         print("MEMORY ERROR!!!")
    #         exit()
    #     else:
    #         exit()
            # ray.shutdown()
            # os.system("source ~/ray_cluster.sh")
            # os.system("restart_cluster")
            # main()
if __name__ == "__main__":

    # scale_throughput()
    # graph, _id = nt.Database.read_topology_dataset_list("Topology_Data","real", find_dic={"name":
    #                                                                                  "30-Node-ONDPBook-Topology_nodes"})[0]
    # print('read graph, calculating chromatic number')
    # rwa, chrom_num = ILP_chromatic_number(graph,max_time=48*3600)
    # nt.Database.update_data_with_id("Topology_Data", "real",_id, newvals={"$set":{"wavelength requirement":chrom_num}})
#     graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"BA",
#                                                                                                      "nodes":14,
#                                                                                                      "ILP Capacity":{
#                                                                                                          "$exists":True}},
# node_data=True, max_count=200)
#     graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"ER",
#                                                                                                      "nodes":14,
#                                                                                                      "ILP Capacity":{
#                                                                                                          "$exists":True}},
#                                                         node_data=True, max_count=200)
#     graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"SBAG",
#                                                                                                      "nodes":14,
#                                                                                                      "ILP Capacity":{
#                                                                                                          "$exists":True}},
#                                                         node_data=True,max_count=200)
    main()
#     ray.shutdown()
#     from NetworkToolkit import Data
#     graph_list = [1]
#     while len(graph_list) > 0:
#         NT = 400
#         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform", max_count=NT,
#                                                             find_dic={"average_ksp_cost": {"$exists": False}},
#                                                             parralel=True)
#         ray.init()
#         GP = Data.GraphProperties.remote()
#
#         funcs = [GP.update_m, GP.update_spanning_tree, \
#                  GP.update_algebraic_connectivity, GP.update_node_variance, GP.update_mean_internodal_distance, \
#                  GP.update_communicability_index, GP.update_comm_traff_ind, GP.update_graph_spectrum, \
#                  GP.update_shortest_path_cost, GP.update_limiting_cut_value]
#         #     ray.init()
#         #     for graph in graph_list:
#         #         for func in funcs:
#         #     #         print(func)
#         #             tempfunc = func
#         #             tasks.append(tempfunc.remote(graph,"MPNN-uniform"))
#         ray.get([func.remote(graph, "MPNN-uniform") for func in funcs for graph in tqdm(graph_list)])
#         #         tasks.append(GP.update_k_shortest_path_cost.remote(graph,"MPNN-uniform",4))
#         #         tasks.append(GP.update_weighted_spectrum_distribution.remote(graph, "MPNN-uniform",4,10))
#         #         tasks.append(GP.update_Dmax_value.remote(graph, "MPNN-uniform",156))
#         #     tasks.append(GP.update_Dmin_value.remote(graph, "MPNN-uniform",156))
#         #     ray.get(tasks)
#         # print(tasks)
#         ray.shutdown()
    # print("waiting to start")
    # for i in tqdm(range(12*3600), desc="starting in"):
    #     time.sleep(1)

    # print("starting now")
    # ILP_throughput(graph_list, max_time=6*3600, collection="topology-paper", workers=int(len(graph_list)/2), threads=1,
    #                num_cpus=2)
    # ILP_chromatic(graph_list, max_time=6*3600)
    # ILP_throughput_scaled(5, 0.1)

