import NetworkToolkit as nt
from tqdm import tqdm
import numpy as np
import ray
import ast
import socket
import os
import networkx as nx
import time
import datetime
import argparse
import MPNN

parser = argparse.ArgumentParser(description='Optical Network Simulator Script')
parser.add_argument('-m', action='store', type=int, default=3000, help="How much memory to allocate to processes")
parser.add_argument('-cpu', action='store', type=int, default=1, help="How many cpus to use")
parser.add_argument('-mc', action='store', type=int, default=1000, help="How many samples to read")
parser.add_argument('--sleep', action='store', type=float, default=1000, help="How long to wait")
args = parser.parse_args()


@ray.remote
class ScaledThroughput:
    def calculate_T_UL_scaled(self, graph_data, scale=1):
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

    def calculate_scaled_throughput_data(self, dataframe, start=0.01, end=1, step=0.02, graph_type="SBAG",
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
                dict1 = {"graph type": graph_type, "scale": i, "throughput data": throughput, "data type": "scaled",
                         "alpha": alpha}

                write_dic = {**dict1, **save_dic}
                nt.Database.insert_data("Topology_Data", "ECOC", write_dic)


def ILP_chromatic_number(graph, max_time=1000):
    network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
    rwa_assignment, min_wave = network.rwa.static_ILP(min_wave=True, max_time=max_time)
    return rwa_assignment, min_wave


@ray.remote(num_cpus=1)
def ILP_uniform_connections(graph_list=None, T_c=None, max_time=1000, collection=None, db="Topology_Data", threads=10):
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
                                                                                             k=10, _id=_id,
                                                                                             threads=threads)
        rwa_assignment = nt.Tools.write_database_dict(rwa_assignment)
        if T_c is not None:
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-connections "
                                                                                   "RWA": rwa_assignment,
                                                                                   "ILP-connections": objective_value}})
        else:
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-uniform-connections "
                                                                                   "RWA": rwa_assignment,
                                                                                   "ILP-uniform-connections": objective_value}})


# , memory=5000*1024*1024
@ray.remote(num_cpus=args.cpu, memory=args.m * 1024 * 1024)
def ILP_connections(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                    threads=10, throughput=False, pb_actor=None, e=20, k=20, fibre_num=1, bandwidth=32e9):
    if socket.gethostname() == "MacBook-Pro":
        print("Laptop Run")
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
        node_file_dir = "/Users/robin/OneDrive/OneDrive - University College London/Code/PycharmProjects/networktoolbox/scripts/nodefiles"
    else:
        os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
            socket.gethostname().split('.')[0])
        node_file_dir = "/scratch/datasets/gurobi/nodefiles"
    for graph, _id, T_c in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

        assert type(graph) == nx.classes.graph.Graph
        assert nx.is_connected(graph) == True
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=bandwidth, fibre_num=fibre_num)
        print("fibres being used: {}".format(fibre_num))
        # try:
        time_start = time.perf_counter()
        data = network.rwa.maximise_connection_demand(T_c=T_c, max_time=max_time,
                                                      e=e,
                                                      k=k, _id=_id,
                                                      threads=threads,
                                                      node_file_dir=node_file_dir,
                                                      node_file_start=0.001,
                                                      emphasis=0,
                                                      max_solutions=100)
        time_taken = time.perf_counter() - time_start
        rwa_assignment = nt.Tools.write_database_dict(data["rwa"])
        print("starting rwa conversion")
        rwa_assignment = nt.Tools.single_to_multi_fibre_rwa(rwa_assignment, network.routing_channels,
                                                            network.channels)
        print(rwa_assignment)
        if fibre_num > 1 and throughput:

            rwa_assignment = nt.Tools.single_to_multi_fibre_rwa(rwa_assignment, network.routing_channels,
                                                                network.channels)
            print(rwa_assignment)
            max_capacity = 0
            rwa_write = []
            for i in range(fibre_num):
                print(i)
                network.physical_layer.add_wavelengths_to_links(rwa_assignment[i])
                network.physical_layer.add_non_linear_NSR_to_links()
                max_capacity += network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment[i])[0]
                rwa_write.append(nt.Tools.write_database_dict(rwa_assignment[i]))
            print("throughput: {} Tbps".format(max_capacity))
        elif fibre_num == 1 and throughput:
            rwa_write = nt.Tools.write_database_dict(rwa_assignment)
            network.physical_layer.add_uniform_launch_power_to_links(network.channels)
            network.physical_layer.add_wavelengths_to_links(data["rwa"])
            network.physical_layer.add_non_linear_NSR_to_links()
            max_capacity = network.physical_layer.get_lightpath_capacities_PLI(data["rwa"])[0]
            # node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
        else:
            rwa_write = nt.Tools.write_database_dict(rwa_assignment)

        nt.Database.update_data_with_id(db, collection, _id,
                                        newvals={"$set": {"ILP-connections RWA": rwa_write,
                                                          "ILP-connections": data["objective"],
                                                          "ILP-connections gap": data["gap"],
                                                          "ILP-connections data written": 1,
                                                          "data written ga": 1,
                                                          "ILP-connections time": time_taken,
                                                          "ILP-connections status": data[
                                                              "status"].value,
                                                          "ILP-connections timestamp": datetime.datetime.utcnow(),
                                                          "ILP-connections e": e,
                                                          "ILP-connections k": k,
                                                          "ILP-connections emphasis": 0,
                                                          "ILP-connections max time": max_time,
                                                          "ILP-connections channel bandwidth": network.channel_bandwidth,
                                                          "ILP-connections channels": network.channels,
                                                          "ILP-connections routing channels": network.routing_channels,
                                                          }})

        if throughput:
        #
        #     network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        #     network.physical_layer.add_wavelengths_to_links(data["rwa"])
        #     network.physical_layer.add_non_linear_NSR_to_links()
        #     max_capacity = network.physical_layer.get_lightpath_capacities_PLI(data["rwa"])
        #     node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}

            nt.Database.update_data_with_id(db, collection, _id,
                                            newvals={"$set": {"ILP-connections RWA": rwa_write,
                                                              "ILP-connections": data["objective"],
                                                              "ILP-connections gap": data["gap"],
                                                              "ILP-connections throughput written": 1,
                                                              "ILP-connections status": data["status"].value,
                                                              # "ILP-connections node pair capacities": node_pair_capacities,
                                                              "ILP-connections Capacity": max_capacity,
                                                              "ILP-connections timestamp": datetime.datetime.utcnow(),
                                                              "ILP-connections e": e,
                                                              "ILP-connections k": k,
                                                              "ILP-connections emphasis": 0,
                                                              "ILP-connections max time": max_time,
                                                              "ILP-connections channel bandwidth": network.channel_bandwidth,
                                                              "ILP-connections channels": network.channels,
                                                              "ILP-connections routing channels": network.routing_channels,
                                                              "ILP-connections launch power type": "uniform",
                                                              "ILP-connections launch power": 0,
                                                              "ILP-connections launch power units": "dBm",
                                                              "ILP-connections fibre number": fibre_num
                                                              }})
            print(datetime.datetime.utcnow())
            nt.Database.update_data_with_id(db, collection, _id,
                                            newvals={
                                                "$set": {"ILP-connections timestamp": datetime.datetime.utcnow()}})

        # except Exception as error:
        #     print("error happened in simulation base: {}".format(error))
        #     nt.Database.update_data_with_id(db, collection, _id,
        #                                     newvals={"$set": {"error occurred": 1,
        #                                                       "error": str(error),
        #                                                       "error host": str(socket.gethostname())}})
        #     continue
            pb_actor.update.remote(1)


@ray.remote(num_cpus=1, memory=3000 * 1024 * 1024)
def ILP_min_path(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                 threads=10, throughput=False):
    if socket.gethostname() == "MacBook-Pro":
        print("Laptop Run")
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
        node_file_dir = "/Users/robin/OneDrive/OneDrive - University College London/Code/PycharmProjects/networktoolbox/scripts/nodefiles"
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

            nt.Database.update_data_with_id(db, collection, _id,
                                            newvals={"$set": {"ILP-connections RWA": rwa_assignment,
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

                nt.Database.update_data_with_id(db, collection, _id,
                                                newvals={"$set": {"ILP-connections RWA": rwa_assignment,
                                                                  "ILP-connections": data["objective"],
                                                                  "ILP-connections gap": data["gap"],
                                                                  "ILP-connections throughput written": 1,
                                                                  "ILP-connections status": data["status"].value,
                                                                  "ILP-connections node pair capacities": node_pair_capacities,
                                                                  "ILP-connections Capacity": max_capacity[0]}})
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
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
        max_time=max_time, threads=threads) for ind in range(workers)])


def parralel_ILP_connections(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data",
                             workers=50, threads=10, port=6379, hostname="128.40.43.93", fibre_num=1):
    print(len(graph_list))
    ray.shutdown()
    ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', dashboard_port=8265)
    # ray.init()
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    pb = nt.Tools.ProgressBar(workers)
    actor = pb.actor
    results = [ILP_connections.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
        max_time=max_time, threads=threads, throughput=True, pb_actor=actor, fibre_num=fibre_num) for ind in range(workers)]
    pb.print_until_done()
    results = ray.get(results)
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


@ray.remote(num_cpus=1, memory=3000 * 1024 * 1024)
def ILP_wave_req(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                 threads=10, e=10, k=10):
    os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
        socket.gethostname().split('.')[0])
    for graph, _id in graph_list:
        try:
            graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
            assert type(graph) == nx.classes.graph.Graph
            network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
            time_start = time.perf_counter()
            print("starting wavelength requirement ILP")
            data = network.rwa.static_ILP(min_wave=True, max_time=max_time, e=e, k=k, threads=threads,
                                          node_file_start=0.01)
            time_taken = time.perf_counter() - time_start

            rwa_assignment = nt.Tools.write_database_dict(data["rwa"])
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-chromatic RWA": rwa_assignment,
                                                                                   "wavelength requirement": data[
                                                                                       "objective"],
                                                                                   "ILP-chromatic status": data[
                                                                                       "status"].value,
                                                                                   "ILP-chromatic gap": data["gap"],
                                                                                   "ILP-chromatic time": time_taken,
                                                                                   "ILP-chromatic data written": 1,
                                                                                   "ILP-chromatic k":k,
                                                                                   "ILP-chromatic e":e,
                                                                                   "ILP-chromatic timestamp": datetime.datetime.utcnow()}})
        except Exception as err:
            print(err)


def ILP_chromatic(graph_list, max_time=1000, collection="topology-paper", db="Topology_Data", workers=10, threads=10,
                  k=10, e=10, local=False, port=6379):
    print(len(graph_list))
    if local:
        ray.init()
    else:
        ray.init(address='128.40.41.48:{}'.format(port), _redis_password='5241590000000000', dashboard_port=8265)
    # NetworkSimulator = (ray.remote(num_cpus=1, num_gpus=0))(nt.NetworkSimulator.NetworkSimulator)
    # simulators = [NetworkSimulator.remote() for i in range(workers)]
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_wave_req.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
        max_time=max_time, threads=threads, k=k, e=e) for ind in range(workers)])


@ray.remote(num_cpus=40, memory=8000 * 1024 * 1024)
def ILP_throughput(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                   threads=10, channel_bandwidth=16e9, e=0, k=20):
    if socket.gethostname() == "MacBook-Pro":
        print("Laptop Run")
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/y/gurobi.lic"
        node_file_dir = "/Users/robin/OneDrive/OneDrive - University College London/Code/PycharmProjects/networktoolbox/scripts/nodefiles"
    else:
        os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi-licences/{}/gurobi.lic".format(
            socket.gethostname().split('.')[0])
        node_file_dir = "/scratch/datasets/gurobi/nodefiles"
    for graph, _id in graph_list:
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=channel_bandwidth)
        data = network.rwa.maximise_uniform_bandwidth_demand(max_time=max_time,
                                                             e=e,
                                                             k=k, _id=_id,
                                                             threads=threads,
                                                             node_file_dir=node_file_dir,
                                                             node_file_start=0.01,
                                                             c_type="I",
                                                             capacity_constraint=False,
                                                             verbose=0,
                                                             emphasis=2)
        try:
            if data["objective"] == 0:
                max_capacity = [0]
                node_pair_capacities = 0
                rwa_assignment = None
            else:
                network.physical_layer.add_uniform_launch_power_to_links(network.channels)
                network.physical_layer.add_wavelengths_to_links(data["rwa"])
                network.physical_layer.add_non_linear_NSR_to_links(channels_full=network.channels,
                                                                   channel_bandwidth=network.channel_bandwidth)
                max_capacity = network.physical_layer.get_lightpath_capacities_PLI(data["rwa"])
                node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
                rwa_assignment = nt.Tools.write_database_dict(data["rwa"])
            nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"ILP-throughput RWA": rwa_assignment,
                                                                                   "ILP-throughput": data["objective"],
                                                                                   "ILP-capacity": max_capacity[0],
                                                                                   "ILP node pair capacities": node_pair_capacities,
                                                                                   "data written": 1,
                                                                                   "ILP-throughput status": data[
                                                                                       "status"].value,
                                                                                   "ILP-throughput gap": data["gap"],
                                                                                   "e": e,
                                                                                   "k": k,
                                                                                   "channel number": network.channels,
                                                                                   "channel bandwidth": network.channel_bandwidth,
                                                                                   "fibre number": network.routing_channels,
                                                                                   "timestamp Capacity": datetime.datetime.utcnow(),
                                                                                   "capacity constraint": "false",
                                                                                   "ILP emphasis": 2
                                                                                   }})

        except Exception as err:
            print(err)


def parralel_ILP_throughput(graph_list, max_time=1000, collection=None, db="Topology_Data", workers=50, num_cpus=2,
                            threads=4, e=0, k=20, port=6379):
    # ray.init(
    ray.init(address='128.40.41.48:{}'.format(port), _redis_password='5241590000000000', ignore_reinit_error=True)
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([ILP_throughput.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
        max_time=max_time, threads=threads, e=e, k=k) for ind in range(workers)])
    ray.shutdown()


@ray.remote(num_cpus=args.cpu, memory=args.m * 1024 * 1024)
def heuristic_throughput(graph_list=None, collection=None, db="Topology_Data", e=10, k=10, route_function="FF-kSP",
                         m_step=100, max_count=10, channel_bandwidth=16e9, m_start=0, fibre_num=1, pb_actor=None):
    for graph, _id, T_c in graph_list:
        # try:
        # nt.Database.update_data_with_id(db, collection, _id,
        #                                 newvals={"$set": {"processing": 1}})
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        assert nx.is_connected(graph) is True
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=channel_bandwidth, routing_func=route_function,
                                            fibre_num=fibre_num)
        # print("channels: {}".format(network.channels))
        # print("routing channels: {}".format(network.routing_channels))
        rwa_assignment = False
        time_start = time.perf_counter()
        M = m_start
        # print("starting routing {}".format(route_function))
        demand_matrix_old = np.zeros((len(graph), len(graph)))
        rwa_active = None
        success = False
        for i in range(max_count):
            while rwa_assignment != True:
                M += m_step
                # if time.perf_counter() - time_start > 1000:
                #     print(len(network.graph))
                #     print(nx.is_connected(network.graph))
                #     print(M)
                demand_matrix_new = np.ceil(np.array(T_c) * M)
                if route_function == "FF-kSP" or "kSP-FF":
                    # print("M: {}".format(M))
                    if not success:
                        alternate_demand = demand_matrix_new - demand_matrix_old
                        connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                        rwa_assignment = network.route(demand_matrix_new - demand_matrix_old, e=e, k=k,
                                                       connection_pairs=connection_pairs)

                    elif (demand_matrix_new - demand_matrix_old).sum() > 0:
                        alternate_demand = demand_matrix_new - demand_matrix_old
                        connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                        rwa_assignment = network.route(demand_matrix_new - demand_matrix_old, e=e, k=k,
                                                       rwa_assignment_previous=rwa_assignment)
                    # print(demand_matrix_new-demand_matrix_old
                    # print(rwa_assignment)
                    if rwa_assignment != True:
                        rwa_active = rwa_assignment
                        demand_matrix_old = demand_matrix_new
                        success = True

                else:
                    alternate_demand = demand_matrix_new - demand_matrix_old
                    connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                    rwa_assignment = network.route(demand_matrix_new, e=e, k=k, connection_pairs=connection_pairs)
            # print(rwa_assignment)
            # print("M: {}".format(M))
            if int(M) > 1:
                M -= m_step

            demand_matrix = np.ceil(np.array(T_c) * M)
            if route_function == "FF-kSP" or "kSP-FF":
                rwa_assignment = rwa_active

            else:
                rwa_assignment = network.route(demand_matrix_new, e=e, k=k)
            if int(M) == 1 and rwa_assignment == False:
                break
            elif int(M) == 1 and rwa_assignment == True:
                print("Cant route base demand")
                print("nodes: {}".format(len(graph)))
                print("edges: {}".format(len(list(graph.edges))))
                print("graph is connected? :{}".format(nx.is_connected(graph)))
                print("min degree: {}".format(min([degree for node, degree in nx.degree(graph)])))
            m_step /= 2
            m_step = np.ceil(m_step)


        assert rwa_assignment != True
        assert rwa_assignment is not None
        # print("hello")
        # print("objective value: {}".format(M))
        time_taken = time.perf_counter() - time_start
        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        # print("num fibre: {}".format(fibre_num))

        if fibre_num > 1:

            rwa_assignment = nt.Tools.single_to_multi_fibre_rwa(rwa_assignment, network.routing_channels,
                                                                network.channels)
            # print(rwa_assignment)
            throughput = 0
            rwa_write = []
            for i in range(fibre_num):
                network.physical_layer.add_wavelengths_to_links(rwa_assignment[i])
                network.physical_layer.add_non_linear_NSR_to_links()
                throughput += network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment[i])[0]
                rwa_write.append(nt.Tools.write_database_dict(rwa_assignment[i]))
            # print("throughput: {} Tbps".format(throughput))
        else:
            network.physical_layer.add_wavelengths_to_links(rwa_assignment)
            network.physical_layer.add_non_linear_NSR_to_links()
            throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)[0]
            # node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
            rwa_write = nt.Tools.write_database_dict(rwa_assignment)
        nt.Database.update_data_with_id(db, collection, _id,
                                        newvals={"$set": {"{} RWA".format(route_function): rwa_write,
                                                          "{}-connections".format(route_function): M,
                                                          "{} throughput written".format(route_function): 1,
                                                          # "{} node pair capacities".format(route_function): node_pair_capacities,
                                                          "{} sequential loading iterations".format(route_function):M,
                                                          "{} time".format(route_function): time_taken,
                                                          "{} Capacity".format(route_function): throughput,
                                                          "{} fibre number".format(route_function): fibre_num,
                                                          "{} channel number".format(route_function): network.channels,
                                                          "{} channel bandwidth".format(
                                                              route_function): network.channel_bandwidth,
                                                          "{} e".format(route_function): e,
                                                          "{} k".format(route_function): k,
                                                          "{} m step".format(route_function): m_step,
                                                          "{} max repeat".format(route_function): max_count,
                                                          "{} m start".format(route_function): m_start,
                                                          "{} processing".format(route_function): 0,
                                                          "{} timestamp".format(
                                                              route_function): datetime.datetime.utcnow(),
                                                          "{} launch power type".format(route_function): "uniform",
                                                          "{} launch power".format(route_function): 0,
                                                          "{} launch power units".format(route_function): "dBm"
                                                          }})

        pb_actor.update.remote(1)
        # except Exception as err:
        #     print("error occured in routing: {}".format(err))


def parralel_heuristic_throughput(graph_list, collection=None, db="Topology_Data", workers=50, route_function="FF-kSP",
                                  e=10, k=10, m_step=100,
                                  channel_bandwidth=16e9, max_count=10, m_start=0, port=6379, hostname="128.40.43.93",
                                  fibre_num=1):
    ray.shutdown()
    # ray.init()
    if hostname is not None:
        ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)
    else:
        ray.init()
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    pb = nt.Tools.ProgressBar(workers)
    actor = pb.actor
    print(indeces)
    # for graph, _id, T_c in graph_list:
    #     nt.Database.update_data_with_id(db, collection, _id,
    #                                     newvals={"$set": {"processing": 1}})
    results = [heuristic_throughput.remote(db=db,
                                           collection=collection,
                                           graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
                                           route_function=route_function, e=e, k=k, m_step=m_step,
                                           channel_bandwidth=channel_bandwidth,
                                           max_count=max_count, m_start=m_start, fibre_num=fibre_num, pb_actor=actor)
               for ind in range(workers)]
    pb.print_until_done()
    results = ray.get(results)
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


def parralel_EA_run(N, E, T_C, alpha, topology_num, workers=1):
    # get indeces [0,1,2] for example for data 0-2

    indeces = nt.Tools.create_start_stop_list(topology_num, workers)
    print(indeces)
    ray.init()
    # Run all the ray instances
    results = ray.get([EA_run.remote(N, E, T_C, alpha, indeces[i + 1] - indeces[i]) for i in range(workers)])


@ray.remote
def dwc_select_graph_generation(real_graph, E, T_C, alpha, _alpha, graph_function=None, graph_function_args=None,
                                actor=None,
                                combined_DWC=False, fibre_limit=None):
    # real_graph, E, T_C, alpha, _alpha, graph_function=None, graph_function_args=None,
    # ind=None
    # # print("starting graph generation")
    # #
    # #
    #
    topology_generator = nt.Topology.Topology()
    if graph_function is None:
        if fibre_limit is None:
            new_graph, node_order = topology_generator.create_real_based_grid_graph(real_graph,
                                                                        E,
                                                                        database_name="Topology_Data",
                                                                        sequential_adding=True,
                                                                        random_start_node=True,
                                                                        overshoot=True,
                                                                        remove_C1_C2_edges=True,
                                                                        SBAG=True,
                                                                        alpha=_alpha,
                                                                        max_degree=100,
                                                                        plot_sequential_graphs=False,
                                                                        print_probs=False,
                                                                        ignore_constraints=True,
                                                                        return_sequence=True
                                                                        )
        elif fibre_limit is not None:
            total_fibre = 1e13
            _break = 0
            original_E = E
            while total_fibre >= fibre_limit:
                if E <= len(real_graph):
                    # print("E: {}".format(E))
                    # print("total fibre distance: {}".format(total_fibre))
                    E = original_E
                new_graph, node_order = topology_generator.create_real_based_grid_graph(real_graph,
                                                                            E,
                                                                            database_name="Topology_Data",
                                                                            sequential_adding=True,
                                                                            random_start_node=True,
                                                                            overshoot=True,
                                                                            remove_C1_C2_edges=True,
                                                                            SBAG=True,
                                                                            alpha=_alpha,
                                                                            max_degree=100,
                                                                            plot_sequential_graphs=False,
                                                                            print_probs=False,
                                                                            ignore_constraints=True,
                                                                            return_sequence=True
                                                                            )
                total_fibre = nt.Tools.total_fibre_length(new_graph)
                # print("total fibre: {}".format(total_fibre))
                # print("fibre limit: {}".format(fibre_limit))
                if total_fibre < fibre_limit and _break == 0:
                    total_fibre = 1e13
                    E += 1
                    # print("E: {}".format(E))
                    # print(total_fibre)
                elif _break == 0 and total_fibre > fibre_limit:
                    E -= 1
                    _break = 1
                elif _break == 1 and total_fibre > fibre_limit:
                    E -=1
                else:
                    # print("final E: {}".format(E))
                    # print("final total fibre: {}".format(total_fibre))
                    # print("fibre limit: {}".format(fibre_limit))
                    break



    else:
        # print("creating graphs with {}".format(str(graph_function)))
        new_graph = graph_function(**graph_function_args)

    inverse_DWC = 1e13
    # print("starting inverse checks")
    if fibre_limit is None and nx.is_connected(new_graph) == E and topology_generator.check_bridges(
            new_graph) and topology_generator.check_min_degree(new_graph):
        # inverse_DWC = nt.Tools.get_demand_weighted_cost([[new_graph, 1]], [T_C], alpha)[0]
        if not combined_DWC:
            inverse_DWC = 1 / nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
        elif combined_DWC:
            alpha_DWC = E / len(real_graph) / (len(real_graph) - 1) * 2
            DWC_structure = nt.Tools.get_demand_weighted_cost([[new_graph, 0]], [T_C], alpha, penalty_num=1000)[0]
            DWC_distance = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
            DWC = alpha * DWC_distance + (1 - alpha) * DWC_structure
            inverse_DWC = 1 / DWC
    elif fibre_limit is not None and topology_generator.check_bridges(new_graph):
        if not combined_DWC:
            inverse_DWC = 1 / nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
        elif combined_DWC:
            alpha_DWC = E / len(real_graph) / (len(real_graph) - 1) * 2
            DWC_structure = nt.Tools.get_demand_weighted_cost([[new_graph, 0]], [T_C], alpha, penalty_num=1000)[0]
            DWC_distance = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
            DWC = alpha * DWC_distance + (1 - alpha) * DWC_structure
            inverse_DWC = 1 / DWC
    else:
        print("Failed")
        exit()



    # print("finished with {}".format(str(graph_function)))
    # new_graph = None
    # inverse_DWC = None

    actor.update.remote(1)

    return new_graph, inverse_DWC, node_order


def parralel_graph_generation_DWC(N, E, grid_graph, T_c, workers=1, hostname="128.40.43.93", port=6379, alpha=[1],
                                  _alpha=5,
                                  graph_num=200, graph_function=None, graph_function_args=None, fibre_limit=None,
                                  combined_DWC=False):
    # ray.init()
    # iterations = np.int(np.floor(N / workers))
    # DWC_best = 1e13
    # graph_best = None
    graphs = []
    # results = []
    # for i in tqdm(range(N)):
    # print(len(results))
    ray.shutdown()
    # ray.init()
    ray.init(address="{}:{}".format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)

    # Invocations of Ray remote functions happen in parallel.
    # All computation is performed in the background, driven by Ray's internal event loop.
    # for _ in tqdm(range(100000)):
    #     # This doesn't block.
    #     slow_function.remote()
    # tasks = [slow_function.remote() for i in tqdm(range(1000000))]
    # ray.get(tasks)
    pb = nt.Tools.ProgressBar(N, description="creating graphs")
    actor = pb.actor
    tasks = [dwc_select_graph_generation.remote(grid_graph, E, T_c, alpha, _alpha, graph_function=graph_function,
                                                graph_function_args=graph_function_args, actor=actor, combined_DWC=combined_DWC,
                                                fibre_limit=fibre_limit) for i in tqdm(range(N), desc="assigning tasks")]
    pb.print_until_done()
    # print(len(tasks))
    # grid_graph, E, T_c, alpha, _alpha, graph_function=graph_function,
    # graph_function_args=graph_function_args, ind=i

    # graph, inverese_DWC = min(results, key=lambda item: item[1])
    results = ray.get(tasks)
    # print(len(results))
    # print(results[0])
    # exit()
    for graph, inverese_DWC, node_order in results:
        if len(graphs) < graph_num:
            graphs.append((graph, inverese_DWC, node_order))
        else:
            graph_worst, inverse_DWC_worst, node_order_worst = min(graphs, key=lambda item: item[1])
            # index = graphs
            index = graphs.index((graph_worst, inverse_DWC_worst, node_order_worst))
            if inverese_DWC > inverse_DWC_worst:
                graphs[index] = (graph, inverese_DWC, node_order)
        # if inverese_DWC < DWC_best:
        #     DWC_best = inverese_DWC
        #     graph_best = graph
    return graphs


@ray.remote
def random_select_graph_generation(grid_graph, E, T_C, alpha=[1], _alpha=5, graph_function=None,
                                   graph_function_args=None, gamma=None, actor=None, fibre_limit=None, combined_DWC=False):
    topology_generator = nt.Topology.Topology()
    if graph_function is None:
        if fibre_limit is None:
            new_graph, node_order = topology_generator.create_real_based_grid_graph(grid_graph,
                                                                        E,
                                                                        database_name="Topology_Data",
                                                                        sequential_adding=True,
                                                                        random_start_node=True,
                                                                        overshoot=True,
                                                                        remove_C1_C2_edges=True,
                                                                        SBAG=True,
                                                                        alpha=_alpha,
                                                                        max_degree=100,
                                                                        plot_sequential_graphs=False,
                                                                        print_probs=False,
                                                                        ignore_constraints=True,
                                                                        return_sequence=True
                                                                        )
        elif fibre_limit is not None:
            total_fibre = 1e13
            _break=0
            E_original = E

            while total_fibre >= fibre_limit:
                if E <= len(grid_graph):
                    E = E_original
                new_graph, node_order = topology_generator.create_real_based_grid_graph(grid_graph,
                                                                            E,
                                                                            database_name="Topology_Data",
                                                                            sequential_adding=True,
                                                                            random_start_node=True,
                                                                            overshoot=True,
                                                                            remove_C1_C2_edges=True,
                                                                            SBAG=True,
                                                                            alpha=_alpha,
                                                                            max_degree=100,
                                                                            plot_sequential_graphs=False,
                                                                            print_probs=False,
                                                                            ignore_constraints=True,
                                                                            return_sequence=True
                                                                            )
                total_fibre = nt.Tools.total_fibre_length(new_graph)
                if total_fibre < fibre_limit and _break == 0:
                    total_fibre = 1e13
                    E += 1
                elif _break == 0 and total_fibre > fibre_limit:
                    E -= 1
                    _break = 1
                elif _break == 1 and total_fibre > fibre_limit:
                    E -=1
                else:
                    break

    else:
        new_graph = graph_function(**graph_function_args)

    i = 0
    while nx.is_connected(new_graph) == False or topology_generator.check_bridges(
            new_graph) == False or topology_generator.check_min_degree(new_graph) == False:
        # print("graph is connected: {}".format(nx.is_connected(new_graph)))
        # print("graph is bi-connected: {}".format(topology_generator.check_bridges(
        #     new_graph)))
        # print("graph has min degree: {}".format(topology_generator.check_min_degree(new_graph)))
        # print(len(new_graph))
        # print(new_graph.number_of_edges())

        if graph_function is None:
            if fibre_limit is None:
                new_graph, node_order = topology_generator.create_real_based_grid_graph(grid_graph,
                                                                            E,
                                                                            database_name="Topology_Data",
                                                                            sequential_adding=True,
                                                                            random_start_node=True,
                                                                            overshoot=True,
                                                                            remove_C1_C2_edges=True,
                                                                            SBAG=True,
                                                                            alpha=_alpha,
                                                                            max_degree=100,
                                                                            plot_sequential_graphs=False,
                                                                            print_probs=False,
                                                                            ignore_constraints=True,
                                                                            return_sequence=True
                                                                            )
            elif fibre_limit is not None:
                total_fibre = 1e13
                while total_fibre >= fibre_limit:
                    new_graph, node_order = topology_generator.create_real_based_grid_graph(grid_graph,
                                                                                E,
                                                                                database_name="Topology_Data",
                                                                                sequential_adding=True,
                                                                                random_start_node=True,
                                                                                overshoot=True,
                                                                                remove_C1_C2_edges=True,
                                                                                SBAG=True,
                                                                                alpha=_alpha,
                                                                                max_degree=100,
                                                                                plot_sequential_graphs=False,
                                                                                print_probs=False,
                                                                                ignore_constraints=True,
                                                                                return_sequence=True
                                                                                )
                    total_fibre = nt.Tools.total_fibre_length(new_graph)

        else:
            new_graph = graph_function(**graph_function_args)

    if actor is not None:
        actor.update.remote(1)
    # print("done")
    # DWC = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
    if not combined_DWC:
        DWC = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
    elif combined_DWC:
        alpha_DWC = E/len(grid_graph)/(len(grid_graph)-1)*2
        DWC_structure = nt.Tools.get_demand_weighted_cost([[new_graph, 0]], [T_C], alpha, penalty_num=1000)[0]
        DWC_distance = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
        # print("alpha: {} DWC_distance: {} DWC_structure: {}".format(alpha, DWC_distance, DWC_structure))
        DWC = alpha_DWC * DWC_distance + (1 - alpha_DWC) * DWC_structure

    return new_graph, DWC, T_C, gamma, node_order


def parralel_random_select_graph_generation(graph_list, E=140, alpha=[1], collection=None, db="Topology_Data",
                                            write=False,
                                            _type=None, purpose=None, notes=None, graph_num=200, gamma_t=[0.0],
                                            hostname="", T_C_list=None, fibre_limit=None, combined_DWC=False,
                                            port=0, graph_function=None, graph_function_args=None, _alpha=5, **kwargs):
    grid_graph = graph_list[0][0]
    if T_C_list is None:
        T_C_list = []
        for gamma in gamma_t:
            network = nt.Network.OpticalNetwork(grid_graph, channel_bandwidth=16e9)
            T_C = network.demand.create_skewed_demand(network.graph, gamma)
            T_C_list.append((T_C, gamma))

    # E = 120
    # alpha = [1]
    topology_num = 1
    pb = nt.Tools.ProgressBar(graph_num * len(T_C_list), description="creating random graphs")
    actor = pb.actor
    tasks = []
    for T_C, gamma in T_C_list:
        for i in tqdm(range(graph_num), desc="creating tasks for gamma:{}".format(gamma)):
            tasks.append(
                random_select_graph_generation.remote(grid_graph, E, T_C, gamma=gamma, alpha=alpha, _alpha=_alpha,
                                                      graph_function=graph_function,
                                                      graph_function_args=graph_function_args,
                                                      actor=actor, fibre_limit=fibre_limit, combined_DWC=combined_DWC))
    results = ray.get(tasks)
    if write:
        for graph, DWC, T_C, gamma, node_order in tqdm(results, desc="writing graphs to database"):
            # print("node order: {}".format(node_order))
            # print(type(nt.Tools.total_fibre_length(graph)))
            # print(type(DWC))
            # print(type(gamma))
            node_order = [int(item) for item in node_order]
            if type(T_C) == np.ndarray:
                T_C = T_C.tolist()
            nt.Database.insert_graph(graph, db, collection, node_data=True, use_pickle=True,
                                     # ,
                                     type=_type, purpose=purpose, T_c=T_C, DWC=DWC, notes=notes,
                                     timestamp=datetime.datetime.utcnow(), gamma=gamma, graph_num=graph_num,
                                     graph_function=str(graph_function), fibre_limit=fibre_limit,
                                     combined_DWC=combined_DWC, total_fibre_length=nt.Tools.total_fibre_length(graph),
                                     graph_function_args=str(graph_function_args), node_order=node_order,
                                     **kwargs)


def DWC_select_graph_generation(graph_list, E=140, alpha=[1], collection=None, db="Topology_Data", write=False,
                                _type=None, purpose=None, notes=None, graph_num=500000, gamma_t=[0.0], hostname="",
                                T_C_list=None, final_graph_num=200,
                                port=0, graph_function=None, graph_function_args=None, _alpha=5, combined_DWC=False,
                                fibre_limit=None, **kwargs):
    grid_graph = graph_list[0][0]

    if T_C_list is None:

        T_C_list = []
        for gamma in gamma_t:
            network = nt.Network.OpticalNetwork(grid_graph, channel_bandwidth=16e9)
            T_C = network.demand.create_skewed_demand(network.graph, gamma)
            T_C_list.append((T_C, gamma))


    # E = 120
    # alpha = [1]

    for T_C, gamma in T_C_list:
        # for item in T_C:
        #     for row in item:
        #         print(row)
        #         assert type(row) != np.int64
            # print(item)
        # print(type(T_C))
        dwc_select_graphs = parralel_graph_generation_DWC(graph_num, E, grid_graph, T_C, workers=1000,
                                                          hostname=hostname,
                                                          port=port,
                                                          alpha=alpha, _alpha=_alpha, graph_num=final_graph_num,
                                                          graph_function=graph_function,
                                                          graph_function_args=graph_function_args,
                                                          fibre_limit=fibre_limit, combined_DWC=combined_DWC)

        if write:
            for graph, inverse_DWC, node_order in tqdm(dwc_select_graphs, desc="writing graphs"):
                items = [graph, db, collection,
                                         _type, purpose, T_C, inverse_DWC, notes,
                                         datetime.datetime.utcnow(), gamma, graph_num,
                                         str(graph_function), fibre_limit,
                                         combined_DWC, nt.Tools.total_fibre_length(graph),
                                         str(graph_function_args), node_order] + [value for key, value in kwargs.items()]
                for item in items:
                    if type(item) == np.int64:
                        print(item)
                        exit()

                if type(T_C) == np.ndarray:
                    T_C = T_C.tolist()

                nt.Database.insert_graph(graph, db, collection, node_data=True, use_pickle=True,
                                         type=_type, purpose=purpose, T_c=T_C, DWC=1 / inverse_DWC, notes=notes,
                                         timestamp=datetime.datetime.utcnow(), gamma=float(gamma), graph_num=graph_num,
                                         graph_function=str(graph_function), fibre_limit=fibre_limit,
                                         combined_DWC=combined_DWC, total_fibre_length=nt.Tools.total_fibre_length(graph),
                                         graph_function_args=str(graph_function_args), node_order=node_order,
                                         node_sequence="sequential",
                                         **kwargs)


def ILP_throughput_scaled(alpha, scale):
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ECOC", find_dic={"ILP Capacity": {
        "$exists": True}, "alpha": alpha},
                                                        node_data=True)
    # {"ILP Capacity":{"$exists":False}

    new_graph_list = []
    for graph, _id in graph_list:
        for s, d in graph.edges:
            graph[s][d]["weight"] = np.ceil(scale * graph[s][d]["weight"])

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
            {"function": test_1, "args": graph_list, "kwargs": {"max_time": 6 * 3600, "workers": 600}})
        task_queue.put(
            {"function": test_2, "args": graph_list, "kwargs": {"max_time": 48 * 3600,
                                                                "collection": "topology-paper",
                                                                "workers": 600}})
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


def main(route_function=None, collection=None, query=None, hostname=None, port=None,
         heuristic=False, ILP=False, max_time=3 * 3600, max_count=100000, fibre_num=1,
         desc=None, skip=0, ILP_threads=1):
    # try:
    graph_list = [1]
    ind = 0
    # try:

    # while len(graph_list) != 0:
    ray.shutdown()

    # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform-55-100-test","T_c",
    #                                                     find_dic={
    #                                                         # "purpose": "ga-analysis",
    #                                                         # "ILP-connections":{"$exists":False},
    #                                                         # "data":"new alpha"
    #                                                     "nodes":{"$gte":55, "$lte":100},
    #                                                         "fibre number":{"$ne":16}
    #                                                     },
    #                                                     parralel=False, max_count=args.mc)

    # parralel_ILP_connections(graph_list[:], collection="ta", max_time= 3*3600, workers=len(graph_list), threads=1, )
    # parralel_ILP_connections(graph_list, collection="ta", max_time=int(3*3600), workers=len(graph_list), threads=4, port=6380, hostname="128.40.43.93")
    # parralel_ILP_throughput(graph_list, collection="Review-Work", max_time=12*3600, workers=len(graph_list), threads=40, e=20, k=20)
    # parralel_heuristic_throughput(graph_list, collection="MPNN-uniform-55-100-test", workers=len(graph_list), route_function="FF-kSP",
    #                               e=20,k=20, m_step=200, channel_bandwidth=32e9, max_count=10, m_start=2000, port=6502,
    #                               hostname="128.40.43.93", fibre_num=16)
    # print("sleeping for: {}h".format(args.sleep))
    # time.sleep(args.sleep*3600)
    print("starting {} simulation".format(desc))
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", collection, "T_c",
                                                        find_dic=query,
                                                        parralel=False, max_count=max_count,
                                                        skip=skip)
    if len(graph_list) == 0:
        return 0
    if heuristic:

        parralel_heuristic_throughput(graph_list, collection=collection, workers=len(graph_list),
                                      route_function=route_function,
                                      e=20, k=20, m_step=200, channel_bandwidth=32e9, max_count=10,
                                      m_start=2000,
                                      port=port,
                                      hostname=hostname, fibre_num=fibre_num)
    elif ILP:
        parralel_ILP_connections(graph_list, collection=collection, max_time=max_time, workers=len(graph_list),
                                 threads=ILP_threads, port=port, hostname=hostname, fibre_num=fibre_num)

    # except Exception as err:
    #     if err is KeyboardInterrupt:
    #         exit()
    #     else:
    #         print(err)
    #         main()

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
