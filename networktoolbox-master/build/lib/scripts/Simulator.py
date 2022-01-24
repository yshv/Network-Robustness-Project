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
                    threads=10, throughput=False, pb_actor=None):
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

            nt.Database.update_data_with_id(db, collection, _id,
                                            newvals={"$set": {"ILP-connections RWA": rwa_assignment,
                                                              "ILP-connections": data["objective"],
                                                              "ILP-connections gap": data["gap"],
                                                              "ILP-connections data written": 1,
                                                              "data written ga": 1,
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
                print(datetime.datetime.utcnow())
                nt.Database.update_data_with_id(db, collection, _id,
                                                newvals={
                                                    "$set": {"ILP-connections timestamp": datetime.datetime.utcnow()}})

        except Exception as error:
            print(error)
            nt.Database.update_data_with_id(db, collection, _id,
                                            newvals={"$set": {"error occurred": 1,
                                                              "error": str(error),
                                                              "error host": str(socket.gethostname())}})
            continue
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
                             workers=50, threads=10, port=6379, hostname="128.40.41.48"):
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
        max_time=max_time, threads=threads, throughput=True, pb_actor=actor) for ind in range(workers)]
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
                                                                                   "ILP-throughput gap": data["gap"]}})
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
        # print("hello")
        # nt.Database.update_data_with_id(db, collection, _id,
        #                                 newvals={"$set": {"processing": 1}})
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
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

                demand_matrix_new = np.ceil(np.array(T_c) * M)
                if route_function == "FF-kSP" or "kSP-FF":
                    # print("M: {}".format(M))
                    if not success:
                        rwa_assignment = network.route(demand_matrix_new - demand_matrix_old, e=e, k=k)
                    elif (demand_matrix_new - demand_matrix_old).sum() > 0:
                        rwa_assignment = network.route(demand_matrix_new - demand_matrix_old, e=e, k=k,
                                                       rwa_assignment_previous=rwa_assignment)
                    # print(demand_matrix_new-demand_matrix_old
                    # print(rwa_assignment)
                    if rwa_assignment != True:
                        rwa_active = rwa_assignment
                        demand_matrix_old = demand_matrix_new
                        success = True

                else:
                    rwa_assignment = network.route(demand_matrix_new, e=e, k=k)

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

        assert nx.is_connected(graph) is True
        assert rwa_assignment != True
        assert rwa_assignment is not None

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
                                                          "{} time".format(route_function): time_taken,
                                                          "{} Capacity".format(route_function): throughput,
                                                          "fibre number": fibre_num,
                                                          "channel number": network.channels,
                                                          "channel bandwidth": network.channel_bandwidth,
                                                          "e": e,
                                                          "k": k,
                                                          "m step": m_step,
                                                          "max repeat": max_count,
                                                          "m start": m_start,
                                                          "processing": 0}})
        pb_actor.update.remote(1)


def parralel_heuristic_throughput(graph_list, collection=None, db="Topology_Data", workers=50, route_function="FF-kSP",
                                  e=10, k=10, m_step=100,
                                  channel_bandwidth=16e9, max_count=10, m_start=0, port=6379, hostname="128.40.41.48",
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
    for graph, _id, T_c in graph_list:
        nt.Database.update_data_with_id(db, collection, _id,
                                        newvals={"$set": {"processing": 1}})
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
def dwc_select_graph_generation(real_graph, E, T_C, alpha, _alpha, graph_function=None, graph_function_args=None, actor=None):
    #real_graph, E, T_C, alpha, _alpha, graph_function=None, graph_function_args=None,
                                # ind=None
    # # print("starting graph generation")
    # #
    # #
    #
    topology_generator = nt.Topology.Topology()
    if graph_function is None:
        new_graph = topology_generator.create_real_based_grid_graph(real_graph,
                                                                    E,
                                                                    database_name="Topology_Data",
                                                                    # collection_name="zib54-SBAG",
                                                                    #                                                     centre_start_node=True,
                                                                    #                                                     sequential_adding= True,
                                                                    #                                                     random_adding=True,
                                                                    numeric_adding=True,
                                                                    random_start_node=True,
                                                                    #                                                     first_start_node=True,
                                                                    #                                                     undershoot=True,
                                                                    overshoot=True,
                                                                    remove_C1_C2_edges=True,
                                                                    SBAG=True,
                                                                    #                                                     waxman_graph=True,
                                                                    alpha=_alpha,
                                                                    #                                                     beta=0.15,
                                                                    max_degree=100,
                                                                    plot_sequential_graphs=False,
                                                                    print_probs=False,
                                                                    ignore_constraints=True)
    else:
        # print("creating graphs with {}".format(str(graph_function)))
        new_graph = graph_function(**graph_function_args)

    inverse_DWC = 1e13
    if nx.is_connected(new_graph) and len(list(new_graph.edges)) == E and topology_generator.check_bridges(
            new_graph) and topology_generator.check_min_degree(
        new_graph):
        # inverse_DWC = nt.Tools.get_demand_weighted_cost([[new_graph, 1]], [T_C], alpha)[0]
        inverse_DWC = 1 / nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
    # print("finished with {}".format(str(graph_function)))
    new_graph = None
    inverse_DWC = None

    actor.update.remote(1)

    return new_graph, inverse_DWC


def parralel_graph_generation_DWC(N, E, grid_graph, T_c, workers=1, hostname="128.40.41.48", port=6379, alpha=[1],
                                  _alpha=5,
                                  graph_num=200, graph_function=None, graph_function_args=None):
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
    pb = nt.Tools.ProgressBar(N)
    actor = pb.actor
    tasks = [dwc_select_graph_generation.remote(grid_graph, E, T_c, alpha, _alpha, graph_function=graph_function,
                                                       graph_function_args=graph_function_args, actor=actor) for i in
        tqdm(range(N))]
    pb.print_until_done()
    # print(len(tasks))
    #grid_graph, E, T_c, alpha, _alpha, graph_function=graph_function,
                                                       # graph_function_args=graph_function_args, ind=i

    # graph, inverese_DWC = min(results, key=lambda item: item[1])
    results = ray.get(tasks)
    print(len(results))
    print(results[0])
    # exit()
    for graph, inverese_DWC in results:
        if len(graphs) < graph_num:
            graphs.append((graph, inverese_DWC))
        else:
            graph_worst, inverse_DWC_worst = min(graphs, key=lambda item: item[1])
            # index = graphs
            index = graphs.index((graph_worst, inverse_DWC_worst))
            if inverese_DWC > inverse_DWC_worst:
                graphs[index] = (graph, inverese_DWC)
        # if inverese_DWC < DWC_best:
        #     DWC_best = inverese_DWC
        #     graph_best = graph
    return graphs


@ray.remote
def random_select_graph_generation(grid_graph, E, T_C, alpha=[1], _alpha=5, graph_function=None,
                                   graph_function_args=None, gamma=None):
    topology_generator = nt.Topology.Topology()
    if graph_function is None:
        new_graph = topology_generator.create_real_based_grid_graph(grid_graph,
                                                                    E,
                                                                    database_name="Topology_Data",
                                                                    # collection_name="zib54-SBAG",
                                                                    #                                                     centre_start_node=True,
                                                                    #                                                     sequential_adding= True,
                                                                    #                                                     random_adding=True,
                                                                    numeric_adding=True,
                                                                    random_start_node=True,
                                                                    #                                                     first_start_node=True,
                                                                    #                                                     undershoot=True,
                                                                    overshoot=True,
                                                                    remove_C1_C2_edges=True,
                                                                    SBAG=True,
                                                                    #                                                     waxman_graph=True,
                                                                    alpha=_alpha,
                                                                    #                                                     beta=0.15,
                                                                    max_degree=100,
                                                                    plot_sequential_graphs=False,
                                                                    print_probs=False,
                                                                    ignore_constraints=True)
        DWC = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
    else:
        new_graph = graph_function(**graph_function_args)
        DWC = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
    while nx.is_connected(new_graph) and len(list(new_graph.edges)) == E and topology_generator.check_bridges(
            new_graph) and topology_generator.check_min_degree(
        new_graph):
        if graph_function is None:
            new_graph = topology_generator.create_real_based_grid_graph(grid_graph,
                                                                        E,
                                                                        database_name="Topology_Data",
                                                                        # collection_name="zib54-SBAG",
                                                                        #                                                     centre_start_node=True,
                                                                        #                                                     sequential_adding= True,
                                                                        #                                                     random_adding=True,
                                                                        numeric_adding=True,
                                                                        random_start_node=True,
                                                                        #                                                     first_start_node=True,
                                                                        #                                                     undershoot=True,
                                                                        overshoot=True,
                                                                        remove_C1_C2_edges=True,
                                                                        SBAG=True,
                                                                        #                                                     waxman_graph=True,
                                                                        alpha=_alpha,
                                                                        #                                                     beta=0.15,
                                                                        max_degree=100,
                                                                        plot_sequential_graphs=False,
                                                                        print_probs=False,
                                                                        ignore_constraints=True)
            DWC = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]
        else:
            new_graph = graph_function(**graph_function_args)
            DWC = nt.Tools.get_demand_weighted_cost_distance([[new_graph, 1]], [T_C], alpha)[0]

    return new_graph, DWC, T_C, gamma


def parralel_random_select_graph_generation(graph_list, E=140, alpha=[1], collection=None, db="Topology_Data",
                                            write=False,
                                            type=None, purpose=None, notes=None, graph_num=200, gamma_t=[0.0],
                                            hostname="",
                                            port=0, graph_function=None, graph_function_args=None, _alpha=5, **kwargs):
    grid_graph = graph_list[0][0]
    T_C_list = []
    for gamma in gamma_t:
        network = nt.Network.OpticalNetwork(grid_graph, channel_bandwidth=16e9)
        T_C = network.demand.create_skewed_demand(network.graph, gamma)
        T_C_list.append((T_C, gamma))

    # E = 120
    # alpha = [1]
    topology_num = 1
    tasks = []
    for T_C, gamma in T_C_list:
        tasks.append([random_select_graph_generation.remote(grid_graph, E, T_C, gamma=gamma, alpha=alpha, _alpha=_alpha,
                                                            graph_function=graph_function,
                                                            graph_function_args=graph_function_args) for i in
                      range(graph_num)])
    results = ray.get(tasks)
    if write:
        for graph, DWC, T_C, gamma in results:
            nt.Database.insert_graph(graph, db, collection, node_data=dict(graph.nodes.data()), use_pickle=True,
                                     type=type, purpose=purpose, T_c=T_C.tolist(), DWC=DWC, notes=notes,
                                     timestamp=datetime.datetime.utcnow(), gamma=gamma, graph_num=graph_num,
                                     graph_function=str(graph_function),
                                     graph_function_args=str(graph_function_args), **kwargs)


def DWC_select_graph_generation(graph_list, E=140, alpha=[1], collection=None, db="Topology_Data", write=False,
                                type=None, purpose=None, notes=None, graph_num=500000, gamma_t=[0.0], hostname="",
                                port=0, graph_function=None, graph_function_args=None, _alpha=5, **kwargs):
    grid_graph = graph_list[0][0]
    T_C_list = []
    for gamma in gamma_t:
        network = nt.Network.OpticalNetwork(grid_graph, channel_bandwidth=16e9)
        T_C = network.demand.create_skewed_demand(network.graph, gamma)
        T_C_list.append((T_C, gamma))

    # E = 120
    # alpha = [1]
    topology_num = 1
    for T_C, gamma in T_C_list:
        dwc_select_graphs = parralel_graph_generation_DWC(graph_num, E, grid_graph, T_C_list[0][0], workers=1000,
                                                          hostname=hostname,
                                                          port=port,
                                                          alpha=alpha, _alpha=_alpha, graph_num=200,
                                                          graph_function=graph_function,
                                                          graph_function_args=graph_function_args)

        if write:
            for graph, inverse_DWC in dwc_select_graphs:
                nt.Database.insert_graph(graph, db, collection, node_data=dict(graph.nodes.data()), use_pickle=True,
                                         type=type, purpose=purpose, T_c=T_C.tolist(), DWC=1 / inverse_DWC, notes=notes,
                                         timestamp=datetime.datetime.utcnow(), gamma=gamma, graph_num=graph_num,
                                         graph_function=str(graph_function),
                                         graph_function_args=str(graph_function_args), **kwargs)


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
         heuristic=False, ILP=False, max_time=3 * 3600):
    # try:
    graph_list = [1]
    ind = 0
    try:

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
        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", collection, "T_c",
                                                            find_dic=query,
                                                            parralel=False, max_count=args.mc)
        if heuristic:

            parralel_heuristic_throughput(graph_list, collection=collection, workers=len(graph_list),
                                          route_function=route_function,
                                          e=20, k=20, m_step=200, channel_bandwidth=32e9, max_count=10,
                                          m_start=2000,
                                          port=port,
                                          hostname=hostname, fibre_num=4)
        elif ILP:
            parralel_ILP_connections(graph_list, collection=collection, max_time=max_time, workers=len(graph_list),
                                     threads=1)



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
    #     real_graph = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "NSFNET"},
    #                                                         node_data=True)[0][0]
    #     for gamma in np.around(np.arange(0, 1.1, 0.2), decimals=1):
    #         dataset_er = nt.Database.read_data_into_pandas("Topology_Data", "ta", find_dic={"type":"ER","purpose":"structural analysis","gamma":gamma})
    #         print(dataset_er[:1])
    #         T_c = dataset_er["traffic_matrix"][0]
    #         alpha = [0.66576701, 0.15450554, 0.05896479, 0.03678006, 0.02237619, 0.01477571,
    #                  0.01055498, 0.0077231,  0.00602809, 0.00448774, 0.00366258, 0.00290902,
    #                  0.00248457, 0.00198062, 0.00163766, 0.00139676, 0.0012504,  0.00101622,
    #                  0.00085918, 0.00083979]
    #         graphs = parralel_graph_generation_DWC(500000, 21, real_graph, workers=1000, T_c=T_c, port=6380, hostname="128.40.43.93", _alpha=0, alpha=alpha)
    #         print(len(graphs))
    #         # print(graph.number_of_edges())
    #         # print(DWC)
    #         for graph, DWC in graphs:
    #             nt.Database.insert_graph(graph, "Topology_Data", "ta", node_data=dict(graph.nodes.data()), use_pickle=True, type="BA", purpose="ga-analysis", gamma=gamma, T_c=T_c, DWC=DWC, data="new alpha")
    #     MPNN.create_graphs_MPNN(nodes=list(range(25, 50, 5)), graph_num=100, collection="MPNN-uniform-25-45-test",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname="128.40.43.93", amount=10, local=False)
    #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=100, collection="MPNN-uniform-55-100-test",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname="128.40.43.93", amount=10, local=False)
    #     ray.shutdown()
    #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=300, collection="MPNN-uniform-55-100",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname="128.40.43.93", amount=10, local=False)
    #     ray.shutdown()
    print("hello")
    # main(route_function="FF-kSP", collection="prufer-select-ga",
    #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48", port=7111,
    #      heuristic=True)
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
                                                        find_dic={"FF-kSP Capacity": {'$exists': True},
                                                                  'nodes': 100},
                                                        max_count=1, use_pickle=True)

    #     dataset = nt.Database.read_data_into_pandas("Topology_Data", "MPNN-uniform-55-100",
    #                                                 find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
    #                                                 max_count=1)
    ## Large scale dwc-select prufer sequence, ER, and BA
    DWC_select_graph_generation(graph_list, E=140, write=False, collection="dwc-select", type="prufer-sequence",
                                notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
                                graph_num=10000,
                                port=7111, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                graph_function=nt.Tools.prufer_sequence_ptd,
                                graph_function_args={"N": len(graph_list[0][0]), "E": 140,
                                                     "grid_graph": graph_list[0][0]}, selection_method="dwc-select")
    # DWC_select_graph_generation(graph_list, E=140, write=False, collection="dwc-select", type="ER",
    #                             notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                             graph_num=10000,
    #                             port=7111, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                             graph_function=nt.Tools.create_spatial_ER_graph,
    #                             graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                             selection_method="dwc-select")
    # DWC_select_graph_generation(graph_list, E=140, write=False, collection="dwc-select", type="BA",
    #                             notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                             graph_num=10000,
    #                             port=7111, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                             graph_function=None,
    #                             _alpha=0,
    #                             graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                             selection_method="dwc-select")

    ## Large scale random graphs BA, ER, SNR-BA, prufer-sequence - "ptd-random"
    parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="BA",
                                            notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
                                            graph_num=10000,
                                            port=7111, hostname="128.40.41.48",
                                            gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                            graph_function=None,
                                            _alpha=0,
                                            graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
                                            selection_method="dwc-select")
    parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="ER",
                                            notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
                                            graph_num=10000,
                                            port=7111, hostname="128.40.41.48",
                                            gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                            graph_function=nt.Tools.create_spatial_ER_graph,
                                            _alpha=0,
                                            graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
                                            selection_method="dwc-select")
    parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="SNR-BA",
                                            notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
                                            graph_num=10000,
                                            port=7111, hostname="128.40.41.48",
                                            gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                            graph_function=None,
                                            _alpha=5,
                                            graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
                                            selection_method="dwc-select")
    parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random",
                                            type="prufer-sequence",
                                            notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
                                            graph_num=10000,
                                            port=7111, hostname="128.40.41.48",
                                            gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                            graph_function=nt.Tools.prufer_sequence_ptd,
                                            _alpha=5,
                                            graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
                                            selection_method="dwc-select")

    ## Small scale random and dwc-select graphs for prufer-sequence
    small_scale_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
                                                                    find_dic={"name": "NSFNET"},
                                                                    node_data=True)
    parralel_random_select_graph_generation(small_scale_graph_list,
                                            E=small_scale_graph_list[0][0].number_of_edges(), write=False,
                                            collection="ptd-random",
                                            type="prufer-sequence",
                                            notes="Small scale random graphs for different traffic matrices, "
                                                  "although this does not make a difference here. Generated via "
                                                  "prufer-sequence method.",
                                            graph_num=10000,
                                            port=7111, hostname="128.40.41.48",
                                            gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                            graph_function=nt.Tools.prufer_sequence_ptd,
                                            _alpha=5,
                                            graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
                                            selection_method="dwc-select")
    DWC_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(),
                                write=False, collection="dwc-select", type="prufer-sequence",
                                notes="Small scale dwc-select graphs for the prufer-sequence method, created from "
                                      "varying traffic matrices (locally skewed).",
                                graph_num=10000,
                                port=7111, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                graph_function=nt.Tools.prufer_sequence_ptd,
                                graph_function_args={"N": len(graph_list[0][0]), "E": 140,
                                                     "grid_graph": graph_list[0][0]}, selection_method="dwc-select")

    ## throughput calculations for large scale: ER, BA, SNR-BA, prufer-sequence (random), ER, BA (dwc-select)
    main(route_function="FF-kSP", collection="ptd-random",
         query={"FF-kSP Capacity": {"$exists": False}}, hostname="128.40.41.48", port=7111,
         heuristic=True)
    main(route_function="FF-kSP", collection="dwc-select",
         query={"FF-kSP Capacity": {"$exists": False}}, hostname="128.40.41.48", port=7111, heuristic=True)

    ## throughput calculations for small-scale: SNR-BA, BA, ER, ga
    main(route_function="FF-kSP", collection="ta",
         query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': False},
                'data': {'$ne': 'new alpha'}, 'type': 'SNR-BA'}, hostname="128.40.41.48",
         port=7111, ILP=True)
    main(route_function="FF-kSP", collection="ta",
         query={'purpose': 'structural analysis', 'type': 'SBAG',
                'ILP-connections': {'$exists': False},
                'Demand Weighted Cost': {'$exists': True},
                "node order": "numeric", "alpha": 5}, hostname="128.40.41.48", port=7111,
         ILP=True)
    main(route_function="FF-kSP", collection="ta",
         query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},
                'type': 'ER'}, hostname="128.40.41.48", port=7111,
         ILP=True)
    main(route_function="FF-kSP", collection="ta",
         query={'purpose': 'structural analysis', 'type': 'ER',
                'ILP-connections': {'$exists': True},
                'Demand Weighted Cost': {'$exists': True}}, hostname="128.40.41.48", port=7111,
         ILP=True)
    main(route_function="FF-kSP", collection="ta",
         query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},
                'data': {'$ne': 'new alpha'}, 'type': 'BA'}, hostname="128.40.41.48",
         port=7111,
         ILP=True)
    main(route_function="FF-kSP", collection="ta",
         query={'purpose': 'structural analysis', 'type': 'BA',
                'ILP-connections': {'$exists': True},
                'Demand Weighted Cost': {'$exists': True},
                }, hostname="128.40.41.48",
         port=7111,
         ILP=True)

    ## throughput calculations for ga-ps
    main(route_function="FF-kSP", collection="prufer-select-ga",
         query={"FF-kSP Capacity": {"$exists": False}}, hostname="128.40.41.48", port=7111, heuristic=True)

    # main(route_function="FF-kSP", collection="prufer-select-ga", query={"type": "prufer-select-ga"}, hostname="128.40.41.48",
    #      port=7111)

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
