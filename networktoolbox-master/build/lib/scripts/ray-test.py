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
    @ray.remote
    def slow_function():
        print('starting')
        for i in range(3600):
            time.sleep(1)
            _time = "a"+"b"
            hello = "hello"
        return 1

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
    exit()
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
    # for T_C, gamma in T_C_list:
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

if __name__ == "__main__":
    small_scale_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
                                                        find_dic={"FF-kSP Capacity": {'$exists': True},
                                                                  'nodes': 100},
                                                        max_count=1, use_pickle=True)
    DWC_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(), write=False, collection="dwc-select", type="prufer-sequence",
                                notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
                                graph_num=10000,
                                port=7111, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                graph_function=nt.Tools.prufer_sequence_ptd,
                                graph_function_args={"N": len(small_scale_graph_list[0][0]), "E": small_scale_graph_list[0][0].number_of_edges(),
                                                     "grid_graph": small_scale_graph_list[0][0]}, selection_method="dwc-select")