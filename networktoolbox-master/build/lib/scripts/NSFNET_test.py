
import NetworkToolkit as nt
import numpy as np
import ray
import time

# rwa = network.rwa.message_passing_routing(demand_matrix, Q_layer=Qlayer)
# print(rwa)
@ray.remote
def mpedp(graph, demand_matrix, Q):
    network = nt.Network.OpticalNetwork(graph)
    time_start = time.perf_counter()
    rwa = network.rwa.message_passing_routing(demand_matrix, Q_layer=Q)
    time_stop = time.perf_counter()
    time_diff=time_stop-time_start
    print("Q: {} time taken: {}".format(Q, time_diff))
    return rwa
def parrelelise_mpedp(w_range, demand_matrix, graph):
    print("parralel")
    ray.init()
    # simulators = [parrelel_network.remote(graph) for i in range(len(w_range))]
    results = ray.get([mpedp.remote(graph, demand_matrix, w) for ind, w in enumerate(w_range)])
    ray.shutdown()
    return results

if __name__ =="__main__":
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "NSFNET"})
    graph = graph_list[0][0]
    Qlayer = 14


    demand_matrix = np.ones((len(graph), len(graph)))
    np.fill_diagonal(demand_matrix, 0)
    w_range = list(range(1,25))
    print(parrelelise_mpedp(w_range, demand_matrix, graph))

