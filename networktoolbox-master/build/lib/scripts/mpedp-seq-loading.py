
import NetworkToolkit as nt
import numpy as np
import ray
from tqdm import tqdm
import time
import mp_module as mp


@ray.remote(num_cpus=1, memory=6000 * 1024 * 1024)
def MPEDP_routing(graph_list=None, collection=None, db="Topology_Data", e=10, k=10, route_function="FF-kSP",
                         m_step=100, max_count=10, channel_bandwidth=16e9, m_start=0):
    for graph, _id, T_c in graph_list:
        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=channel_bandwidth, routing_func=route_function)
        rwa_assignment = False

        M = m_start
        print("starting routing {}".format(route_function))
        demand_matrix_old = np.zeros((len(graph), len(graph)))
        rwa_active = None
        success = False
        connection_list = []
        edge_list = [[s, d] for s,d in graph.edges()]
        for i in range(max_count):
            while rwa_assignment != True:
                M += m_step

                # demand_matrix_old = np.ceil(np.array(T_c) * M)
                demand_matrix_new = np.ceil(np.array(T_c) * M)
                if (demand_matrix_new - demand_matrix_old).sum() > 0:
                    for i in range(len(graph)):
                        for j in range(i + 1, len(graph)):
                            for k in range(int(demand_matrix_new[i][j])):
                                connection_list.append([i+1, j + 1])

                    #     print(connection_list)
                    npair = len(connection_list)
                    #     print(npair)
                    ntrial = 1
                    niter = 10000
                    # print(edge_list)
                    # print(connection_list)
                    print("M: {}".format(M))
                    i = 1
                    while True:

                        time_start = time.perf_counter()
                        print("routing {} demands".format(len(connection_list[:i])))
                        Q = int(len(connection_list)/2+20)
                        Q = Q if Q < network.channels else network.channels
                        Q = 100
                        print("Q: {}".format(Q))
                        # weights = np.ones(graph.number_of_edges())
                        weights = [graph[s][d]["weight"] for s,d in graph.edges()]
                        rwa = mp.mp(len(graph), edge_list, weights, connection_list[:i],
                                    Q,
                                    ntrial, niter)
                        time_taken = time.perf_counter() - time_start
                        print("time taken: {}".format(time_taken))
                        print(rwa)
                        i*=2
                    demand_matrix_old = demand_matrix_new


def parralel_MPEDP_routing(graph_list, collection=None, db="Topology_Data", workers=50,
                                  route_function="FF-kSP", e=10, k=10, m_step=100,
                                  channel_bandwidth=16e9, max_count=10, m_start=0, port=6379,
                                  hostname="128.40.41.48"):
    ray.shutdown()
    ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    results = ray.get([MPEDP_routing.remote(
        db=db,
        collection=collection,
        graph_list=graph_list[indeces[ind]:indeces[ind + 1]], route_function=route_function, e=e, k=k,
        m_step=m_step, channel_bandwidth=channel_bandwidth,
        max_count=max_count, m_start=m_start) for ind in tqdm(range(workers))])
    # ray.shutdown()


if __name__ == "__main__":
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data",
                                                        "MPNN-uniform-test",
                                                        "T_c",
                                                        find_dic={"nodes":10, "edges":12},
                                                        max_count=1)

    print("processing {} graphs".format(len(graph_list)))
    parralel_MPEDP_routing(graph_list, collection="MPNN-uniform-test", workers=len(graph_list), hostname="128.40.43.93", port=6380, m_step=500)
    # Read in graphs ("MPNN-uniform-test", "nodes":10)
