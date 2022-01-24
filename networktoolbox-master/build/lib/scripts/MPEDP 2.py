import ray
import NetworkToolkit as nt
import networkx as nx
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import time
import mp_module as mp
import socket

def distribute_func(func, graph_list, workers=1):
    # get indeces [0,1,2] for example for data 0-2
    indeces = nt.Tools.create_start_stop_list(len(graph_list), workers)
    print(indeces)
    # Run all the ray instances
    results = ray.get([func.remote(graph_list[indeces[i]:indeces[i+1]]) for i in range(workers)])

@ray.remote(memory=13000*1024*1024)
def MPEDP_run(graph_list):
    MP_time = []
    MP_length = []
    RWA = []
    
    for graph, index in graph_list:
        print("starting MPEDP on {}".format(socket.gethostname()))

        edge_list = [np.sum([list(x),[1,1]],axis=0) for x in graph.edges()]

    #     print(edge_list)
        N = len(graph.nodes())
        Nedge = len(edge_list)
        Qlayer = 156
        network = nt.Network.OpticalNetwork(graph)
        connection_list = []
        demand_matrix = np.ones((N,N))
        np.fill_diagonal(demand_matrix,0)
        #     print(demand_matrix)

        for i in range(N):
            for j in range(i+1,N):
                for k in range(int(demand_matrix[i][j])):
                    connection_list.append([i+1,j+1])

    #     print(connection_list)
        npair = len(connection_list)
        #     print(npair)
        ntrial = 1
        niter = 10000

        try:
            time_start = time.perf_counter()
            rwa = mp.mp(N, edge_list, connection_list, Qlayer, ntrial, niter)

            time_taken = time.perf_counter() - time_start

            print(time_taken)
            #     print(rwa)

            #     print(len(rwa[153][0]))
            total_length = 0
            for i in range(Qlayer):
                if len(rwa[i]) > 0:
                    for j in range(len(rwa[i])):
                        total_length += len(rwa[i][j])-1

            print(total_length)
            MP_time.append(time_taken)
            MP_length.append(total_length)
            RWA.append(rwa)

            nt.Database.update_data_with_id("Topology_Data", "MPEDP", index,{"$set": {"MP computation time": time_taken}})
            nt.Database.update_data_with_id("Topology_Data", "MPEDP", index,{"$set": {"MP total length": total_length}})
            nt.Database.update_data_with_id("Topology_Data", "MPEDP", index,{"$set": {"MP RWA": rwa}})

        except Exception as error:
            print(error)
            nt.Database.update_data_with_id("Topology_Data", "MPEDP", index,
                                            newvals={"$set": {"error occurred": 1,"error": str(error),
                                                              "error host": str(socket.gethostname())}})
    
    return MP_time, MP_length, RWA


if __name__ == "__main__":
    ray.init()
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPEDP", find_dic={"computational scaling data":1, "error occurred":1, "MP RWA":{"$exists":False}},use_pickle=True)
    distribute_func(MPEDP_run, graph_list, workers=int(len(graph_list)/2))
    ray.shutdown()