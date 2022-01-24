import argparse
import NetworkToolkit as nt
import ray
from NetworkToolkit import Data
from tqdm import tqdm

@ray.remote(num_cpus=1, memory=500*1024*1024)
def assign_topology_parameters(graph, collection):
    funcs = [nt.Data.GraphProperties.update_m,
             nt.Data.GraphProperties.update_spanning_tree,
             nt.Data.GraphProperties.update_algebraic_connectivity,
             nt.Data.GraphProperties.update_node_variance,
             nt.Data.GraphProperties.update_mean_internodal_distance,
             nt.Data.GraphProperties.update_communicability_index,
             nt.Data.GraphProperties.update_comm_traff_ind,
             nt.Data.GraphProperties.update_graph_spectrum,
             nt.Data.GraphProperties.update_shortest_path_cost]
    for func in funcs:
        func(graph, collection)

if __name__ == "__main__":


    graph_list = [1]
    while len(graph_list) > 0:
        NT = 10000

        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform", max_count=NT,
                                                            find_dic={
    'degree variance': {
        '$exists': False
    }
},
                                                            parralel=True)
        ray.shutdown()
        ray.init(address='128.40.41.48:6379', _redis_password='5241590000000000', dashboard_port=8265)
        # GP = ray.remote(nt.Data.GraphProperties).remote()


        # funcs = [ray.remote(nt.Data.GraphProperties.update_m), ray.remote(nt.Data.GraphProperties.update_spanning_tree),
        #          ray.remote(nt.Data.GraphProperties.update_algebraic_connectivity), ray.remote(nt.Data.GraphProperties.update_node_variance),
        #         ray.remote(nt.Data.GraphProperties.update_mean_internodal_distance), ray.remote(nt.Data.GraphProperties.update_communicability_index),
        #         ray.remote(nt.Data.GraphProperties.update_comm_traff_ind), ray.remote(nt.Data.GraphProperties.update_graph_spectrum),
        #          ray.remote(nt.Data.GraphProperties.update_shortest_path_cost)]

        # tasks=[func.remote(graph, "MPNN-uniform") for func in funcs for graph in tqdm(graph_list)]
        ray.get([assign_topology_parameters.remote(graph, "MPNN-uniform") for graph in tqdm(graph_list)])

        #     ray.init()
        #     for graph in graph_list:
        #         for func in funcs:
        #     #         print(func)
        #             tempfunc = func
        #             tasks.append(tempfunc.remote(graph,"MPNN-uniform"))
        #     ray.get([GP.update_M.remote(graph, "MPNN-uniform") for graph in tqdm(graph_list)])
        # for func in tqdm(funcs):
        #     ray.get([func.remote(graph, "MPNN-uniform") for graph in graph_list])
        # ray.get([func.remote(graph, "MPNN-uniform") for graph in graph_list for func in tqdm(funcs)])
        #         tasks.append(GP.update_k_shortest_path_cost.remote(graph,"MPNN-uniform",4))
        #         tasks.append(GP.update_weighted_spectrum_distribution.remote(graph, "MPNN-uniform",4,10))
        #         tasks.append(GP.update_Dmax_value.remote(graph, "MPNN-uniform",156))
        #     tasks.append(GP.update_Dmin_value.remote(graph, "MPNN-uniform",156))
        #     ray.get(tasks)
        # print(tasks)
        ray.shutdown()