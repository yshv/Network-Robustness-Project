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
from simulation_base import *

parser = argparse.ArgumentParser(description='Optical Network Simulator Script')
parser.add_argument('-m', action='store', type=int, default=3000, help="How much memory to allocate to processes")
parser.add_argument('-cpu', action='store', type=int, default=1, help="How many cpus to use")
parser.add_argument('-mc', action='store', type=int, default=1000, help="How many samples to read")
parser.add_argument('--sleep', action='store', type=float, default=1000, help="How long to wait")
args = parser.parse_args()


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
    #         graphs = parralel_graph_generation_DWC(500000, 21, real_graph, workers=1000, T_c=T_c, port=6380, hostname="128.40.41.48", _alpha=0, alpha=alpha)
    #         print(len(graphs))
    #         # print(graph.number_of_edges())
    #         # print(DWC)
    #         for graph, DWC in graphs:
    #             nt.Database.insert_graph(graph, "Topology_Data", "ta", node_data=dict(graph.nodes.data()), use_pickle=True, type="BA", purpose="ga-analysis", gamma=gamma, T_c=T_c, DWC=DWC, data="new alpha")
    #     MPNN.create_graphs_MPNN(nodes=list(range(25, 50, 5)), graph_num=100, collection="MPNN-uniform-25-45-test",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname="128.40.41.48", amount=10, local=False)
    #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=100, collection="MPNN-uniform-55-100-test",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname="128.40.41.48", amount=10, local=False)
    #     ray.shutdown()
    #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=300, collection="MPNN-uniform-55-100",
    #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #                             port=6380, hostname="128.40.41.48", amount=10, local=False)
    #     ray.shutdown()
    print("hello")
    # main(route_function="FF-kSP", collection="prufer-select-ga",
    #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48", port=7112,
    #      heuristic=True)
    # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
    #                                                     find_dic={"FF-kSP Capacity": {'$exists': True},
    #                                                               'nodes': 100},
    #                                                     max_count=1, use_pickle=True)

    #     dataset = nt.Database.read_data_into_pandas("Topology_Data", "MPNN-uniform-55-100",
    #                                                 find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
    #                                                 max_count=1)
    ## Large scale dwc-select prufer sequence, ER, and BA
    gamma_value = np.around(np.arange(0, 1.1, 0.2), decimals=1)
    graph_list = []
    for gamma in gamma_value:
        graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "prufer-select-ga", 'T_c', node_data=True,
                                                             find_dic={'gamma': gamma},
                                                             max_count=1, use_pickle=True)
    T_C_list = [(graph[2], gamma) for graph, gamma in zip(graph_list, gamma_value)]

    # DWC_select_graph_generation(graph_list, E=345, write=True, collection="snr-ba-distance", _type="SNR-BA",
    #                             notes="SNR-BA graphs with fibre limit of 280000km of fibre",
    #                             graph_num=600, final_graph_num=200, T_C_list=T_C_list,
    #                             port=7112, hostname="128.40.41.23",
    #                             graph_function=None, fibre_limit=280000, combined_DWC=True,
    #                             graph_function_args=None, selection_method="dwc-select")
    # parralel_random_select_graph_generation(graph_list, E=345, write=True, collection="snr-ba-distance", _type="SNR-BA",
    #                                         notes="random snr-ba graphs for ptd comparison.",
    #                                         graph_num=200,
    #                                         T_C_list=T_C_list,
    #                                         port=7112, hostname="128.40.41.23",
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                                         graph_function=None,
    #                                         _alpha=5,
    #                                         fibre_limit=280000,
    #                                         combined_DWC=True,
    #                                         graph_function_args=None,
    #                                         selection_method="random")
    # DWC_select_graph_generation(graph_list, E=140, write=False, collection="dwc-select", type="ER",
    #                             notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                             graph_num=10000,
    #                             port=7112, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                             graph_function=nt.Tools.create_spatial_ER_graph,
    #                             graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                             selection_method="dwc-select")
    # DWC_select_graph_generation(graph_list, E=140, write=False, collection="dwc-select", type="BA",
    #                             notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                             graph_num=10000,
    #                             port=7112, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                             graph_function=None,
    #                             _alpha=0,
    #                             graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                             selection_method="dwc-select")

    ## Large scale random graphs BA, ER, SNR-BA, prufer-sequence - "ptd-random"
    # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="BA",
    #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                                         graph_num=10000,
    #                                         port=7112, hostname="128.40.41.48",
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                                         graph_function=None,
    #                                         _alpha=0,
    #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                                         selection_method="dwc-select")
    # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="ER",
    #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                                         graph_num=10000,
    #                                         port=7112, hostname="128.40.41.48",
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                                         graph_function=nt.Tools.create_spatial_ER_graph,
    #                                         _alpha=0,
    #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                                         selection_method="dwc-select")
    # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="SNR-BA",
    #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                                         graph_num=10000,
    #                                         port=7112, hostname="128.40.41.48",
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                                         graph_function=None,
    #                                         _alpha=5,
    #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                                         selection_method="dwc-select")
    # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random",
    #                                         type="prufer-sequence",
    #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #                                         graph_num=10000,
    #                                         port=7112, hostname="128.40.41.48",
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                                         graph_function=nt.Tools.prufer_sequence_ptd,
    #                                         _alpha=5,
    #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #                                         selection_method="dwc-select")

    ## hostname="128.40.41.48"
    # small_scale_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
    #                                                                 find_dic={"name": "NSFNET"},
    #                                                                 node_data=True)
    # parralel_random_select_graph_generation(small_scale_graph_list,
    #                                         E=small_scale_graph_list[0][0].number_of_edges(), write=True,
    #                                         collection="ptd-random",
    #                                         type="prufer-sequence",
    #                                         notes="Small scale random graphs for different traffic matrices, "
    #                                               "although this does not make a difference here. Generated via "
    #                                               "prufer-sequence method.",
    #                                         graph_num=10000,
    #                                         port=7112, hostname="128.40.41.48",
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                                         graph_function=nt.Tools.prufer_sequence_ptd,
    #                                         _alpha=5,
    #                                         graph_function_args={"E": small_scale_graph_list[0][0].number_of_edges(), "grid_graph": small_scale_graph_list[0][0]},
    #                                         selection_method="random")
    # DWC_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(),
    #                             write=True, collection="prufer-select", type="prufer-sequence",
    #                             notes="Small scale dwc-select graphs for the prufer-sequence method, created from "
    #                                   "varying traffic matrices (locally skewed).",
    #                             graph_num=10000,
    #                             port=7112, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #                             graph_function=nt.Tools.prufer_sequence_ptd,
    #                             graph_function_args={"N": len(small_scale_graph_list[0][0]), "E": small_scale_graph_list[0][0].number_of_edges(),
    #                                                  "grid_graph": small_scale_graph_list[0][0]}, selection_method="dwc-select")
    # exit()

    ## throughput calculations for large scale: ER, BA, SNR-BA, prufer-sequence (random), ER, BA (dwc-select)
    # main(route_function="kSP-FF", collection="ptd-random",
    #      query={"kSP-FF Capacity": {"$exists": False}}, hostname="128.40.41.48", port=7112,
    #      heuristic=True, fibre_num=16,
    #      desc="ptd-random", max_count=1)
    # main(route_function="FF-kSP", collection="dwc-select",
    #      query={"FF-kSP Capacity": {"$exists": False}}, hostname="128.40.41.48", port=7112,
    #      heuristic=True, fibre_num=16,
    #      desc="dwc-select")

    ## throughput calculations for small-scale: SNR-BA, BA, ER, ga
    # main(route_function="FF-kSP", collection="ta",
    #      query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},
    #             'data': {'$ne': 'new alpha'}, 'type': 'SNR-BA'}, hostname="128.40.41.48",
    #      port=7112, ILP=True, desc="SNR-BA small")
    # main(route_function="FF-kSP", collection="ta",
    #      query={'purpose': 'structural analysis', 'type': 'SBAG',
    #             'ILP-connections': {'$exists': True},
    #             'ILP-connections Capacity': {"$exists":False},
    #             'Demand Weighted Cost': {'$exists': True},
    #             "node order": "numeric", "alpha": 5}, hostname="128.40.41.48", port=7112,
    #      ILP=True, , desc="SBAG small")
    # main(route_function="FF-kSP", collection="ta",
    #      query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},'ILP-connections Capacity': {"$exists":False},
    #             'type': 'ER'}, hostname="128.40.41.48", port=7112,
    #      ILP=True, desc="ER small")
    # main(route_function="FF-kSP", collection="ta",
    #      query={'purpose': 'structural analysis', 'type': 'ER',
    #             'ILP-connections': {'$exists': True},
    #             'Demand Weighted Cost': {'$exists': True}}, hostname="128.40.41.48", port=7112,
    #      ILP=True, desc="")
    # main(route_function="FF-kSP", collection="ta",
    #      query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},
    #             'ILP-connections Capacity': {"$exists":False},
    #             'data': {'$ne': 'new alpha'}, 'type': 'BA'}, hostname="128.40.41.48",
    #      port=7112,
    #      ILP=True, desc="BA small")
    # main(route_function="FF-kSP", collection="ta",
    #      query={'purpose': 'structural analysis', 'type': 'BA',
    #             'ILP-connections': {'$exists': True},
    #             'ILP-connections Capacity': {"$exists":False},
    #             'Demand Weighted Cost': {'$exists': True},
    #             }, hostname="128.40.41.48",
    #      port=7112,import NetworkToolkit as nt
    # from tqdm import tqdm
    # import numpy as np
    # import ray
    # import ast
    # import socket
    # import os
    # import networkx as nx
    # import time
    # import datetime
    # import argparse
    # import MPNN
    # from simulation_base import *
    #
    # parser = argparse.ArgumentParser(description='Optical Network Simulator Script')
    # parser.add_argument('-m', action='store', type=int, default=3000, help="How much memory to allocate to processes")
    # parser.add_argument('-cpu', action='store', type=int, default=1, help="How many cpus to use")
    # parser.add_argument('-mc', action='store', type=int, default=1000, help="How many samples to read")
    # parser.add_argument('--sleep', action='store', type=float, default=1000, help="How long to wait")
    # args = parser.parse_args()
    #
    #
    #
    # if __name__ == "__main__":
    #     # scale_throughput()
    #     # graph, _id = nt.Database.read_topology_dataset_list("Topology_Data","real", find_dic={"name":
    #     #                                                                                  "30-Node-ONDPBook-Topology_nodes"})[0]
    #     # print('read graph, calculating chromatic number')
    #     # rwa, chrom_num = ILP_chromatic_number(graph,max_time=48*3600)
    #     # nt.Database.update_data_with_id("Topology_Data", "real",_id, newvals={"$set":{"wavelength requirement":chrom_num}})
    #     #     graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"BA",
    #     #                                                                                                      "nodes":14,
    #     #                                                                                                      "ILP Capacity":{
    #     #                                                                                                          "$exists":True}},
    #     # node_data=True, max_count=200)
    #     #     graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"ER",
    #     #                                                                                                      "nodes":14,
    #     #                                                                                                      "ILP Capacity":{
    #     #                                                                                                          "$exists":True}},
    #     #                                                         node_data=True, max_count=200)
    #     #     graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"SBAG",
    #     #                                                                                                      "nodes":14,
    #     #                                                                                                      "ILP Capacity":{
    #     #                                                                                                          "$exists":True}},
    #     #                                                         node_data=True,max_count=200)
    #     #     real_graph = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name": "NSFNET"},
    #     #                                                         node_data=True)[0][0]
    #     #     for gamma in np.around(np.arange(0, 1.1, 0.2), decimals=1):
    #     #         dataset_er = nt.Database.read_data_into_pandas("Topology_Data", "ta", find_dic={"type":"ER","purpose":"structural analysis","gamma":gamma})
    #     #         print(dataset_er[:1])
    #     #         T_c = dataset_er["traffic_matrix"][0]
    #     #         alpha = [0.66576701, 0.15450554, 0.05896479, 0.03678006, 0.02237619, 0.01477571,
    #     #                  0.01055498, 0.0077231,  0.00602809, 0.00448774, 0.00366258, 0.00290902,
    #     #                  0.00248457, 0.00198062, 0.00163766, 0.00139676, 0.0012504,  0.00101622,
    #     #                  0.00085918, 0.00083979]
    #     #         graphs = parralel_graph_generation_DWC(500000, 21, real_graph, workers=1000, T_c=T_c, port=6380, hostname="128.40.41.48", _alpha=0, alpha=alpha)
    #     #         print(len(graphs))
    #     #         # print(graph.number_of_edges())
    #     #         # print(DWC)
    #     #         for graph, DWC in graphs:
    #     #             nt.Database.insert_graph(graph, "Topology_Data", "ta", node_data=dict(graph.nodes.data()), use_pickle=True, type="BA", purpose="ga-analysis", gamma=gamma, T_c=T_c, DWC=DWC, data="new alpha")
    #     #     MPNN.create_graphs_MPNN(nodes=list(range(25, 50, 5)), graph_num=100, collection="MPNN-uniform-25-45-test",
    #     #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #     #                             port=6380, hostname="128.40.41.48", amount=10, local=False)
    #     #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=100, collection="MPNN-uniform-55-100-test",
    #     #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #     #                             port=6380, hostname="128.40.41.48", amount=10, local=False)
    #     #     ray.shutdown()
    #     #     MPNN.create_graphs_MPNN(nodes=list(range(55, 105, 5)), graph_num=300, collection="MPNN-uniform-55-100",
    #     #                             alpha_t_range=[0], scale_range=[1], edge_factor=0.25,
    #     #                             port=6380, hostname="128.40.41.48", amount=10, local=False)
    #     #     ray.shutdown()
    #     print("hello")
    #     # main(route_function="FF-kSP", collection="prufer-select-ga",
    #     #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48", port=7112,
    #     #      heuristic=True)
    #     # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
    #     #                                                     find_dic={"FF-kSP Capacity": {'$exists': True},
    #     #                                                               'nodes': 100},
    #     #                                                     max_count=1, use_pickle=True)
    #
    #     #     dataset = nt.Database.read_data_into_pandas("Topology_Data", "MPNN-uniform-55-100",
    #     #                                                 find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
    #     #                                                 max_count=1)
    #     ## Large scale dwc-select prufer sequence, ER, and BA
    #     gamma_value = np.around(np.arange(0, 1.1, 0.2), decimals=1)
    hostname = "128.40.41.48"
    port = 7112
    graph_list = []
    for gamma in gamma_value:
        graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "prufer-select-ga", 'T_c', node_data=True,
                                                             find_dic={'gamma': gamma},
                                                             max_count=1, use_pickle=True)
    T_C_list = [(graph[2], gamma) for graph, gamma in zip(graph_list, gamma_value)]
    #


    # parralel_random_select_graph_generation(graph_list, E=345, write=True, collection="snr-ba-distance-sequence",
    #                                         _type="SNR-BA", notes="random snr-ba graphs for ptd comparison.",
    #                                         graph_num=200, T_C_list=T_C_list, port=7112, hostname="128.40.41.23",
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], graph_function=None, _alpha=5,
    #                                         fibre_limit=280000, combined_DWC=True, graph_function_args=None,
    #                                         selection_method="random", node_order_type="sequential", start_node="random")
    DWC_select_graph_generation(graph_list, E=365, write=True, collection="snr-ba-distance-sequential", _type="SNR-BA",
                                notes="SNR-BA graphs with fibre limit of 280000km of fibre, with sequential node order",
                                graph_num=1000, final_graph_num=200, T_C_list=T_C_list,
                                port=port, hostname=hostname,
                                graph_function=None, fibre_limit=280000, combined_DWC=True,
                                graph_function_args=None, selection_method="dwc-select",
                                node_order_type="sequential", start_node="random")
    #
    # print("DONE!!!")
    # exit()
    small_scale_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
                                                                    find_dic={"name": "NSFNET"},
                                                                    node_data=True)
    DWC_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(), write=True,
                                collection="snr-ba-distance-sequential", _type="SNR-BA",
                                notes="SNR-BA graphs with fibre limit of 42000km of fibre, with sequential node order. Based on NSFNET.",
                                graph_num=1000, final_graph_num=200, T_C_list=T_C_list,
                                port=port, hostname=hostname,
                                graph_function=None, fibre_limit=42000, combined_DWC=True,
                                graph_function_args=None, selection_method="dwc-select",
                                node_order_type="sequential", start_node="random")
    # parralel_random_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(),
    #                                         write=True, collection="snr-ba-distance-sequential",
    #                                         _type="SNR-BA", notes="random snr-ba graphs for ptd comparison. Based on NSFNET.",
    #                                         graph_num=200, T_C_list=T_C_list, port=port, hostname=hostname,
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], graph_function=None, _alpha=5,
    #                                         fibre_limit=42000, combined_DWC=True, graph_function_args=None,
    #                                         selection_method="random", node_order_type="sequential", start_node="random")
    DWC_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(), write=True,
                                collection="ba-distance-sequential", _type="BA",
                                notes="BA graphs with fibre limit of 42000km of fibre, with sequential node order. Based on NSFNET.",
                                graph_num=1000, final_graph_num=200, T_C_list=T_C_list,
                                port=port, hostname=hostname, _alpha=0,
                                graph_function=None, fibre_limit=42000, combined_DWC=True,
                                graph_function_args=None, selection_method="dwc-select",
                                node_order_type="sequential", start_node="random")
    # parralel_random_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(),
    #                                         write=True, collection="ba-distance-sequential",
    #                                         _type="BA",
    #                                         notes="random ba graphs for ptd comparison. Based on NSFNET.",
    #                                         graph_num=200, T_C_list=T_C_list, port=port, hostname=hostname,
    #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], graph_function=None, _alpha=0,
    #                                         fibre_limit=42000, combined_DWC=True, graph_function_args=None,
    #                                         selection_method="random", node_order_type="sequential",
    #                                         start_node="random")
    main(route_function="FF-kSP", collection="vector-ga-distance",
         query={"ILP-connections Capacity": {"$exists": False}}, hostname=hostname, port=port, ILP=True,
         desc="Ruijie PTD throughput calc", fibre_num=1, max_count=30000)
    main(route_function="FF-kSP", collection="snr-ba-distance-sequential",
         query={"FF-kSP Capacity": {"$exists": False}, "nodes":{"$gte":30}}, hostname=hostname, port=port, heuristic=True,
         desc="Ruijie PTD throughput calc", fibre_num=16, max_count=30000)
    main(collection="snr-ba-distance-sequential",
         query={"ILP-connections Capacity": {"$exists": False}, "nodes":{"$lte":20}}, hostname=hostname, port=port, ILP=True,
         desc="Ruijie PTD throughput calc", fibre_num=1, max_count=30000)
    main(collection="ba-distance-sequential",
         query={"ILP-connections Capacity": {"$exists": False}, "nodes": {"$lte": 20}}, hostname=hostname, port=port,
         ILP=True,
         desc="Ruijie PTD throughput calc", fibre_num=1, max_count=30000)


    #     # DWC_select_graph_generation(graph_list, E=140, write=False, collection="dwc-select", type="ER",
    #     #                             notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #     #                             graph_num=10000,
    #     #                             port=7112, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                             graph_function=nt.Tools.create_spatial_ER_graph,
    #     #                             graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #     #                             selection_method="dwc-select")
    #     # DWC_select_graph_generation(graph_list, E=140, write=False, collection="dwc-select", type="BA",
    #     #                             notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #     #                             graph_num=10000,
    #     #                             port=7112, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                             graph_function=None,
    #     #                             _alpha=0,
    #     #                             graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #     #                             selection_method="dwc-select")
    #
    #     ## Large scale random graphs BA, ER, SNR-BA, prufer-sequence - "ptd-random"
    #     # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="BA",
    #     #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #     #                                         graph_num=10000,
    #     #                                         port=7112, hostname="128.40.41.48",
    #     #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                                         graph_function=None,
    #     #                                         _alpha=0,
    #     #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #     #                                         selection_method="dwc-select")
    #     # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="ER",
    #     #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #     #                                         graph_num=10000,
    #     #                                         port=7112, hostname="128.40.41.48",
    #     #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                                         graph_function=nt.Tools.create_spatial_ER_graph,
    #     #                                         _alpha=0,
    #     #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #     #                                         selection_method="dwc-select")
    #     # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random", type="SNR-BA",
    #     #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #     #                                         graph_num=10000,
    #     #                                         port=7112, hostname="128.40.41.48",
    #     #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                                         graph_function=None,
    #     #                                         _alpha=5,
    #     #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #     #                                         selection_method="dwc-select")
    #     # parralel_random_select_graph_generation(graph_list, E=140, write=False, collection="ptd-random",
    #     #                                         type="prufer-sequence",
    #     #                                         notes="dwc-select method for large scale graphs, using varying traffic, saved under T_c matrix",
    #     #                                         graph_num=10000,
    #     #                                         port=7112, hostname="128.40.41.48",
    #     #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                                         graph_function=nt.Tools.prufer_sequence_ptd,
    #     #                                         _alpha=5,
    #     #                                         graph_function_args={"E": 140, "grid_graph": graph_list[0][0]},
    #     #                                         selection_method="dwc-select")
    #
    #     ## hostname="128.40.41.48"
    #     # small_scale_graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
    #     #                                                                 find_dic={"name": "NSFNET"},
    #     #                                                                 node_data=True)
    #     # parralel_random_select_graph_generation(small_scale_graph_list,
    #     #                                         E=small_scale_graph_list[0][0].number_of_edges(), write=True,
    #     #                                         collection="ptd-random",
    #     #                                         type="prufer-sequence",
    #     #                                         notes="Small scale random graphs for different traffic matrices, "
    #     #                                               "although this does not make a difference here. Generated via "
    #     #                                               "prufer-sequence method.",
    #     #                                         graph_num=10000,
    #     #                                         port=7112, hostname="128.40.41.48",
    #     #                                         gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                                         graph_function=nt.Tools.prufer_sequence_ptd,
    #     #                                         _alpha=5,
    #     #                                         graph_function_args={"E": small_scale_graph_list[0][0].number_of_edges(), "grid_graph": small_scale_graph_list[0][0]},
    #     #                                         selection_method="random")
    #     # DWC_select_graph_generation(small_scale_graph_list, E=small_scale_graph_list[0][0].number_of_edges(),
    #     #                             write=True, collection="prufer-select", type="prufer-sequence",
    #     #                             notes="Small scale dwc-select graphs for the prufer-sequence method, created from "
    #     #                                   "varying traffic matrices (locally skewed).",
    #     #                             graph_num=10000,
    #     #                             port=7112, hostname="128.40.41.48", gamma_t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #     #                             graph_function=nt.Tools.prufer_sequence_ptd,
    #     #                             graph_function_args={"N": len(small_scale_graph_list[0][0]), "E": small_scale_graph_list[0][0].number_of_edges(),
    #     #                                                  "grid_graph": small_scale_graph_list[0][0]}, selection_method="dwc-select")
    #     # exit()
    #
    #     ## throughput calculations for large scale: ER, BA, SNR-BA, prufer-sequence (random), ER, BA (dwc-select)
    #     # main(route_function="kSP-FF", collection="ptd-random",
    #     #      query={"kSP-FF Capacity": {"$exists": False}}, hostname="128.40.41.48", port=7112,
    #     #      heuristic=True, fibre_num=16,
    #     #      desc="ptd-random", max_count=1)
    #     # main(route_function="FF-kSP", collection="dwc-select",
    #     #      query={"FF-kSP Capacity": {"$exists": False}}, hostname="128.40.41.48", port=7112,
    #     #      heuristic=True, fibre_num=16,
    #     #      desc="dwc-select")
    #
    #     ## throughput calculations for small-scale: SNR-BA, BA, ER, ga
    #     # main(route_function="FF-kSP", collection="ta",
    #     #      query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},
    #     #             'data': {'$ne': 'new alpha'}, 'type': 'SNR-BA'}, hostname="128.40.41.48",
    #     #      port=7112, ILP=True, desc="SNR-BA small")
    #     # main(route_function="FF-kSP", collection="ta",
    #     #      query={'purpose': 'structural analysis', 'type': 'SBAG',
    #     #             'ILP-connections': {'$exists': True},
    #     #             'ILP-connections Capacity': {"$exists":False},
    #     #             'Demand Weighted Cost': {'$exists': True},
    #     #             "node order": "numeric", "alpha": 5}, hostname="128.40.41.48", port=7112,
    #     #      ILP=True, , desc="SBAG small")
    #     # main(route_function="FF-kSP", collection="ta",
    #     #      query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},'ILP-connections Capacity': {"$exists":False},
    #     #             'type': 'ER'}, hostname="128.40.41.48", port=7112,
    #     #      ILP=True, desc="ER small")
    #     # main(route_function="FF-kSP", collection="ta",
    #     #      query={'purpose': 'structural analysis', 'type': 'ER',
    #     #             'ILP-connections': {'$exists': True},
    #     #             'Demand Weighted Cost': {'$exists': True}}, hostname="128.40.41.48", port=7112,
    #     #      ILP=True, desc="")
    #     # main(route_function="FF-kSP", collection="ta",
    #     #      query={'purpose': 'ga-analysis', 'ILP-connections': {'$exists': True},
    #     #             'ILP-connections Capacity': {"$exists":False},
    #     #             'data': {'$ne': 'new alpha'}, 'type': 'BA'}, hostname="128.40.41.48",
    #     #      port=7112,
    #     #      ILP=True, desc="BA small")
    #     # main(route_function="FF-kSP", collection="ta",
    #     #      query={'purpose': 'structural analysis', 'type': 'BA',
    #     #             'ILP-connections': {'$exists': True},
    #     #             'ILP-connections Capacity': {"$exists":False},
    #     #             'Demand Weighted Cost': {'$exists': True},
    #     #             }, hostname="128.40.41.48",
    #     #      port=7112,
    #     #      ILP=True)
    #
    #     ## throughput calculations for ga-ps
    #     # main(route_function="FF-kSP", collection="HTD-distance",
    #     #      query={"FF-kSP Capacity":{"$exists":False}}, hostname="128.40.41.23", port=7112, heuristic=True,
    #     #      desc="Ruijie PTD throughput calc", fibre_num=16, max_count=30000)
    #     main(route_function="FF-kSP", collection="vector-ga-distance",
    #          query={"ILP-connections": {"$exists": False}}, hostname="128.40.41.23", port=7112, ILP=True,
    #          desc="Ruijie PTD throughput calc ILP", fibre_num=1, max_count=30000, max_time=int(3600*24*2))
    #     # main(route_function="FF-kSP", collection="MPNN-uniform-25-45-BA",
    #     #      query={"FF-kSP Capacity":{"$exists":False}}, hostname="128.40.41.48", port=7112, heuristic=True,
    #     #      desc="MPNN new 20 node data", fibre_num=4, max_count=30000)
    #     # main(route_function="FF-kSP", collection="MPNN-uniform-test",
    #     #      query={"nodes":{"$gte":10, "$lte":20},"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #     #      port=7112, heuristic=True,
    #     #      desc="N < 20 FF", fibre_num=1, max_count=30000, skip=3003)
    #     # main(route_function="kSP-FF", collection="MPNN-uniform-test",
    #     #      query={"nodes":{"$gte":10, "$lte":20},"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #     #      port=7112, heuristic=True,
    #     #      desc="N < 20 kSP", fibre_num=1, max_count=30000)
    #     # main(route_function="FF-kSP", collection="MPNN-uniform-25-45-test",
    #     #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #     #      port=7112, heuristic=True,
    #     #      desc="N < 45 FF", fibre_num=4, max_count=30000)
    #     # main(route_function="kSP-FF", collection="MPNN-uniform-25-45-test",
    #     #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #     #      port=7112, heuristic=True,
    #     #      desc="N < 45 kSP", fibre_num=4, max_count=30000)
    #     # main(route_function="FF-kSP", collection="MPNN-uniform-55-100-test",
    #     #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #     #      port=7112, heuristic=True,
    #     #      desc="N < 100 FF", fibre_num=16, max_count=30000)
    #     # main(route_function="kSP-FF", collection="MPNN-uniform-55-100-test",
    #     #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #     #      port=7112, heuristic=True,
    #     #      desc="N < 100 kSP", fibre_num=16, max_count=30000)
    #
    #     # main(route_function="FF-kSP", collection="prufer-select-ga",
    #     #      query={"fibre number":4}, hostname="128.40.41.48", port=7112, heuristic=True,
    #     #      desc="prufer-select-ga large", fibre_num=16)
    #     # main(route_function="FF-kSP", collection="topology-paper",
    #     #      query={"FF-kSP Capacity": {"$exists":}, "name":"ER", "nodes":{"$gte":25, "$lte":45}}, hostname="128.40.41.48", port=7112, heuristic=True,
    #     #      desc="prufer-select-ga large", fibre_num=4)
    #     # main(route_function="FF-kSP", collection="topology-paper",
    #     #      query={"FF-kSP Capacity": {"$exists": True}, "name": "BA", "nodes": {"$gte": 25, "$lte": 45}},
    #     #      hostname="128.40.41.48", port=7112, heuristic=True,
    #     #      desc="capacity calculations for MPNN", fibre_num=4)
    #
    #     # main(route_function="FF-kSP", collection="prufer-select-ga", query={"type": "prufer-select-ga"}, hostname="128.40.41.48",
    #     #      port=7112)
    #
    # #     ray.shutdown()
    # #     from NetworkToolkit import Data
    # #     graph_list = [1]
    # #     while len(graph_list) > 0:
    # #         NT = 400
    # #         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform", max_count=NT,
    # #                                                             find_dic={"average_ksp_cost": {"$exists": False}},
    # #                                                             parralel=True)
    # #         ray.init()
    # #         GP = Data.GraphProperties.remote()
    # #
    # #         funcs = [GP.update_m, GP.update_spanning_tree, \
    # #                  GP.update_algebraic_connectivity, GP.update_node_variance, GP.update_mean_internodal_distance, \
    # #                  GP.update_communicability_index, GP.update_comm_traff_ind, GP.update_graph_spectrum, \
    # #                  GP.update_shortest_path_cost, GP.update_limiting_cut_value]
    # #         #     ray.init()
    # #         #     for graph in graph_list:
    # #         #         for func in funcs:
    # #         #     #         print(func)
    # #         #             tempfunc = func
    # #         #             tasks.append(tempfunc.remote(graph,"MPNN-uniform"))
    # #         ray.get([func.remote(graph, "MPNN-uniform") for func in funcs for graph in tqdm(graph_list)])
    # #         #         tasks.append(GP.update_k_shortest_path_cost.remote(graph,"MPNN-uniform",4))
    # #         #         tasks.append(GP.update_weighted_spectrum_distribution.remote(graph, "MPNN-uniform",4,10))
    # #         #         tasks.append(GP.update_Dmax_value.remote(graph, "MPNN-uniform",156))
    # #         #     tasks.append(GP.update_Dmin_value.remote(graph, "MPNN-uniform",156))
    # #         #     ray.get(tasks)
    # #         # print(tasks)
    # #         ray.shutdown()
    # # print("waiting to start")
    # # for i in tqdm(range(12*3600), desc="starting in"):
    # #     time.sleep(1)
    #
    # # print("starting now")
    # # ILP_throughput(graph_list, max_time=6*3600, collection="topology-paper", workers=int(len(graph_list)/2), threads=1,
    # #                num_cpus=2)
    # # ILP_chromatic(graph_list, max_time=6*3600)
    # # ILP_throughput_scaled(5, 0.1)
    #      ILP=True)

    ## throughput calculations for ga-ps
    # main(route_function="FF-kSP", collection="HTD-distance",
    #      query={"FF-kSP Capacity":{"$exists":False}}, hostname="128.40.41.23", port=7112, heuristic=True,
    #      desc="Ruijie PTD throughput calc", fibre_num=16, max_count=30000)
    # main(route_function="FF-kSP", collection="vector-ga-distance",
    #      query={"ILP-connections": {"$exists": False}}, hostname="128.40.41.23", port=7112, ILP=True,
    #      desc="Ruijie PTD throughput calc ILP", fibre_num=1, max_count=30000, max_time=int(3600*24*2))
    # main(route_function="FF-kSP", collection="MPNN-uniform-25-45-BA",
    #      query={"FF-kSP Capacity":{"$exists":False}}, hostname="128.40.41.48", port=7112, heuristic=True,
    #      desc="MPNN new 20 node data", fibre_num=4, max_count=30000)
    # main(route_function="FF-kSP", collection="MPNN-uniform-test",
    #      query={"nodes":{"$gte":10, "$lte":20},"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #      port=7112, heuristic=True,
    #      desc="N < 20 FF", fibre_num=1, max_count=30000, skip=3003)
    # main(route_function="kSP-FF", collection="MPNN-uniform-test",
    #      query={"nodes":{"$gte":10, "$lte":20},"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #      port=7112, heuristic=True,
    #      desc="N < 20 kSP", fibre_num=1, max_count=30000)
    # main(route_function="FF-kSP", collection="MPNN-uniform-25-45-test",
    #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #      port=7112, heuristic=True,
    #      desc="N < 45 FF", fibre_num=4, max_count=30000)
    # main(route_function="kSP-FF", collection="MPNN-uniform-25-45-test",
    #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #      port=7112, heuristic=True,
    #      desc="N < 45 kSP", fibre_num=4, max_count=30000)
    # main(route_function="FF-kSP", collection="MPNN-uniform-55-100-test",
    #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #      port=7112, heuristic=True,
    #      desc="N < 100 FF", fibre_num=16, max_count=30000)
    # main(route_function="kSP-FF", collection="MPNN-uniform-55-100-test",
    #      query={"FF-kSP Capacity": {"$exists": True}}, hostname="128.40.41.48",
    #      port=7112, heuristic=True,
    #      desc="N < 100 kSP", fibre_num=16, max_count=30000)

    # main(route_function="FF-kSP", collection="prufer-select-ga",
    #      query={"fibre number":4}, hostname="128.40.41.48", port=7112, heuristic=True,
    #      desc="prufer-select-ga large", fibre_num=16)
    # main(route_function="FF-kSP", collection="topology-paper",
    #      query={"FF-kSP Capacity": {"$exists":}, "name":"ER", "nodes":{"$gte":25, "$lte":45}}, hostname="128.40.41.48", port=7112, heuristic=True,
    #      desc="prufer-select-ga large", fibre_num=4)
    # main(route_function="FF-kSP", collection="topology-paper",
    #      query={"FF-kSP Capacity": {"$exists": True}, "name": "BA", "nodes": {"$gte": 25, "$lte": 45}},
    #      hostname="128.40.41.48", port=7112, heuristic=True,
    #      desc="capacity calculations for MPNN", fibre_num=4)

    # main(route_function="FF-kSP", collection="prufer-select-ga", query={"type": "prufer-select-ga"}, hostname="128.40.41.48",
    #      port=7112)

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
