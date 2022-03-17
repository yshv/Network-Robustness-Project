import NetworkToolkit as nt
import networkx as nx
import random
from datetime import datetime
import ray
import copy

@ray.remote
def ba_generation_distributed(nodes, m, edge_step, edge_count, graph_num):
    graph = ba_generation(nodes, m)
    nt.Database.insert_graph(graph, "Topology_Data", "robustness-sim-test", node_data=True, use_pickle=True, type="BA",
                             timestamp=datetime.utcnow(), graph_num=graph_num, edge_removal_count=edge_count, edge_removal_step=edge_step,
                             BA_m=m, edge_removal_idx=0)
    for i in range(1, edge_count+1):
        edges = list(graph.edges)
        choices = random.sample(edges, edge_step)
        graph_copy = copy.deepcopy(graph)
        graph_copy.remove_edges_from(choices)
        while nx.is_connected(graph_copy) != True or min([d for n, d in graph_copy.degree()])<=2:
            choices = random.sample(edges, edge_step)
            graph_copy = copy.deepcopy(graph)
            graph_copy.remove_edges_from(choices)
        nt.Database.insert_graph(graph_copy, "Topology_Data", "robustness-sim-test", node_data=True, use_pickle=True, type="BA",
                                 timestamp=datetime.utcnow(), graph_num=graph_num, edge_removal_count=edge_count, edge_removal_step=edge_step,
                                 BA_m=m, edge_removal_idx=i)
        graph = graph_copy




def ba_generation(nodes, m):
    graph = nx.barabasi_albert_graph(nodes, m)
    while nx.is_connected(graph) != True or min([d for n, d in graph.degree()])<=2:
        graph = nx.barabasi_albert_graph(nodes, m)
    return graph

if __name__ == "__main__":
    nodes = 15
    graph_num = 500
    edge_step = 2
    edge_count = 10
    ray.init()
    tasks = [ba_generation_distributed.remote(nodes, 5, edge_step, edge_count, i) for i in range(0, graph_num)]
    ray.get(tasks)


