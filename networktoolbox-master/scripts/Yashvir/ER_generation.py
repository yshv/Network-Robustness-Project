import NetworkToolkit as nt
import networkx as nx
import random
from datetime import datetime
import ray
import copy

@ray.remote
def ba_generation_distributed(nodes, p, edge_step, edge_count, graph_num):
    graph = ba_generation(nodes, p)
    nt.Database.insert_graph(graph, "Topology_Data", "robustness-sim-21", node_data=True, use_pickle=True, type="ER",
                             timestamp=datetime.utcnow(), graph_num=graph_num, edge_removal_count=edge_count, edge_removal_step=edge_step,
                             ER_p=p, edge_removal_idx=0)
    for i in range(1, edge_count+1):
        edges = list(graph.edges)
        choices = random.sample(edges, edge_step)
        graph_copy = copy.deepcopy(graph)
        graph_copy.remove_edges_from(choices)
        while nx.is_connected(graph_copy) != True or min([d for n, d in graph_copy.degree()])<=2:
            choices = random.sample(edges, edge_step)
            graph_copy = copy.deepcopy(graph)
            graph_copy.remove_edges_from(choices)
        nt.Database.insert_graph(graph_copy, "Topology_Data", "robustness-sim-21", node_data=True, use_pickle=True, type="ER",
                                 timestamp=datetime.utcnow(), graph_num=graph_num, edge_removal_count=edge_count, edge_removal_step=edge_step,
                                 ER_p=p, edge_removal_idx=i)
        graph = graph_copy




def ba_generation(nodes, p):
    graph = nx.erdos_renyi_graph(nodes, p)
    num_edges = graph.number_of_edges()
    while nx.is_connected(graph) != True or min([d for n, d in graph.degree()])<=2 or num_edges != 21:
        graph = nx.erdos_renyi_graph(nodes, p)
        num_edges = graph.number_of_edges()
    return graph

if __name__ == "__main__":
    nodes = 10
    graph_num = 20
    edge_step = 2
    edge_count = 3
    ray.init()
    tasks = [ba_generation_distributed.remote(nodes, 0.45, edge_step, edge_count, i) for i in range(0, graph_num)]
    ray.get(tasks)


