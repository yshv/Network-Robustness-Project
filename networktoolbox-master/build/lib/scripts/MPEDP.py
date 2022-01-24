import NetworkToolkit as nt
import networkx as nx
import numpy as np


if __name__=="__main__":
    kwargs = {"computational scaling data":1, "date written":"12/01/2021"}
    nodes = list(range(10,51,5))
    num_graphs = 10
    for node_num in nodes:
        for i in range(num_graphs):
            k_regular_graph = nx.generators.random_regular_graph(4, node_num)
            nt.Database.insert_graph(k_regular_graph, "Topology_Data", "MPEDP", use_pickle=True, **kwargs)
