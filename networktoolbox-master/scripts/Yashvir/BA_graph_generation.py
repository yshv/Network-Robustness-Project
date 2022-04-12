import NetworkToolkit as nt
import networkx as nx
import time
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    nodes = 10
    for i in range(101, 102):
        graph = nx.barabasi_albert_graph(nodes, 3, seed=None, initial_graph=None)
        print(graph.number_of_edges())
        nx.write_gpickle(graph, "../TYP_code/networktoolbox-master/scripts/Yashvir/test_data/{}({})-0_3.gpickle".format(nodes,i))


