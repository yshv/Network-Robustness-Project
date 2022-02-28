import NetworkToolkit as nt
import networkx as nx
import time
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    nodes = 15
    p = 0.3
    lambda_list = []

    graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/ER_Data/15(1)_0.3.gpickle")
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    nx.draw(graph)
    print(graph.number_of_edges())
    for x in range (1, 7):
        for y in range(1, 3):
            bridges = list(nx.bridges(graph))
            sample = random.sample(graph.edges, 1)

            if sample in bridges:
                sample = random.sample(graph.edges, 1)
            else:
                graph.remove_edges_from(sample)

        nx.write_gpickle(graph, "/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/ER_Data/15(1)-{}_0.3.gpickle".format(x*2))