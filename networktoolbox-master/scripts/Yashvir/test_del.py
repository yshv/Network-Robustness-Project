import NetworkToolkit as nt
import networkx as nx
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    nodes=20
    p_list = [0.3]
    lambda_list = []
    for p in p_list:
        graph = nx.read_gpickle("../TYP_Code/networktoolbox-master/scripts/Yashvir/Data/graph_{}_{}.gpickle".format(nodes,p))
        graph_1 = nx.read_gpickle("../TYP_Code/networktoolbox-master/scripts/Yashvir/Data/graph_{}-3_{}.gpickle".format(nodes,p))
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        print(graph.number_of_edges())
        print(graph_1.number_of_edges())
        plt.figure(1)
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.figure(2)
        nx.draw(graph_1, with_labels=True, font_weight='bold')
        plt.show()