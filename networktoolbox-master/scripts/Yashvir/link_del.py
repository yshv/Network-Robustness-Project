import NetworkToolkit as nt
import networkx as nx
import time
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    nodes = 15
    p = 0.3
    lambda_list = []
    for i in range(1,6):
        graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/BA_Data/{}({})_3.gpickle".format(nodes,i))
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

        for x in range (1, 7):
            for y in range(1, 3):
                bridges = list(nx.bridges(graph))
                sample = random.sample(graph.edges, 1)

                if sample in bridges:
                    sample = random.sample(graph.edges, 1)
                else:
                    graph.remove_edges_from(sample)

            nx.write_gpickle(graph, "/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/BA_Data/15({})-{}_3.gpickle".format(i,x*2))





