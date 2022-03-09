import NetworkToolkit as nt
import networkx as nx
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for x in range(1, 11):
        for i in range (0, 14, 2):
            graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/15_36_ER_data/36({})-{}_0.342.gpickle".format(x,i))
            graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
            print(graph.number_of_edges())
            nx.draw(graph)
            plt.show()