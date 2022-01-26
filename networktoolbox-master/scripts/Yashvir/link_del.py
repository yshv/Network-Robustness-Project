import NetworkToolkit as nt
import networkx as nx
import time
import random


link_remove = 1 

if __name__ == "__main__":

    nodes=15
    p_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lambda_list = []
    for p in p_list:
        graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/Data/graph_{}_{}.gpickle".format(nodes,p))
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

        remove = random.sample(graph.edges(), link_remove)
        graph.remove_edges_from(remove)\

        nx.write_gpickle(graph, "/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/Data/graph_{}-{}_{}.gpickle".format(nodes,link_remove,p))






