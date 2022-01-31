import NetworkToolkit as nt
import networkx as nx
import time
import random

if __name__ == "__main__":

    nodes=20
    p_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lambda_list = []
    for p in p_list:
        graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/Data/graph_{}_{}.gpickle".format(nodes,p))
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

        for i in range(1, 6):
            graph.remove_edges_from(random.sample(graph.edges(), i))
            list(nx.isolates(graph))
            graph.remove_nodes_from(list(nx.isolates(graph)))
            nx.write_gpickle(graph, "/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/Data/graph_{}-{}_{}.gpickle".format(nodes,i,p))






