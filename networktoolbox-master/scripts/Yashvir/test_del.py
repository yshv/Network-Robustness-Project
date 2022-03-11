import NetworkToolkit as nt
import networkx as nx
import time
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    nodes = 10
    edges = 15
    p = 0.3
    lambda_list = []
    for i in range(1,11):
        graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/test_data/15({})-0_0.3.gpickle".format(i))
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

        for x in range (1, 3):
            for y in range(1, 3):
                bridges = list(nx.bridges(graph))
                sample = random.sample(graph.edges, 1)
                check =  any(item in bridges for item in sample)
                while(check == True):
                    sample = random.sample(graph.edges, 1)
                    check =  any(item in bridges for item in sample)
                    
                graph.remove_edges_from(sample)

            nx.write_gpickle(graph, "/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/test_data/15({})-{}_0.3.gpickle".format(i,x*2))

