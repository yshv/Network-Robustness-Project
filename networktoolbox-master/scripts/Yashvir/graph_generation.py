import networkx as nx

if __name__ == "__main__":
    nodes=15
    p_list = [0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9]
    for p in p_list:
        graph = nx.erdos_renyi_graph(nodes, p)
        print(graph.number_of_edges())
        nx.write_gpickle(graph, "/home/uceeatz/Code/networktoolbox/scripts/Yashvir/Data/graph_{}_{}.gpickle".format(nodes,p))


        