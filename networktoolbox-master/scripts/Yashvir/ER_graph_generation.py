import networkx as nx

if __name__ == "__main__":
    nodes = 15
    p = 0.3
    for i in range(1,6):
        graph = nx.erdos_renyi_graph(nodes, p)
        print(graph.number_of_edges())
        nx.write_gpickle(graph, "../TYP_code/networktoolbox-master/scripts/Yashvir/ER_data/{}({})-0_{}.gpickle".format(nodes,i,p))