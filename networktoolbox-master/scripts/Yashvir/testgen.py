import networkx as nx

if __name__ == "__main__":
    nodes = 10
    p = 0.3
    desired = 15
    counter = 1
    while counter != 21:
        graph = nx.erdos_renyi_graph(nodes, p)
        edges = graph.number_of_edges()
        print(graph.number_of_edges())
        if edges == desired: 
            nx.write_gpickle(graph, "../TYP_code/networktoolbox-master/scripts/Yashvir/test_data/{}({})-0_{}.gpickle".format(edges,counter,p))
            counter += 1
