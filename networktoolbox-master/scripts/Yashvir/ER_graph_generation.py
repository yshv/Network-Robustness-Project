import networkx as nx

if __name__ == "__main__":
    nodes = 15
    p = 0.342
    desired = 36
    counter = 100
    while counter != 201:
        graph = nx.erdos_renyi_graph(nodes, p)
        edges = graph.number_of_edges()
        print(graph.number_of_edges())
        if edges == desired: 
            nx.write_gpickle(graph, "../TYP_code/networktoolbox-master/scripts/Yashvir/15_36_ER_data/{}({})-0_{}.gpickle".format(edges,counter,p))
            counter += 1
