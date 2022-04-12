import networkx as nx

if __name__ == "__main__":
    nodes = 10
    p = 0.45
    desired = 21
    counter = 1
    while counter != 100:
        graph = nx.erdos_renyi_graph(nodes, p)
        edges = graph.number_of_edges()
        print(graph.number_of_edges())
        if edges == desired: 
            nx.write_gpickle(graph, "../TYP_code/networktoolbox-master/scripts/Yashvir/15_36_ER_data/{}({})-0_{}.gpickle".format(edges,counter,p))
            counter += 1
