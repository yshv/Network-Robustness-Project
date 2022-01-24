from unittest import TestCase
import networkx as nx
import NetworkToolkit as nt

def create_random_coordinates(x_range=range(56,66), y_range=range(8,18), nodes=range(
    1,11)):
    """Method to create random coordinates for testing
    :param x_range: range for x coordinates
    :param y_range: range for y coordinates
    :param nodes: range of nodes to use
    :return: dictionary of dictionaries for node attributes to be fed to nx.graph and nodes
    """
    assert len(x_range) and len(y_range) == len(nodes)
    node_attr = {}
    for ind, node in enumerate(nodes):
        node_attr[node] = dict(x=x_range[ind], y=y_range[ind])
    return node_attr, list(nodes)
def simple_random_graph(nodes, edges):
    graph = nx.dense_gnm_random_graph(nodes, edges)
    graph.remove_node(0)
    node_attr = create_random_coordinates(x_range=range(0, nodes), y_range=range(0,
                                          nodes), nodes = range(1,nodes+1))[0]
    nx.set_node_attributes(graph, node_attr)
    return graph

def simple_graph():
    """
    Method to create a simple nx.Graph() for testing
    :return: nx.Graph()
    """
    graph = nx.Graph()
    node_attr, nodes = create_random_coordinates()
    edges = [(1, 2), (3, 1)]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, node_attr)
    return graph



def simple_multi_graph():
    """
    Method to create a simple nx.MultiGraph() for testing
    return: nx.MultiGraph()
    """
    graph = nx.MultiGraph()
    node_attr, nodes = create_random_coordinates()
    edges = [(1, 2), (1, 2), (3, 1)]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, node_attr)
    return graph

def test_real_graph():
    """
    Method to pull a real topology for testing purposes.
    :return: nx.Graph() from real database
    """
    graph = nt.Database.read_topology_dataset_list("Topology_Data", "real", find_dic={},
                                                   node_data=True)[0]
    print(graph)
    return graph[0]


class TestTopology(TestCase):
    def test_assign_distances_grid(self):
        # test that assures multi graph works as well
	
        graph = simple_multi_graph()
        graph_normal = simple_graph()
        top = nt.Topology.Topology()
        graph = top.assign_distances_grid(graph, harvesine=True)
        graph_normal = top.assign_distances_grid(graph_normal, harvesine=True)
        edge_data = graph.edges.data()
        edge_data_normal = graph_normal.edges.data()
        # print(edge_data)
        # print(edge_data_normal)
        self.assertEqual(list(edge_data)[0], list(edge_data_normal)[0])
        self.assertEqual(list(edge_data)[-1], list(edge_data_normal)[-1])

    def test_create_real_based_grid_graph_normal(self):
        graph = test_real_graph()
        top = nt.Topology.Topology()
        choice_prob = top.choice_prob(SBAG=True, alpha=1)
        graph = top.create_real_based_grid_graph(graph, len(list(graph.edges)),
                                                 database_name="Topology",
                                                 collection_name="test")


    def test_create_real_based_grid_Graph_alternating_pref(self):
        graph = test_real_graph()
        top = nt.Topology.Topology()
        choice_prob = top.choice_prob(SBAG=True, alpha=1, normalise=True)
        graph = top.create_real_based_grid_graph(graph, len(list(graph.edges)),
                                                 database_name="Topology",
                                                 collection_name="test",
                                                 alternating_pref_random=True)
    def test_get_closest_nodes(self):
        graph = simple_graph()
        closest = nt.Topology.get_closest_nodes(graph, 1, closet_num=2)


    def test_choose_source_node_sequential(self):
        graph1 = simple_random_graph(10, 15)
        graph2 = simple_random_graph(5, 10)
        source = nt.Topology.choose_source_node_sequential(graph1, graph2)


    def test_gabriel_threshold(self):
        top = nt.Topology.Topology()
        graph = simple_random_graph(30, 5)
        edge_choice_num = range(1,4)
        for m in edge_choice_num:
            for source in graph.nodes:
                j1 = nt.Topology.N_D(source, list(graph.nodes), graph, m,
                                    top.choice_prob(SBAG=True, alpha=1))
                N_D = nt.Topology.gabriel_threshold(nt.Topology.N_D)
                j2 = N_D(source, list(graph.nodes), graph, m, top.choice_prob(
                    SBAG=True, alpha=1))

                for j in j1:
                    self.assertNotIn(j, list(graph.neighbors(source)))
                for j in j2:
                    self.assertNotIn(j, list(graph.neighbors(source)))
    def test_relative_neighborhood_threshold(self):
        top = nt.Topology.Topology()
        graph = simple_random_graph(30, 5)
        edge_choice_num = range(1,4)
        for m in edge_choice_num:
            for source in graph.nodes:
                j1 = nt.Topology.N_D(source, list(graph.nodes), graph, m,
                                    top.choice_prob(SBAG=True, alpha=1))
                N_D = nt.Topology.relative_neighbourhood_threshold(nt.Topology.N_D)
                j2 = N_D(source, list(graph.nodes), graph, m, top.choice_prob(
                    SBAG=True, alpha=1))
                for j in j1:
                    self.assertNotIn(j, list(graph.neighbors(source)))
                for j in j2:
                    self.assertNotIn(j, list(graph.neighbors(source)))

    def test_grabriel_constraint_real_based(self):
        nodes = [10, 15, 20, 29]
        nodes = range(10, 30)
        edges = range(15, 40)
        #edges = [15, 25, 35, 40]
        top = nt.Topology.Topology()

        for node_num, edge_num in zip(nodes, edges):
            graph = nt.Topology.scatter_nodes(node_num, _mean=[30, 90], _std=[3, 16])
            #graph = simple_random_graph(node_num, edge_num)
            print("graph created")
            graph = top.create_real_based_grid_graph(graph, edge_num,
                                                     database_name="Topology",
                                                     collection_name="real",
                                                     gabriel_constraint=True, alpha=1,
                                                     SBAG=True, normalise=True)


    def test_relative_neighborhood_constraint_real_based(self):
        nodes = range(10, 30)
        edges = range(15, 40)
        # edges = [15, 25, 35, 40]
        top = nt.Topology.Topology()

        for node_num, edge_num in zip(nodes, edges):
            print(node_num)
            print(edge_num)
            graph = nt.Topology.scatter_nodes(node_num, _mean=[30, 90], _std=[3, 16])
            # graph = simple_random_graph(node_num, edge_num)
            print("graph created")
            graph = top.create_real_based_grid_graph(graph, edge_num,
                                                     database_name="Topology",
                                                     collection_name="real",
                                                     relative_neighbourhood_constraint=True,
                                                     alpha=1, SBAG=True, normalise=True)

    def test_random_start_node_real_based(self):
        nodes = 10
        edges = 15
        top = nt.Topology.Topology()
        graph = nt.Topology.scatter_nodes(nodes, _mean=[30, 90], _std=[3, 16])
        start_node_same = 0
        for i in range(30):

            graph_1 = top.create_real_based_grid_graph(graph, edges,
                                                       database_name="Topology",
                                                       collection_name="real",
                                                       relative_neighbourhood_constraint=False,
                                                       alpha=1, SBAG=True,
                                                       normalise=True,
                                                       random_start_node=True)
            graph_2 = top.create_real_based_grid_graph(graph, edges,
                                                       database_name="Topology",
                                                       collection_name="real",
                                                       relative_neighbourhood_constraint=False,
                                                       alpha=1, SBAG=True,
                                                       normalise=True,
                                                       random_start_node=True)
            if list(graph_1.nodes)[0] == list(graph_2.nodes)[0]:
                start_node_same += 1

        self.assertLess(start_node_same, 10)

    def test_centroid_start_node_real_based(self):
        nodes = 10
        edges = 15
        top = nt.Topology.Topology()
        graph = nt.Topology.scatter_nodes(nodes, _mean=[30, 90], _std=[3, 16])
        start_node_same = 0
        for i in range(30):
            start_node1 = nt.Topology.choose_starting_node_centroid(graph)
            start_node2 = nt.Topology.choose_starting_node_centroid(graph)
            self.assertEqual(start_node1, start_node2)


    def test_SNR_ratio_triangle_edge_removal_real_based(self):
        nodes = 10
        edges = 20
        top = nt.Topology.Topology()
        graph = nt.Topology.scatter_nodes(nodes, _mean=[30, 90], _std=[3, 16])
        graph = top.create_real_based_grid_graph(graph, edges, database_name="Topology",
                                                 collection_name="real",
                                                 relative_neighbourhood_constraint=False,
                                                 alpha=1, SBAG=True, normalise=True,
                                                 overshoot=True, undershoot=False,
                                                 random_start_node=True,
                                                 remove_unfeasible_edges=True)


    def test_remove_nodes(self):
        graph = simple_random_graph(10, 25)
        edges_before = len(list(graph.edges))
        def edges_func(graph): return list(graph.edges)
        nt.Topology.remove_edges(graph, 3, remove_func=edges_func)
        edges_after = len(list(graph.edges))
        self.assertEqual(edges_before-3, edges_after)

    def test_node_order(self):
        nodes = 10
        edges = 15
        top = nt.Topology.Topology()
        graph = nt.Topology.scatter_nodes(nodes, _mean=[30, 90], _std=[3, 16])

        node_order1 = nt.Topology.create_node_order(graph)
        node_order2 = nt.Topology.create_node_order(graph, random_adding=False,
                                      sequential_adding=True)
        print(node_order1)
        print(node_order2)

    def test_CONUS_graph_creation(self):
        CONUS_graphs = nt.Database.read_topology_dataset_list("Topology_Data",
                                                              "real", find_dic={
                                                                "source":"CONUS"
                                                                                },
                                                              node_data=True)
        top = nt.Topology.Topology()
        for graph, _id in CONUS_graphs:
            graph = top.create_real_based_grid_graph(graph, len(list(graph.edges)),
                                                     database_name="Topology_Data",
                                                     collection_name="waxman",
                                                     sequential_adding=True, alpha=1,
                                                     SBAG=True, undershoot=True)
            print("done")


