from unittest import TestCase
import NetworkToolkit as nt
import numpy as np
import networkx as nx

class TestHeuristics(TestCase):
    def test_edge_disjoint_routing(self):
        graph = nt.Database.read_topology_dataset_list("Topology_Data", "test-ILP", find_dic={}, max_count=150)[142][0]
        print("nodes: {}".format(len(graph)))
        network = nt.Network.OpticalNetwork(graph)
        # connection_requests = network.demand.create_uniform_connection_requests(1)
        connection_requests = np.zeros((len(graph),len(graph)))
        connection_requests[0][1] = 1
        connection_requests[1][0] = 1
        print("connection requests: \n{}".format(connection_requests))


        network.rwa.edge_disjoint_routing(graph, connection_requests, T=250)
    def test_FF(self):
        graph = nx.Graph()
        nodes = [1,2,3,4]
        edges = [(1,2), (2,3), (3,4), (4,1)]
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        heuristic = nt.Routing.Heuristics.Heuristics(graph, 10, 32*1e9)
        rwa = {0:[[1,2,3]], 1:[[2,3,4]], 2:[[3,2,1]], 3:[[2,3,4]], 4:[]}
        W = np.zeros((len(list(graph.edges)), heuristic.channels))
        for key in rwa:
            for path in rwa[key]:
                W = heuristic.add_wavelength_path_to_W(graph, W, path, key)
        new_path = [3,2,1]
        FF = heuristic.FF_return(heuristic.graph, heuristic.channels, W, new_path)
        print("FF: {}".format(FF))
        self.assertEqual(FF, 2)





