from unittest import TestCase
import unittest
import NetworkToolkit as nt
import HTD_script as htd
import networkx as nx

# from .HTD_script import htd_network_design, decide_gate_nodes_set, decide_gate_nodes_all, \
# determine_inter_subnetwork_gate_edges_set, determine_inter_subnetwork_gate_edges_all, subnetwork_partition, \
# edge_partition
graph, _id, T_c = nt.Database.read_topology_dataset_list("Topology_Data", "MPNN-uniform-25-45",
                                                         "T_c",
                                                         node_data=True,
                                                         find_dic={"FF-kSP Capacity": {
                                                             "$exists": True},
                                                             "nodes": 25}, max_count=1,
                                                         use_pickle=True)[0]
E = 35
N_sub = 5


class Test(TestCase):

    def test_subnetwork_partition(self):
        coordinate = htd.subcenter_generation(N_sub)
        subnetworks = htd.subnetwork_partition(graph, N_sub, coordinate)
        for subnetwork in subnetworks:
            self.assertIs(len(subnetwork), int(len(graph) / N_sub))
        # self.subnetworks = subnetworks

    def test_edge_partition(self):
        coordinate = htd.subcenter_generation(N_sub)
        subnetworks = htd.subnetwork_partition(graph, N_sub, coordinate)
        edge_matrix = htd.edge_partition(subnetworks, E, len(subnetworks), T_c)
        edge_matrix_sum = edge_matrix.sum()
        E_debug = E
        self.assertEqual(edge_matrix.sum(), E, msg="Edge matrix matches up with original number of edges wanted.")

    def test_decide_gate_nodes_set(self):
        coordinate = htd.subcenter_generation(N_sub)
        subnetworks = htd.subnetwork_partition(graph, N_sub, coordinate)
        edge_matrix = htd.edge_partition(subnetworks, E, len(subnetworks), T_c)
        gate_nodes = htd.decide_gate_nodes_set(subnetworks[0], subnetworks[1], T_c, edge_matrix[0, 1])
        # self.assertEqual(edge_matrix[0, 1], len(gate_nodes),
        #                   msg="number of gate nodes matches the edge numbers chosen.")
        for node in gate_nodes:
            self.assertIn(node, list(subnetworks[0].nodes), msg="gatenodes are in desired subnetwork")
            self.assertIn(node, list(graph.nodes))

    def test_decide_gate_nodes_all(self):
        coordinate = htd.subcenter_generation(N_sub)
        subnetworks = htd.subnetwork_partition(graph, N_sub, coordinate)
        edge_matrix = htd.edge_partition(subnetworks, E, len(subnetworks), T_c)
        all_gatenodes = htd.decide_gate_nodes_all(graph, subnetworks, T_c, edge_matrix)
        for ind, gatenodes in enumerate(all_gatenodes):
            for item in gatenodes:
                # self.assertIn(item, list(subnetwork.nodes), msg="gatenodes {} are in the subnetwork.".format(ind))
                self.assertIn(item, list(graph.nodes), msg="gatenodes {} are in the original graph.".format(ind))

    def test_determine_inter_subnetwork_gate_edges_all(self):
        design_graph = nx.Graph()
        coordinate = htd.subcenter_generation(N_sub)
        subnetworks = htd.subnetwork_partition(graph, N_sub, coordinate)
        edge_matrix = htd.edge_partition(subnetworks, E, len(subnetworks), T_c)
        all_gatenodes = htd.decide_gate_nodes_all(graph, subnetworks, T_c, edge_matrix)
        design_graph = htd.determine_inter_subnetwork_gate_edges_all(design_graph, subnetworks, all_gatenodes,
                                                                     edge_matrix)
        edge_numbers = sum(
            [edge_matrix[i, j] for i in range(len(subnetworks)) for j in range(len(subnetworks)) if
             i != j])
        self.assertEqual(len(design_graph.edges), int(edge_numbers / 2),
                         msg="edge numbers added for inter network connections match those decided earlier.")

    def test_htd_network_design(self):
        design_graph = htd.htd_network_design(graph, E, T_c, 4, [1])
        self.assertIs(nx.is_connected(design_graph), True)
        self.assertIs(len(list(design_graph.edges)), E)



if __name__ == '__main__':
    unittest.main()




