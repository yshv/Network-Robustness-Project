from unittest import TestCase
import NetworkToolkit as nt
import numpy as np

class TestPhysicalLayer(TestCase):
    def test_get_snr_non_linear(self):
        graph = nt.Database.read_topology_dataset_list("Topology_Data", "real", find_dic={"source":"CONUS"},
                                                       max_count=10)
        graph = graph[1][0]
        print("number of edges: {}".format(len(list(graph.edges))))
        network = nt.Network.OpticalNetwork(graph)
        network.physical_layer.add_wavelengths_full_occupation(network.channels)
        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_non_linear_NSR_to_links()
        demand_matrix = np.ones((len(graph), len(graph)))
        nt.Tools.get_h_brute_force_non_uniform(graph, demand_matrix)