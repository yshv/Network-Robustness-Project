from unittest import TestCase
import NetworkToolkit as nt

class TestILP(TestCase):
    def test_maximise_uniform_bandwidth_demand(self):
        graph = nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                                       find_dic={})
        graph = graph[0][0]
        network = nt.Network.OpticalNetwork(graph)
        network.rwa.maximise_uniform_bandwidth_demand()

