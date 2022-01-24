from unittest import TestCase
import NetworkToolkit as nt

class TestNetworkSimulator(TestCase):
    def test_incremental_uniform_demand_simulation(self):
        self.fail()

    def test_incremental_uniform_demand_simulation_graph_list(self):
        simulator = nt.NetworkSimulatorParralel.NetworkSimulator()
        graphs = nt.Database.read_topology_dataset_list("Topology_Data", "real",
                                            find_dic={"nodes": {
                                                "$lt": 16}})
        simulator.incremental_uniform_demand_simulation_graph_list(graphs)
