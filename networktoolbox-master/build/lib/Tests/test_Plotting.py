from unittest import TestCase
import NetworkToolkit as nt

class Test(TestCase):
    def test_plot_graph_google_earth(self):
        graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "real", find_dic={}, node_data=True)

        for graph, _id in graph_list:
            nt.Plotting.plot_graph_google_earth(graph, "/Users/robin/Desktop", name=_id)




