from unittest import TestCase
import networkx as nx
import NetworkToolkit as nt

class TestTopology(TestCase):
    def test_create_real_based_topologies(self):
        collection_name = "test"
        db_name="Topology_Data"
        _node_range = list(range(10, 40, 5))
        nt.Data.create_real_based_topologies(db_name=db_name, collection_name=collection_name,
                                                              node_range=_node_range,
                                                              SBAG = True, alpha=100)
                                                              

                                                              
                                                