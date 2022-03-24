import NetworkToolkit as nt
import time
from datetime import datetime
import ray
import networkx as nx
num_cpus = 1


@ray.remote(num_cpus=num_cpus)
def phy_conn_distributed(graph, _id, max_time=3600, e=20, k=20, threads=1, node_file_start=0.01,
                           db=None, collection=None, actor=None):
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    network = nt.Network.OpticalNetwork(graph, channel_bandwidth=170*1e9)
    print("nodes: {}".format(graph.nodes))
    edges = nx.number_of_edges(graph)
    nodes = nx.number_of_nodes(graph)
    phy_conn = (edges)/(nodes*(nodes-1))
    nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"phy_conn": phy_conn}})
    if actor is not None:
        actor.update.remote(1)
        
if __name__== "__main__":

    hostname = "128.40.41.48"
    port = 7112
    # ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)
    ray.init()
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "robustness-sim-basic", find_dic={"phy_conn":{"$exists":False}})
    pb = nt.Tools.ProgressBar(len(graph_list))
    actor = pb.actor
    tasks = [phy_conn_distributed.remote(graph, _id, db="Topology_Data", collection="robustness-sim-basic", actor=actor, max_time=3600, threads=num_cpus) for graph, _id in graph_list]
    pb.print_until_done()
    ray.get(tasks)
