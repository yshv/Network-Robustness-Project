import NetworkToolkit as nt
import time
from datetime import datetime
import ray
import networkx as nx
num_cpus = 1


def static_ilp_distributed(graph, _id, max_time=3600, e=20, k=20, threads=1, node_file_start=0.01,
                           db=None, collection=None, actor=None):
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    network = nt.Network.OpticalNetwork(graph, channel_bandwidth=170*1e9)
    print("nodes: {}".format(graph.nodes))
    time_start = time.perf_counter()
    data = network.rwa.static_ILP(min_wave=True, max_time=max_time, e=e, k=k, threads=threads,
                                  node_file_start=node_file_start)
    time_taken = time.perf_counter()-time_start
    connectivity = nx.edge_connectivity(graph)
    diameter = nx.diameter(graph)
    alge_con = nx.algebraic_connectivity(graph)
    max_edge_conn = max(nx.edge_betweenness_centrality(graph).values()) 
    nt.Database.update_data_with_id(db, collection, _id, newvals={"$set": {"lambda_r":data["objective"],
                                                                           "lambda_r optimisation status": str(data["status"]),
                                                                           "lambda_r time": time_taken,
                                                                           "lambda_r max time":max_time,
                                                                           "lambda_r e": e,
                                                                           "lambda_r k": k,
                                                                           "lambda_r threads":threads,
                                                                           "lambda_r node_file_start":node_file_start,
                                                                           "lambda_r timestamp": datetime.utcnow(),
                                                                           "edge conn": connectivity,
                                                                           "diamter": diameter,
                                                                           "algebraic connectivity": alge_con,
                                                                           "max edge": max_edge_conn
                                                                           }})
    if actor is not None:
        actor.update.remote(1)
if __name__== "__main__":

    hostname = "128.40.41.48"
    port = 7112
    # ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "robustness-sim-test-test", find_dic={"lambda_r":{"$exists":False}})
    for graph, _id in graph_list:
        static_ilp_distributed(graph, _id, db="Topology_Data", collection="robustness-sim-test-test", max_time=3600, threads=num_cpus)

