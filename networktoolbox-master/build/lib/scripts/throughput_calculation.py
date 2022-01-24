import NetworkToolkit as nt
import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import ray
import numpy as np
from tqdm import tqdm
import pickle
import dill
import progressbar as pbar
from asyncio import Event
from typing import Tuple
from time import sleep

import ray
# For typing purposes
from ray.actor import ActorHandle
from tqdm import tqdm


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter
    
class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return
            
            
@ray.remote
def calculate_throughput(graph, _id, rwa, pba, collection=None, db="Topology_Data", pb=None):
#     top = nt.Topology.Topology()
#     graph = top.assign_distances_grid(graph, pythagorus=False,
#                                            harvesine=True)
    network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
    network.physical_layer.add_wavelengths_to_links(rwa)
    network.physical_layer.add_uniform_launch_power_to_links(network.channels)
#     network.physical_layer.add_uniform_optimised_launch_power_to_links(network.channels)
    network.physical_layer.add_non_linear_NSR_to_links()
    throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa)[0]
    nt.Database.update_data_with_id(db, collection, _id,
                                                newvals={"$set": {"ILP-connections Capacity":throughput[0]}})
#     if pb is not None:
#         pb = dill.loads(pb)
    pba.update.remote(1)
#     print(throughput)
    
def calculate_throughput_parralel(graph_list, collection=None):
    ray.shutdown()
    # port = 7111
    # hostname = "128.40.41.14"
    # ray.init(address='{}:{}'.format(hostname, port))
    ray.init()
    print("starting throughput calculation...")
#     pb = tqdm(total=len(graph_list))
    pb = ProgressBar(len(graph_list))
#     with tqdm(total=len(graph_list)) as pb:
#     print(type(pb))
#     pb = dill.dumps(pb)
#     print(pb)
    actor = pb.actor
    tasks = [calculate_throughput.remote(graph, _id, rwa, actor, collection=collection) for graph, _id, rwa in tqdm(graph_list, desc="adding tasks")]
    pb.print_until_done()
    tasks = ray.get(tasks)
#     for graph, _id, rwa in graph_list:
#         network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
#         network.physical_layer.add_wavelengths_to_links(rwa)
#         network.physical_layer.add_uniform_optimised_launch_power_to_links(network.channels)
#         network.physical_layer.add_non_linear_NSR_to_links()
#         throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment[i])[0]
#         print(throughput)
def assign_distances(graph, grid_graph, write=False, collection=None, db="Topology_Data", _id=None):
    if 0 in graph.nodes():
        graph = nx.relabel_nodes(graph, lambda x: x+1) 
    
    nx.set_node_attributes(graph, dict(grid_graph.nodes.data()))
#     print(dict(grid_graph.nodes.data()))
#     print(graph.nodes.data())
    top = nt.Topology.Topology()
    graph = top.assign_distances_grid(graph, pythagorus=False,
                                           harvesine=True)
    if write:
        topology_data = nt.Tools.graph_to_database_topology(graph)
        node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
        nt.Database.update_data_with_id(db, collection, _id,
                                                newvals={"$set": {"topology data":topology_data, 
                                                                  "node data":node_data}})
    return graph


if __name__ == "__main__":
    
    grid_graph = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", find_dic={"name":"NSFNET"},use_pickle=True, node_data=True)[0][0]

    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ta","ILP-connections RWA",find_dic={'purpose':'structural analysis','type':'SBAG',
                                                                  'ILP-connections':{'$exists':True},
                                                                  'Demand Weighted Cost':{'$exists':True},"ILP-connections Capacity":{'$exists':False},
                                                               "node order":"numeric", "alpha":5}, use_pickle=True)
    graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "ta","ILP-connections RWA", find_dic={'purpose':'structural analysis','type':'ER',
                                                                  'ILP-connections':{'$exists':True},"ILP-connections Capacity":{'$exists':False},
                                                                  'Demand Weighted Cost':{'$exists':True}}, use_pickle=True)
    graph_list += nt.Database.read_topology_dataset_list("Topology_Data", "ta","ILP-connections RWA", find_dic={'purpose':'structural analysis','type':'BA',
                                                                  'ILP-connections':{'$exists':True},"ILP-connections Capacity":{'$exists':False},
                                                                  'Demand Weighted Cost':{'$exists':True}}, use_pickle=True)
    
    calculate_throughput_parralel(graph_list, collection="ta")

