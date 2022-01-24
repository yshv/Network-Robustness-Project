import sys
sys.path.append("/home/uceeatz/Code/geneticalgorithm")
from geneticalgorithm import geneticalgorithm as ga
# import geneticalgorithm as ga
import NetworkToolkit as nt
import networkx as nx
import numpy as np
import ray


class PTD():
    '''
    Class of GA topology design
    :param N: number of nodes
    :param E: number of edges
    :param T_c: normalized traffic matrix
    :param alpha: weight list for kth shortest paths
    :param grid_graph: the graph used to decide node locations
    :return dictionary: {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}
    '''
    def __init__(self, N, E, T_c, alpha,grid_graph):
        self.N = N
        self.E = E
        self.T_c = [T_c]
        self.alpha = alpha
        self.top = nt.Topology.Topology()
        self.solution_graph = None
        self.algorithm_param =  {'max_num_iteration': 500,
                                   'population_size':500,
                                   'mutation_probability':0.1,
                                   'elit_ratio': 0.01,
                                   'crossover_probability': 0.8,
                                   'parents_portion': 0.3,
                                   'crossover_type':'uniform',
                                   'max_iteration_without_improv':None}
        self.grid_graph = grid_graph

    def build_graph_from_vector(self, Prufer_sq):
#         edges_poss = [(i + 1, j + 1) for i in range(self.N) for j in range(self.N) if j > i]
#         edges = []
#         for ind, item in enumerate(graph_vector):
#             if item == True:
#                 edges.append(edges_poss[ind])
        graph = nx.Graph()
        graph.add_nodes_from(list(np.arange(1,N+1)))
        All_node = np.arange(1,N+1)

#         while True:
#         Prufer_sq = list(np.random.randint(1,N+1,size = N-2))
        Prufer_sq = list(Prufer_sq.astype(int))
#         print(Prufer_sq)
#         print(len(Prufer_sq))
        eligible_nodes = list(set(All_node).difference(set(Prufer_sq)))
#             if len(eligible_nodes) == E-(N-1):
#                 break

        sorted(eligible_nodes)

        leaf_nodes = eligible_nodes.copy()


        temp_sq = list(Prufer_sq.copy())

        for i in range(len(Prufer_sq)):

            graph.add_edge(eligible_nodes[0],temp_sq[0])
            eligible_nodes.pop(0)

            if len(temp_sq)>=2:
                if temp_sq[0] not in temp_sq[1:]:
                    eligible_nodes.append(temp_sq[0])
                    sorted(eligible_nodes)
            elif len(temp_sq) == 1:
                eligible_nodes.append(temp_sq[0])

            temp_sq.pop(0)

    #     print(temp_sq)
    #     print(eligible_nodes)

        graph.add_edge(eligible_nodes[0],eligible_nodes[1])

    #     print(leaf_nodes)
        for i in range(len(leaf_nodes)-1):
            graph.add_edge(leaf_nodes[i],leaf_nodes[i+1])

        graph.add_edge(leaf_nodes[0],leaf_nodes[-1])

    #     print(len(graph.edges))

        nx.set_node_attributes(graph, dict(grid_graph.nodes.data()))
        top = nt.Topology.Topology()
        graph = top.assign_distances_grid(graph, pythagorus=False,
                                           harvesine=True)
#         print(graph.edges.data())
        return graph

    def objective(self, graph_vector):
        objective_value = 0
        # take vector
        # create nx.graph
        graph = self.build_graph_from_vector(graph_vector)
#         print(graph.edges)
#         print(graph.nodes)
#         if nx.is_connected(graph) and len(list(graph.edges)) == self.E and self.top.check_bridges(graph) and self.top.check_min_degree(graph):
        if len(list(graph.edges)) == self.E:
            objective_value = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], self.T_c, self.alpha)[0]
        else:
            objective_value+=1000

        # Penalty function for graphs that don't meet this objective
        # if not self.top.check_bridges(graph) or not self.top.check_min_degree(graph):
        #     objective_value += 1000
        # objective_value += graph.number_of_edges()
        return objective_value

    def run_ga(self):
        self.varbound = np.array([[1,self.N]]*int(self.N-2))
        self.model = ga(function=self.objective, dimension=self.N-2,variable_type='int',
                        variable_boundaries=self.varbound, algorithm_parameters=self.algorithm_param)
        self.model.run()
        best_solution = self.get_solution()
        objective_value = self.model.output_dict["function"]
        solution_report = self.model.report
#         print(solution_report)
        return {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}

    def get_solution(self):
        graph_vector = self.model.output_dict["variable"]
        self.solution_graph = self.build_graph_from_vector(graph_vector)
        return self.solution_graph
    
def distribute_func(func, N,E, T_C,alpha,grid_graph,topology_num,write=False,workers=10, gamma=None):
    '''
    Method of distributing the GA function on mutiple servers
    :param func: the running function
    :param N: number of nodes
    :param E: number of edges
    :param T_c: normalized traffic matrix
    :param alpha: weight list for kth shortest paths
    :param grid_graph: the graph used to decide node locations
    :param topology_num: the number of topologies need to be designed
    :param gamma_value: the gamma_value which indicates the skew of the traffic
    :param write: whether write the graph into the database
    :param workers: the number of workers in running
    :return dictionary: {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}
    '''
    # get indeces [0,1,2] for example for data 0-2
    indeces = nt.Tools.create_start_stop_list(topology_num, workers)
#     print(indeces)
    # Run all the ray instances
    results = ray.get([func.remote(N,E, T_C,alpha,grid_graph,indeces[i+1]-indeces[i],write, gamma) for i in range(workers)])
    return results
    
@ray.remote
def GA_run(N,E, T_C,alpha,grid_graph,topology_num,write,gamma):
    
    Solutions = []
    for i in range(topology_num):
        ptd = PTD(N,E, T_C,alpha,grid_graph)
        solution = ptd.run_ga()
        Solutions.append(solution)
        if write == True:
            graph = solution['graph']
            dwc = solution["objective_value"]
            topology_data = nt.Tools.graph_to_database_topology(graph)
            node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
            nt.Database.insert_graph(graph, "Topology_Data", "prufer-select-ga", node_data=True, use_pickle=True,
                                     type = "prufer-select-ga", T_c=T_C.tolist(), DWC = dwc,alpha=alpha, topology_data = topology_data,
                                     gamma=gamma)
    
    return Solutions

if __name__ == "__main__":
    ray.shutdown()
    port = 7111
    hostname = "128.40.41.14"
    ray.init(address='{}:{}'.format(hostname, port))
    # ray.init()
    graph_list =  nt.Database.read_topology_dataset_list("Topology_Data", "HTD-test", node_data=True,
                                                    find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
                                                    max_count=1, use_pickle=True)

    # dataset = nt.Database.read_data_into_pandas("Topology_Data", "HTD-test",
    #                                             find_dic={"FF-kSP Capacity": {'$exists': True}, 'nodes': 100},
    #                                             max_count=1)




    grid_graph = graph_list[0][0]
    N = len(grid_graph)
    E = 140
    alpha = [1]
    topology_num = 200
    gamma_t = [(0.4,5), (0.6,200), (0.8, 200), (1.0, 200)]

    #     if topology_num>0:
    T_C_list = []
    tasks = []
    for gamma, num in gamma_t:
        network = nt.Network.OpticalNetwork(grid_graph, channel_bandwidth=16e9)
        T_C = network.demand.create_skewed_demand(network.graph, gamma)
        # T_C_list.append(T_C, gamma)

        tasks += [GA_run.remote(N, E, T_C, alpha, grid_graph, 1, True, gamma) for _ in range(num)]
        # results = distribute_func(GA_run, N,E, T_C,alpha,grid_graph,topology_num, write = True, workers=300, gamma=gamma)
    print(len(tasks))
    results = ray.get(tasks)
#     ptd = PTD(N,E, T_C,alpha,grid_graph)
#     solution = ptd.run_ga()