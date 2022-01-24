import NetworkToolkit as nt
import networkx as nx
import numpy as np
# import copy
# import matplotlib.pyplot as plt
# import sklearn.linear_model as lm
# import sklearn
# import time
from operator import itemgetter
from geneticalgorithm import geneticalgorithm as ga
import ray
# import pandas as pd
# import dask
# from scipy.stats import spearmanr as spearman
# from scipy.stats import kendalltau
# import random
from tqdm import tqdm

class Topology_vector_PTD():
    '''
    Class of GA topology design
    :param N: number of nodes
    :param E: number of edges
    :param T_c: normalized traffic matrix
    :param alpha: weight list for kth shortest paths
    :param grid_graph: the graph used to decide node locations
    :return dictionary: {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}
    '''
    
    
    def __init__(self, N, E, T_c, alpha,grid_graph,length_limit):
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
        self.length_limit = length_limit

    def build_graph_from_vector(self, graph_vector):
        edges_poss = [(i + 1, j + 1) for i in range(self.N) for j in range(self.N) if j > i]
        edges = []
        for ind, item in enumerate(graph_vector):
            if item == True:
                edges.append(edges_poss[ind])
        graph = nx.Graph()
        graph.add_nodes_from(list(range(1,self.N+1)))
        graph.add_edges_from(edges)
        
        nx.set_node_attributes(graph, dict(self.grid_graph.nodes.data()))
        top = nt.Topology.Topology()
        graph = top.assign_distances_grid(graph, pythagorus=False, harvesine=True)
        
#         print(self.grid_graph.nodes.data())
        
        return graph

    def objective(self, graph_vector):
        objective_value = 0
        # take vector
        # create nx.graph
        graph = self.build_graph_from_vector(graph_vector)
        total_length = np.array([graph[s][d]["weight"]*80 for s,d in graph.edges]).sum()
        # print(graph.edges)
        # print(graph.nodes)
        if total_length <= self.length_limit and self.top.check_bridges(graph):
            N = len(graph.nodes())
            E = len(graph.edges())
            alpha = E/N/(N-1)*2
            DWC_structure = nt.Tools.get_demand_weighted_cost([[graph,0]], self.T_c, self.alpha,penalty_num=1000)[0]
            DWC_distance = nt.Tools.get_demand_weighted_cost_distance([[graph, 1]], self.T_c, self.alpha)[0]
            objective_value = alpha*DWC_distance + (1-alpha)* DWC_structure
        else:
            objective_value+=1000

        # Penalty function for graphs that don't meet this objective
        # if not self.top.check_bridges(graph) or not self.top.check_min_degree(graph):
        #     objective_value += 1000
        # objective_value += graph.number_of_edges()
        return objective_value
    
    def build_graph_from_prufer(self, Prufer_sq):
    #         edges_poss = [(i + 1, j + 1) for i in range(self.N) for j in range(self.N) if j > i]
    #         edges = []
    #         for ind, item in enumerate(graph_vector):
    #             if item == True:
    #                 edges.append(edges_poss[ind])
            graph = nx.Graph()
            graph.add_nodes_from(list(np.arange(1,self.N+1)))
            All_node = np.arange(1,self.N+1)

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

            nx.set_node_attributes(graph, dict(self.grid_graph.nodes.data()))
            top = nt.Topology.Topology()
            graph = top.assign_distances_grid(graph, pythagorus=False, harvesine=True)
    #         print(graph.edges.data())
            return graph
    
    def pop_initial(self):
        initial_pop = []
#         All_node = np.arange(1, self.N + 1)
        for i in range(self.algorithm_param['population_size']):
            Prufer_sq = np.random.randint(1,self.N+1,size = self.N-2)
            initial_graph = self.build_graph_from_prufer(Prufer_sq)
            adjacent_matrix = nx.convert_matrix.to_numpy_array(initial_graph,dtype= np.int, weight = None)
#             print(adjacent_matrix)
            graph_vector = [adjacent_matrix[i][j] for i in range(self.N) for j in range(self.N) if j > i]
#             print(graph_vector)
#             exit()
                
            
            initial_pop.append(graph_vector)
#             print('{} graph generated'.format(i+1))
        return initial_pop

    

    def run_ga(self):
        self.varbound = np.array([[0,1]]*int((self.N**2-self.N)/2))
        self.model = ga(function=self.objective, dimension=int((self.N**2-self.N)/2),variable_type='bool',
                        variable_boundaries=self.varbound, algorithm_parameters=self.algorithm_param,convergence_curve=False)
        pop = self.pop_initial()
        
        self.model.run(pop)
        best_solution = self.get_solution()
        objective_value = self.model.output_dict["function"]
        solution_report = self.model.report
#         print(solution_report)
        return {"graph":best_solution, "objective_value":objective_value,'solution_report':solution_report}

    def get_solution(self):
        graph_vector = self.model.output_dict["variable"]
        self.solution_graph = self.build_graph_from_vector(graph_vector)
        return self.solution_graph
    
    
def distribute_func(func, N,E, T_C,alpha,grid_graph,topology_num,length_limit,gamma,write=False,workers=50):
    # get indeces [0,1,2] for example for data 0-2
    indeces = nt.Tools.create_start_stop_list(topology_num, workers)
    print(indeces)
    # Run all the ray instances
    results = ray.get([func.remote(N,E, T_C,alpha,grid_graph,indeces[i+1]-indeces[i],length_limit,gamma,write) for i in range(workers)])
    return results




@ray.remote
def GA_length_run(N,E, T_C,alpha,grid_graph,length_limit,gamma,write=False, collection="vector-ga-distance"):
    
    Solutions = []
    # for i in range(topology_num):
    ptd = Topology_vector_PTD(N,E, T_c,alpha,grid_graph,length_limit)
    solution = ptd.run_ga()
    Solutions.append(solution)
    if write == True:
        graph = solution['graph']
        dwc = solution["objective_value"]
        topology_data = nt.Tools.graph_to_database_topology(graph)
        node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
        nt.Database.insert_graph(graph, "Topology_Data", collection, node_data=True, use_pickle=True,
                                 type = "vector-ga",T_c = T_C,gamma = gamma,DWC = dwc, alpha=alpha, topology_data = topology_data,length_limit = length_limit)
    
    return solution


if __name__ == "__main__":
    hostname = "128.40.41.48"
    port = 7112
    ray.init(address="{}:{}".format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)

    gamma_value = np.around(np.arange(0, 1.1, 0.2),decimals=1)


    results = []
    tasks=[]
    dataset = nt.Database.read_data_into_pandas("Topology_Data", "ta",
                                                find_dic={'purpose': 'ga-analysis',
                                                          'ILP-connections': {'$exists': True},
                                                          "gamma": gamma_value[0], 'type': 'ga'}, max_count=1)
    grid_graph = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", node_data=True,
                                                        find_dic={'name': 'NSFNET'}, max_count=1, use_pickle=True)[0][0]
    for gamma in gamma_value:


        T_c=nt.Demand.Demand.create_locally_skewed_demand(grid_graph, gamma=gamma)

        N = len(grid_graph)
        E = grid_graph.number_of_edges()
        topology_num = 200
        avg_length = 2000
        # grid_graph = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", node_data=True,
        #                                                 find_dic={'name': 'NSFNET'},max_count=1,use_pickle=True)[0][0]
        alpha = [1]
        length_limit = np.int(avg_length*E)

        tasks+= [GA_length_run.remote(N, E, T_c,alpha,grid_graph,length_limit,gamma,write=True,
                                      collection="vector-ga-distance-revised") for i in tqdm(range(topology_num))]
    results = ray.get(tasks)