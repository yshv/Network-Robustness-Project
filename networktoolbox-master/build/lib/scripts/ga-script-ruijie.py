import NetworkToolkit as nt
import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn
import time
from operator import itemgetter
# from geneticalgorithm import geneticalgorithm as ga
import geneticalgorithm as ga
import ray
from tqdm import tqdm


class PTD():
    def __init__(self, N, E, T_c, alpha=[1]):
        self.N = N
        self.E = E
        self.T_c = [T_c]
        self.alpha = alpha
        self.top = nt.Topology.Topology()
        self.solution_graph = None
        self.algorithm_param = {'max_num_iteration': 1000,
                                'population_size': 500,
                                'mutation_probability': 0.1,
                                'elit_ratio': 0.01,
                                'crossover_probability': 0.8,
                                'parents_portion': 0.3,
                                'crossover_type': 'uniform',
                                'max_iteration_without_improv': None}

    def build_graph_from_vector(self, graph_vector):
        edges_poss = [(i + 1, j + 1) for i in range(self.N) for j in range(self.N) if j > i]
        edges = []
        for ind, item in enumerate(graph_vector):
            if item == True:
                edges.append(edges_poss[ind])
        graph = nx.Graph()
        graph.add_nodes_from(list(range(1, self.N + 1)))
        graph.add_edges_from(edges)
        return graph

    def objective(self, graph_vector):
        objective_value = 0
        # take vector
        # create nx.graph
        graph = self.build_graph_from_vector(graph_vector)
        # print(graph.edges)
        # print(graph.nodes)
        if nx.is_connected(graph) and len(list(graph.edges)) == self.E and self.top.check_bridges(
                graph) and self.top.check_min_degree(graph):
            objective_value = nt.Tools.get_demand_weighted_cost([[graph, 1]], self.T_c, self.alpha)[0]
        else:
            objective_value += 1000

        # Penalty function for graphs that don't meet this objective
        # if not self.top.check_bridges(graph) or not self.top.check_min_degree(graph):
        #     objective_value += 1000
        # objective_value += graph.number_of_edges()
        return objective_value

    def run_ga(self):
        self.varbound = np.array([[0, 1]] * int((self.N ** 2 - self.N) / 2))
        self.model = ga(function=self.objective, dimension=int((self.N ** 2 - self.N) / 2), variable_type='bool',
                        variable_boundaries=self.varbound, algorithm_parameters=self.algorithm_param)
        self.model.run()
        best_solution = self.get_solution()
        objective_value = self.model.output_dict["function"]
        solution_report = self.model.report
        #         print(solution_report)
        return {"graph": best_solution, "objective_value": objective_value, 'solution_report': solution_report}

    def get_solution(self):
        graph_vector = self.model.output_dict["variable"]
        self.solution_graph = self.build_graph_from_vector(graph_vector)
        return self.solution_graph


def distribute_func(func, N, E, T_C, alpha, topology_num, gamma_value, workers=1):
    # get indeces [0,1,2] for example for data 0-2
    indeces = nt.Tools.create_start_stop_list(topology_num, workers)
    print(indeces)
    # Run all the ray instances
    results = ray.get([func.remote(N, E, T_C, alpha, indeces[i + 1] - indeces[i], gamma_value) for i in tqdm(range(workers))])
    return results


@ray.remote
def EA_run(N, E, T_C, alpha, topology_num, gamma_value):
    Solutions = []
    for i in range(topology_num):
        ptd = PTD(N, E, T_C, alpha)
        solution = ptd.run_ga()
        Solutions.append(solution)
        graph = solution['graph']
        dwc = solution["objective_value"]
        nt.Database.insert_graph(graph, "Topology_Data", "ta", node_data=True, use_pickle=True, type="ga", T_c=T_C,
                                 gamma=gamma_value, DWC=dwc, purpose="ga-analysis", alpha=alpha, data="new alpha")

    return solution


if __name__ == "__main__":
    ray.shutdown()
    hostname = "128.40.43.93"
    port = 6380
    # ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)
    ray.init()
    for gamma_value in np.around(np.arange(0, 1.1, 0.2), decimals=1):
        #         graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "ta",
        #                                                             find_dic={"purpose":"scaling",'distance_scale':scale,"gamma":0,'flag':'feed tb','type':'ga'},use_pickle=True)
        dataset = nt.Database.read_data_into_pandas("Topology_Data", "ta", find_dic={"purpose": "ga-analysis"},
                                                    max_count=1)
        T_C = dataset["T_c"][0]
        #         T_B = dataset["T_b"][0]

        N = dataset["nodes"][0]
        E = dataset["edges"][0]
        alpha = [0.66576701, 0.15450554, 0.05896479, 0.03678006, 0.02237619,
                 0.01477571, 0.01055498, 0.0077231, 0.00602809, 0.00448774,
                 0.00366258, 0.00290902, 0.00248457, 0.00198062, 0.00163766,
                 0.00139676, 0.0012504, 0.00101622, 0.00085918, 0.00083979]
        #         topology_num = 200-len(graph_list)
        #         print(topology_num)
        topology_num = 200
        if topology_num > 0:
            results = distribute_func(EA_run, N, E, T_C, alpha, topology_num, gamma_value, workers=100)