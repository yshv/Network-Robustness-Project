import NetworkToolkit as nt
import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import mp_module as mp
import ray
from tqdm import tqdm
import datetime
from read_data import *


def resource_visualise(edge_list, Qlayer, rwa):
    edge_tuple = [tuple(x) for x in edge_list]
    ru = dict(zip(edge_tuple, tuple(np.zeros(len(edge_list)).tolist())))
    # print(ru)
    # print(edge_list)
    for q in range(Qlayer):
        for path in rwa[q]:
            for i in range(len(path) - 1):
                if tuple([path[i], path[i + 1]]) in ru.keys():
                    ru[tuple([path[i], path[i + 1]])] += 1 / Qlayer
                elif tuple([path[i + 1], path[i]]) in ru.keys():
                    ru[tuple([path[i + 1], path[i]])] += 1 / Qlayer

    ru_thresh = list(filter(lambda x: True if x > 0.75 else False, ru.values()))
    print("ru_thresh: {}".format(ru_thresh))


def MP_cost_function(graph, gamma=1, rwa=None, wavelength_cost=False):
    if rwa is not None:
        for wavelength in rwa:
            for path in wavelength:
                edges = nt.Tools.nodes_to_edges(path)
                for s, d in edges:
                    graph[s][d]["congestion"] += 1
        weights = np.ones(len(graph.edges()))
        for edge_ind, (s, d) in enumerate(graph.edges):
            weights[edge_ind] *= (graph[s][d]["weight"] + graph[s][d]["congestion"] ** gamma)
    return weights


def MP_weights_wavelengths(graph, Q=156, rwa=None, gamma=1):
    weights = np.ones((len(graph.edges), Q))
    wavelength_weights = np.ones(Q)
    if rwa is not None:

        for q_ind, q in enumerate(rwa):
            cost = 1
            for path in q:
                edges = nt.Tools.nodes_to_edges(path)
                cost += len(edges)
            cost = (1 / cost) ** gamma
            wavelength_weights[q_ind] = cost
        for e_ind, edge in enumerate(graph.edges):
            weights[e_ind] = wavelength_weights
    else:
        for ind, edge in enumerate(graph.edges):
            for q in range(Q):
                weights[ind, q] = q
    return weights


def init_congestion(graph):
    for s, d in graph.edges:
        graph[s][d]["congestion"] = 0
    return graph


@ray.remote
def dynamic_RWA(graph, demand, weights, Qlayer, ntrail=1, niter=500, method='MP', batch_type='size', batch_size=10,
                batch_time=1,
                epsilon=1e-3, load=None, max_amount=None, print_rwa=False,
                set_id=None, save_collection=None):
    if max_amount == None:
        max_amount = len(demand["id"])
    edge_list = list(graph.edges())
    BP = 0  # blocking probability
    RU = []  # resource utilization rate
    block = 0  # request block number
    blocked_paths = {}
    lightpath_dict = {'traffic_id': [], 'wavelength': [], 'path': []}  # active lightpaths

    if 0 in graph.nodes():
        graph = nx.relabel_nodes(graph, lambda x: x + 1)

    # N_t = len(demand["id"]) # Number of traffic
    N_t = max_amount
    print("N_t: {}".format(N_t))
    Nnode = len(graph.nodes())  # Number of node
    edges = [list(x) for x in graph.edges()]  # Edge list
    Nedge = len(edges)  # Number of Edges

    # print(edges)

    #     pairs # Connection requests
    #     npair = len(pairs) # Number of connection requests
    #     ntrial = 1 #Number of trying times for message passing
    #     niter = 1000 # Nmuber of iterations in each try
    # aPath = [[[1, 8], [1, 2, 6], [1, 3, 4]],[[3,10,7],[5, 2, 1]]] # Blocked Resource (Previous lightpaths)
    aPath = [[] for k in range(Qlayer)]  # Blocked Resource (Previous lightpaths)

    Total_wavelength = Qlayer * Nedge
    Used_wavelength = 0

    # Batch the demand by time
    demand['batch'] = np.zeros(N_t)  # Batch index
    demand["wavelength_num"] = np.ones((N_t))
    unassigned = demand['wavelength_num']

    batch_index = 0
    Batch = [[]]

    if batch_type == 'time':
        real_time = batch_time
        for index in range(N_t):
            if demand['time'][index] < real_time:
                demand['batch'][index] = batch_index
                Batch[batch_index].append(index)
            else:
                real_time = np.ceil(demand['time'][index] / batch_time)
                batch_index += 1
                Batch.append([index])

                demand['batch'][index] = batch_index

    if batch_type == 'size':
        real_index = batch_size
        for index in range(N_t):
            if index < real_index:
                Batch[batch_index].append(index)
            else:
                real_index += batch_size
                batch_index += 1
                Batch.append([index])

            #     print(Batch)
    #     print(len(Batch))
    in_batch = 0
    rwas = {str(i): {} for i in range(len(Batch))}
    BC = {str(i): {} for i in range(len(Batch))}
    pairs_save = {str(i): {} for i in range(len(Batch))}
    additional_id_info_dic = {str(i): {} for i in range(len(Batch))}
    flag=0
    first_time=False
    for _index_, batch in enumerate(tqdm(Batch, desc="{} load: {} Erlang".format(method, load))):
        # if 194 in demand["id"][batch[0]:batch[-1]]:
        #     print("demand ids: {}".format(demand["id"][batch[0]:batch[-1] + 1]))
        #     print("s: {}".format(demand['sn'][batch[0]:batch[-1] + 1]))
        #     print("d: {}".format(demand['dn'][batch[0]:batch[-1] + 1]))
        #     print("establish: {}".format(demand['establish'][batch[0]:batch[-1] + 1]))

        #         print('{} % finished'.format(i/N_t*100))
        pairs = []

        est_num = 0  # number of request need to be established in the batch
        teardown_num=0
        in_batch_tear = 0
        est_id = []
        #         print('batch=',batch)

        # Deal with each batch
        batch_id_list = [demand['id'][j] for j in batch]
        batch_id_list_establish = [demand['id'][j] for j in batch if demand['establish'][j]==1]
        for i in batch:

            if demand['establish'][i] == 1 and demand['id'][i] + int(N_t / 2) in batch_id_list:
                in_batch += 1
                in_batch_tear += 1
            if demand['establish'][i] == 1 and demand['id'][i] + int(N_t / 2) not in batch_id_list:
                if (demand["sn"][i], demand["dn"][i]) == (90, 45):
                    flag += 1
                est_num += 1
                est_id.append(demand['id'][i])
                for j in range(np.int(demand["wavelength_num"][i])):
                    pairs.append([demand['sn'][i], demand['dn'][i]])
                # if 194 in demand["id"][batch[0]:batch[-1]]:
                #     print("pairs: {}".format(pairs))
                # print(batch_id_list)
                # print(batch)

            elif demand["establish"][i] == 0:
                teardown_num+=1
                lightpath_id = np.int(demand['id'][i] - N_t / 2)
                # if lightpath_id == 1276:
                #     print("timestep: {}".format(i))
                # elif flag ==1 and not first_time:
                #     first_time=True
                #
                # elif flag==1 and first_time:
                #     count=0
                #     for wavelength in aPath:
                #         for path in wavelength:
                #             if (path[0], path[-1]) == (90, 45):
                #                 count+=1
                #     print("count: {}".format(count))
                #     print("timestep: {}".format(i))
                #     if count !=1:
                #         print(previous_rwa[98])
                #         print(rwa_dict[98])
                #         print(aPath[98])
                #         print(old_aPath[98])
                #         exit()

                pos = []
                for k in range(len(lightpath_dict['traffic_id'])):
                    if lightpath_dict['traffic_id'][k] == lightpath_id:
                        # if lightpath_dict['traffic_id'][k]==1276:
                        #     print("timestep: {}".format(i))
                        #     print(aPath[98])
                        pos.append(k)
                old_aPath = copy.deepcopy(aPath)
                if len(pos) > 0:
                    for j in range(np.int(demand["wavelength_num"][i])):
                        #                         print('lightpath_id:',lightpath_id)
                        #                         print('lightpath needs to be teardown:',lightpath_dict['path'][pos[j]])
                        #                         print('lightpath active:',aPath[lightpath_dict['wavelength'][pos[j]]])
                        #                         print('number:',np.int(demand["wavelength_num"][i]))

                        aPath[lightpath_dict['wavelength'][pos[j]]].remove(lightpath_dict['path'][pos[j]])
                        Used_wavelength -= len(lightpath_dict['path'][pos[j]]) - 1

                elif lightpath_id not in batch_id_list:
                    pass
                    # block += 1
                    # print("rwa: {}".format(rwa))
                    # print("s: {}".format(demand['sn'][batch]))
                    # print("d: {}".format(demand['dn'][k]))
                    # print("unassigned k: {}".format(unassigned[k]))
                    # print("path-s, path-d : {},{}".format(path[0], path[-1]))
                    # exit()
                    # print("established k: {}".format(demand['establish'][k]))
                    # print("lightpath dict: {}".format(lightpath_dict))
                    # print("batch list id: {}".format(batch_id_list))
                    # print("demand est: {}".format(demand["establish"][batch[0]:batch[-1]]))
                    # print("demand id: {}".format(demand["id"][batch[0]:batch[-1]]))
                    # print('unestablished_id:',lightpath_id)
                    # print('lightpath_id: {}'.format(lightpath_id))
                    # print('est_id: {}'.format(est_id))
                    # exit()

        if est_num > 0:
            time_start = time.perf_counter()

            # print('pairs:',pairs)
            # print('aPath:',aPath)

            if method == 'MP':
                resource_visualise(edge_list, Qlayer, aPath)
                # print("aPath: {}".format(aPath))

                # print("weights: {}".format(weights))
                rwa = mp.mp(Nnode, edges, weights, pairs, Qlayer, 2, ntrail, niter, aPath,
                            epsilon)  # Message passing function
                # weights = MP_cost_function(graph, gamma=0.5, rwa=rwa)
                weights = MP_weights_wavelengths(graph, Q=Qlayer, rwa=aPath, gamma=1.5).astype(int)
                # print("rwa: {}".format(rwa))

            elif method != 'MP':
                network = nt.Network.OpticalNetwork(graph, routing_func=method)
                network.routing_channels = Qlayer
                network.rwa.channels = Qlayer
                previous_rwa = nt.Tools.rwa_list_to_dict(copy.deepcopy(aPath))
                conn_matrix = nt.Tools.pairs_list_to_mat(network.graph, pairs)
                # print("aPath old: {}".format(aPath))
                # print("previous RWA: {}".format(previous_rwa))

                # print(conn_matrix)
                # print('routing channels: {}'.format(network.routing_channels))

                rwa, blocked_connections, additional_id_info = network.route(conn_matrix, rwa_assignment_previous=previous_rwa, k=10, e=0,
                                                         return_blocked=True, connection_pairs=pairs, _ids=est_id, save_ids=True)
                # print(additional_id_info)
                rwa_dict=rwa
                rwas[str(_index_)] = nt.Tools.write_database_dict(rwa)

                BC[str(_index_)] = {"blocked": int(blocked_connections), "established":int(len(pairs)-blocked_connections),
                               "total":int(len(pairs)), "teardown number":int(teardown_num)-int(in_batch_tear), "establish number":int(est_num)}
                pairs_save[str(_index_)] = pairs
                additional_id_info_dic[str(_index_)] = additional_id_info
                # print(BC)
                # print(blocked_connections)
                if print_rwa:
                    print("rwa {}, {}: {}".format(_index_, pairs, rwa))

                # print(rwa)
                if rwa == True:
                    # print("blocked!!!!")
                    # print("aPath: {}".format(aPath))
                    # print("rwa: {}".format(rwa))
                    time_cost = time.perf_counter() - time_start
                    # print("time cost: {}".format(time_cost))
                    blocked_paths[str(_index_)] = pairs
                    # print(blocked_paths)
                    # exit()
                    rwa = []
                else:
                    rwa_new = []
                    rwa = nt.Tools.rwa_dict_to_list(rwa)
                    # print("apath_new: {}".format(aPath))
                    for ind_i, channel in enumerate(rwa):
                        paths = []
                        for ind_j, path in enumerate(channel):
                            # print("path: {}".format(path))
                            # print("path a: {}".format(aPath[ind_i]))

                            if path not in aPath[ind_i]:
                                paths.append(path)
                        rwa_new.append(paths)
                    # print("rwa_new: {}".format(rwa_new))
                    rwa = rwa_new

                    block += blocked_connections

                    # aPath = rwa
                # print(rwa)
                # print('routing channels: {}'.format(network.channels))
                # print(rwa)

                # print(rwa)
            # fake rwa solution for test
            # rwa = [[] for k in range(Qlayer)]
            # for q in range(len(pairs)):
            #     rwa[q].append(pairs[q])

            #                 print('rwa:',rwa)
            time_cost = time.perf_counter() - time_start
            # print("time cost: {}".format(time_cost))
            # print('pairs:',pairs)
            # print('rwa:',rwa)
            # print('time_cost:',time_cost)
            # time_cost <= min(demand["timeout"][batch[0]:batch[-1] + 1]) and

            old_aPath=aPath
            if len(rwa) > 0:
                print("connections blocked: {}".format(block / (N_t/2)))
                for q in range(Qlayer):
                    if len(rwa[q]) > 0:
                        # if method == "MP":
                        aPath[q] += rwa[q]

                        # else:
                        #     aPath = rwa
                        # for path in rwa[q]:
                        for k in batch:
                            if ([path[0], path[-1]] == [demand['sn'][k], demand['dn'][k]] or [path[-1],
                                                                                              path[0]] == [
                                    demand['sn'][k], demand['dn'][k]]) and demand['establish'][k] == 1 and \
                                    unassigned[k] > 0 and demand['id'][k] + int(N_t / 2) not in batch_id_list:
                                unassigned[k] -= 1
                        for item in additional_id_info:
                        # for k in batch:

                            # if 194 in demand['id'][batch[0]:batch[-1] + 1]:
                            #     print("demand id: {}".format(demand['id'][k]))
                            #     print("rwa: {}".format(rwa))
                            #     print("s: {}".format(demand['sn'][k]))
                            #     print("d: {}".format(demand['dn'][k]))
                            #     print("unassigned k: {}".format(unassigned[k]))
                            #     print("path-s, path-d : {},{}".format(path[0], path[-1]))
                            #     # exit()
                            #     print("established k: {}".format(demand['establish'][k]))

                            #     if demand['id'][k] ==
                            lightpath_dict['traffic_id'].append(item["id"])
                            lightpath_dict['wavelength'].append(item["wavelength"])
                            lightpath_dict['path'].append(item["path"])

                            Used_wavelength += len(item["path"]) - 1



                # print('lightpath:', lightpath_dict)
                # #
                # print("demand s: {}".format(demand["sn"][batch[0]:batch[-1]]))
                # print("demand d: {}".format(demand["dn"][batch[0]:batch[-1]]))
                #
                # exit()


            # else:
            #     block += est_num

        #         print('aPath:',aPath)
        RU.append(Used_wavelength / Total_wavelength)

    BP = block / ((N_t/2)-in_batch)
    # print(rwa)
    for key_i in rwas.keys():
        for key_j in rwas[key_i].keys():
            for ind, path in enumerate(rwas[key_i][key_j]):
                new_path = []
                for item in path:
                    new_path.append(int(item))
                rwas[key_i][key_j][ind] = new_path
                assert type(path) == list
    # print(rwas)
    # print(BC)
    # print(type(blocked_paths))
    nt.Database.insert_graph(graph, "Topology_Data", save_collection, node_data=True,
                             use_pickle=True, **{"blocking probability": float(BP),
                                                "resource utilisation": RU,
                                                "set_id": int(set_id),
                                                "load": int(load),
                                                "blocked paths": blocked_paths,
                                                "routing method": str(method),
                                                "batch time": float(batch_time),
                                                "batch size": int(batch_size),
                                                "batch type": str(batch_type),
                                                "Q layer": int(Qlayer),
                                                # "heuristic sorting":"sequential",
                                                "heuristic sorting":"batch pairs",
                                                "notes":"changed kSP-FF path choice",
                                                "demand path": "/home/uceeatz/Code/test/networktoolbox/scripts/dynamic-simulation/data",
                                                "rwas":rwas,
                                                "info":BC,
                                                "pairs":pairs_save,
                                                "additional id info":additional_id_info_dic
                                                })

    return BP, RU, load, blocked_paths


# graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPEDP", find_dic={"computational scaling data":1,"nodes":20},use_pickle=True)
# # print(len(graph_list))
# graph = graph_list[0][0]
# if 0 in graph.nodes():
#     graph = nx.relabel_nodes(graph, lambda x: x+1)
# n_t = 1000
# min_bandwidth = 50
# max_bandwidth = 100
# _mu = 10
# timeout_limit = 60
# bandwidth_per_wavelength = 100
# demand = nt.Demand.Demand(graph)
# weights = np.ones(len(graph.edges())) # Edge weights
# Qlayer = 10 # Number of wavelengths
# avg_RU = []
# BP = []
# load = [0.1,0.2,0.3,0.4,0.5]
# # load = [0.1]
# avgdegree = len(graph.edges())/len(graph.nodes())*2
# avgsize = np.ceil((min_bandwidth+max_bandwidth)/bandwidth_per_wavelength)/2
# Lambda = [1/(x*Qlayer*len(graph.edges())/avgsize*(1/_mu)) for x in load]


# for _lambda in Lambda:
#     demand_data = demand.create_poisson_process_demand(n_t, min_bandwidth, max_bandwidth, _lambda, _mu, timeout_limit, bandwidth_per_wavelength,
#                                           demand_distribution=np.asarray([]),batch_time = 1)

#     print('Demand generated!')
#     print(_lambda)
#     print(Qlayer)
#     bp,ru = dynamic_RWA(graph,demand_data,weights,Qlayer,niter =500,method = 'MP',batch_type = 'size',batch_size = 10)
#     print('Blocking probability:',bp)
#     print('Resource utilization rate',ru)
#     avg_RU.append(np.mean(ru))
#     BP.append(bp)

# print('Blocking probability:',BP)
# print('Resource utilization rate',avg_RU)
def dynamic_sim_dist(graph, demand_data, methods, db="Topology_Data", collection=None, niter=100, batch_type='size',
                     batch_size=10, hostname="128.40.43.93", port=6380, local=False, epsilon=1e-3,
                     write_local=False, write_path=None, results_name=None, demand_dic=None, Q=156,
                     save_collection=None):
    ray.shutdown()
    if local:
        ray.init()
    else:
        ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000', ignore_reinit_error=True)
    tasks = []
    for method in methods:
        for ind_i, set_id in enumerate(demand_data.keys()):
            for ind_j, load in enumerate(demand_data[set_id].keys()):
                # print(demand_data[set_id].keys())
                # print("set_ID: {} ".format(set_id))
                # print("load: {}".format(load))
                demand = demand_data[set_id][load]
                weights = MP_weights_wavelengths(graph, Q=Q, rwa=[], gamma=1.5)
                tasks.append(dynamic_RWA.remote(graph, demand, weights, Q, niter=niter, method=method,
                                                batch_type=batch_type, batch_size=batch_size, load=load,
                                                epsilon=epsilon, max_amount=None, print_rwa=False,
                                                set_id=set_id,
                                                save_collection=save_collection))

    # for method in methods:
    #     for ind, (graph, _id) in enumerate(graph_list):
    #         for ind_, key in enumerate(dataset['demand_dict'][ind]):
    #             # print(type(dataset['demand_dict'][ind]['4']["demand_data"]))
    #             demand_data = dataset['demand_dict'][ind][key]['demand_data']
    #             load = dataset['demand_dict'][ind][key]['load']
    #             Qlayer = dataset['demand_dict'][ind][key]['Qlayer']
    #             # Qlayer=20
    #             # Qlayer = 30
    #             # graph = graph_list[0][0]
    #             # _id = graph_list[0][1]
    #             print("load: {}".format(load))
    #             graph = init_congestion(graph)
    #             # weights = MP_cost_function(graph, gamma=2, rwa=[])
    #             weights = MP_weights_wavelengths(graph, Q=Qlayer, rwa=[], gamma=1.5)
    #             tasks.append(dynamic_RWA.remote(graph, demand_data, weights, Qlayer, niter=niter, method=method,
    #                                             batch_type=batch_type, batch_size=batch_size, load=load,
    #                                             epsilon=epsilon, max_amount=None, print_rwa=False))
    # if ind_ == 0:
    #     break

    results = ray.get(tasks)
    print(results)
    results_iter = iter(results)
    if not write_local:
        for ind_m, method in enumerate(methods):
            for ind, (graph, _id) in enumerate(graph_list):
                result_dict = {}
                for ind_, key in enumerate(dataset['demand_dict'][ind]):
                    load = dataset['demand_dict'][ind][key]['load']
                    next_result = next(results_iter)
                    result_dict[str(ind_)] = {"blocking probability": str(next_result[0]),
                                              "resource utilisation": next_result[1],
                                              "load": str(next_result[2]),
                                              "blocked paths": next_result[3]}
                    # if ind_ == 0:
                    #     break
                print(result_dict)
                nt.Database.update_data_with_id(db, collection, _id,
                                                {"$set": {"{} nonlinear results batch {}".format(str(method),
                                                                                                 batch_size): result_dict,
                                                          "timestamp {} nonlinear results batch {}".format(
                                                              str(method), batch_size): datetime.datetime.utcnow()}})
    else:
        pass


def assemble_data(graph_query, graph_collection, loads, repeats):
    print(loads)
    demand_data = {set_id: {load: read_demand(load, set_id) for load in loads} for set_id in repeats}
    # for set_id in repeats:
    #     for load in loads:
    #         demand_data[set_id][load] = read_demand(load, set_id)
    # demand_data = {set_id: {load: read_demand(load, set_id)} for set_id in repeats for load in loads}
    graph = \
        nt.Database.read_topology_dataset_list("Topology_Data", graph_collection, find_dic=graph_query, max_count=1)[0][
            0]
    # print(demand_data[1].keys())
    return graph, demand_data


# graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "MPEDP-test-SBAG", find_dic={"nodes": 30},
#                                                     use_pickle=True, max_count=1)
# dataset = nt.Database.read_data_into_pandas("Topology_Data", "MPEDP-test-SBAG", find_dic={"nodes": 30}, max_count=1)
# print(dataset["demand_dict"][0])
# demand_dict = dataset['demand_dict'][0]
# exit()
graph, demand_data = assemble_data({"nodes": 100, 'edges': 130}, "HTD-test", list(range(1,11)), list(range(0,10)))
# print(graph)
# print(demand_data[1].keys())
dynamic_sim_dist(graph, demand_data, ["FF-kSP"], collection="HTD-test", niter=400, batch_size=200,
                 epsilon=1.5 * 1e-3, local=False, write_local=True, write_path="results", results_name="mp-edp",
                 hostname="128.40.41.48",
                 port=7112, Q=156, save_collection="MPEDP-heuristics-results")
# niter = 200
# reinforcement factor = 5*10^-4
# for method in ["MP", "FF-kSP","kSP-FF"]:
#     print(method)
#     for ind, (graph,_id) in enumerate(graph_list):
#         result_dict = {}
#         for ind_, key in enumerate(dataset['demand_dict'][ind]):
#             # print(demand_dict[key])
#         # # print(demand_dict)
#             demand_data = dataset['demand_dict'][ind][key]['demand_data']
#             load = dataset['demand_dict'][ind][key]['load']
#             Qlayer = dataset['demand_dict'][ind][key]['Qlayer']
#             # Qlayer = 20
#             # graph = graph_list[0][0]
#             # _id = graph_list[0][1]
#             weights = np.ones(len(graph.edges()))
#
#             # print(load)
#             # print(demand_data)
#
#             bp,ru = dynamic_RWA(graph,demand_data,weights,Qlayer,niter=2500,method = method,batch_type = 'size',batch_size = 10)
#             result_dict[str(ind_)]= {"blocking probability" : str(bp),
#                                              "resource utilisation": ru,
#                                              "load": str(load)}
#
#             print('Blocking probability:',bp)
#             # print('Resource utilization rate',ru)
#             # print(result_dict)
#         nt.Database.update_data_with_id("Topology_Data", "MPEDP-test", _id, {"$set": {"{} results".format(method): result_dict}})
