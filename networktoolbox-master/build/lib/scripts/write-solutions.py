import os
import NetworkToolkit as nt
from bson import ObjectId
from tqdm import tqdm
import numpy as np
import networkx as nx
import ast

data={}
for file_name in tqdm(os.listdir("/scratch/datasets/gurobi/solutions")):
    try:
        _id = file_name[:-6].replace("_","")
        # print(_id)
        _id = ObjectId(str(_id))
        number = int(file_name[-5])
        file = open("/scratch/datasets/gurobi/solutions/{}".format(file_name), "r")



        # print(number)
        # exit()
        graph,_id, data_written= nt.Database.read_topology_dataset_list("Topology_Data",
                                               "ta", "data written",
                                               find_dic={"_id":_id}, use_pickle=True)[0]
        # print(graph.nodes)
        if data_written:
            continue
        graph= nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

        k_SP=nt.Routing.Tools.get_k_shortest_paths_MNH(graph, e=10, k=10)
        network=nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        W=network.channels
        K=len(k_SP)

        delta = [[[0 for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]
        contents=file.readlines()
        for ind, line in enumerate(contents):
            if ind == 1:
                # print(line)
                objective_string = line.split()
                objective_value = ast.literal_eval(objective_string[-1])
                # objective_value = float(line[-1])
            if line[:3] =="rwa":
                rest, solution=line.split()
                # print(rest.split('-'))
                rest, w,wval, k,kval, z,zval = rest.split('-')
                delta[int(zval)][int(kval)][int(wval)] = int(solution)
        if objective_value == 0:
            continue
        ilp=nt.Routing.ILP.ILP(graph, network.channels, network.channel_bandwidth)
        rwa = ilp.convert_delta_to_rwa_assignment(delta, k_SP, W,numeric_delta=True)

        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(rwa)
        network.physical_layer.add_non_linear_NSR_to_links()
        max_capacity = network.physical_layer.get_lightpath_capacities_PLI(rwa)
        node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
        rwa = nt.Tools.write_database_dict(rwa)
        # print(_id)
        # type(_id)
        # print(type(rwa))
        # print(rwa)
        cast_int = lambda number: int(number)
        for wavelength in rwa.keys():
            new_paths=[]
            for path in rwa[wavelength]:
                new_paths.append(list(map(lambda node: int(node), path)))
            rwa[wavelength] = new_paths
        # print(rwa)
                # for node in path:
                #     if type(node) is np.int64:
                #         print(type(node))
                #         print(node)
                #         index = rwa[wavelength].index(path)
                #     assert type(node) is not np.int64

        data[str(_id)]={number:{"rwa":rwa, "capacity":max_capacity[0], "node pair capacities":node_pair_capacities}}
    except Exception as err:
        print(err)
        continue
    # exit()
for _id in data.keys():
    rwa = data[_id][max(data[_id].keys())]["rwa"]
    capacity = data[_id][max(data[_id].keys())]["capacity"]
    node_pair_capacities = data[_id][max(data[_id].keys())]["node pair capacities"]
    # print(rwa)
    # print(max(data[_id].keys()))
    nt.Database.update_data_with_id("Topology_Data", "ta", ObjectId(_id), newvals={"$set": {"ILP-throughput RWA": rwa,
                                                                                  "ILP-throughput capacity":capacity,
                                                                                  "ILP-throughput node pair capacity":node_pair_capacities,
                                                                                  "partial solution":1}})