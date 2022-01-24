import ray
import ast
import NetworkToolkit as nt
from tqdm import tqdm
import numpy as np

@ray.remote
class ScaledThroughput:
    def calculate_T_UL_scaled(self,graph_data, scale=1):
        """

        :param graph_data:
        :param scale:
        :return:
        """
        graph = nt.Tools.read_database_topology(graph_data["topology data"])
        for s, d in graph.edges:
            graph[s][d]["weight"] = graph[s][d]["weight"] * scale
        new_rwa = {}
        # print(graph_data["ILP RWA assignment"])
        for key in graph_data["ILP RWA assignment"]:
            new_rwa[ast.literal_eval(key)] = graph_data["ILP RWA assignment"][key]

        network = nt.Network.OpticalNetwork(graph, channel_bandwidth=16e9)
        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(new_rwa)
        network.physical_layer.add_non_linear_NSR_to_links()
        max_capacity = network.physical_layer.get_lightpath_capacities_PLI(new_rwa)
        return max_capacity


    def calculate_scaled_throughput_data(self,dataframe, start=0.01, end=1, step=0.02, graph_type="SBAG",
                                         save_dic={}):
        """

        :param dataframe:
        :param start:
        :param end:
        :param step:
        :return:
        """

        for i in tqdm(np.arange(start, end, step)):
            for index, graph_data in dataframe.iterrows():
                throughput = self.calculate_T_UL_scaled(graph_data, scale=i)[0]
                # print(throughput)
                dict1 = {"graph type": graph_type, "scale": i, "throughput data": throughput}

                write_dic = {**dict1, **save_dic}
                nt.Database.insert_data("Topology_Data", "topology-paper", write_dic)
        # return scaled_data


if __name__ == "__main__":
    ray.init(address='auto', redis_password='5241590000000000')
    import argparse

    parser = argparse.ArgumentParser(description='Create or process data')
    parser.add_argument('--data', action='store', type=str)
    parser.add_argument('--start', action='store', type=int)
    parser.add_argument('--end', action='store', type=int)
    parser.add_argument('-fd', nargs="+", default=None)
    parser.add_argument('-sd', nargs="+", default=None)
    parser.add_argument('-rs', action='store', type=float, default=0.0)
    parser.add_argument('-re', action='store', type=float, default=1.0)
    parser.add_argument('--rstep', action='store', type=float, default=0.02)
    parser.add_argument('-w', action="store", type=int, default=1)
    parser.add_argument('-rn', action="store", type=int, default=0)
    args = parser.parse_args()
    data = vars(args)['data']
    find_dic_list = vars(args)['fd']
    data_ind_start = vars(args)['start']
    data_ind_end = vars(args)['end']
    save_dic_list = vars(args)["sd"]
    range_start = vars(args)["rs"]
    range_end = vars(args)["re"]
    range_step = vars(args)["rstep"]
    workers = vars(args)["w"]
    rename = vars(args)["rn"]

    print("find dict exists")
    find_dic = {}
    print(find_dic_list)
    if find_dic_list is not None:
        for ind, item in enumerate(find_dic_list):
            if ind * 2 == len(find_dic_list): break
            if find_dic_list[ind * 2 + 1][0:2] == "-s":
                find_dic[find_dic_list[ind * 2]] = str(find_dic_list[ind * 2 + 1][3:])
            elif find_dic_list[ind * 2 + 1][0:2] == "-i":
                find_dic[find_dic_list[ind * 2]] = int(find_dic_list[ind * 2 + 1][2:])
            elif find_dic_list[ind * 2 + 1][0:2] == "-f":
                find_dic[find_dic_list[ind * 2]] = float(find_dic_list[ind * 2 + 1][2:])
    else:
        find_dic={}

    save_dic = {}
    if save_dic_list:
        for ind, item in enumerate(save_dic_list):
            if ind * 2 == len(save_dic_list): break
            if save_dic_list[ind * 2 + 1][0:2] == "-s":
                save_dic[save_dic_list[ind * 2]] = str(save_dic_list[ind * 2 + 1][3:])
            elif save_dic_list[ind * 2 + 1][0:2] == "-i":
                save_dic[find_dic_list[ind * 2]] = int(save_dic_list[ind * 2 + 1][2:])
            elif save_dic_list[ind * 2 + 1][0:2] == "-f":
                save_dic[find_dic_list[ind * 2]] = float(save_dic_list[ind * 2 + 1][2:])
    else:
        save_dic = {}

    print(find_dic)
    print(save_dic)
    # dataframe = nt.Database.read_data_into_pandas("Topology_Data", "BA", find_dic={})[4200:4400]
    # dataframe = nt.Database.read_data_into_pandas("Topology_Data", "ER", find_dic={})[4200:4400]
    dataframe = nt.Database.read_data_into_pandas("Topology_Data", data, find_dic=find_dic)[data_ind_start:data_ind_end]
    if rename == 1:
        dataframe["ILP RWA assignment"] = dataframe["ILP capacity RWA assignment"]

    print(len(dataframe["ILP RWA assignment"].dropna()))
    dataframe = dataframe.dropna(subset=["ILP RWA assignment"])
    simulators = [ScaledThroughput.remote() for i in range(workers)]
    data_len = len(dataframe)
    increment = np.floor(data_len / workers)
    start_stop = list(range(0, data_len, int(increment)))
    if len(start_stop) == workers:
        start_stop.append(data_len)
    else:
        start_stop[-1] = data_len

    results = ray.get([s.calculate_scaled_throughput_data.remote(dataframe.iloc[start_stop[ind]:start_stop[ind+1]],
                                                                 graph_type=data, save_dic=save_dic,
                                                                 start=range_start, end=range_end, step=range_step)
                       for ind, s in enumerate(simulators)])




