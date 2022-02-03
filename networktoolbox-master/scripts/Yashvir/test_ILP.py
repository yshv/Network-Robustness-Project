import NetworkToolkit as nt
import networkx as nx
import time

if __name__ == "__main__":
    # read graph
    # create T_c
    # run ILP
    nodes=10
    p_list = [0.3]
    lambda_list = []
    for p in p_list:
        graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/Data/graph_10_0.3.gpickle".format(nodes,p))
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        assert type(graph) == nx.classes.graph.Graph
        network = nt.Network.OpticalNetwork(graph)
        print("starting wavelength requirement ILP")
        time_start = time.perf_counter()

        data = network.rwa.static_ILP(min_wave=True, max_time=3600, e=20, k=20, threads=20,
                                    node_file_start=0.01)
        time_taken = time.perf_counter()-time_start
        print("time: {}s".format(time_taken))
        lambda_list.append((graph.number_of_edges(), data["objective"], data["status"], time_taken))


    with open("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/Data/ILP_10-results.txt", 'w') as f:
        f.write("E \t lambda \t status \t time taken\n")
        for E, objective, status, time in lambda_list:

            f.write("{} \t {} \t {} \t {}".format(E, objective, status, time))
            f.write("\n")
