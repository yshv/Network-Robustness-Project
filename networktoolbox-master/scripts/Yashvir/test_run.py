import NetworkToolkit as nt
import networkx as nx
import time

if __name__ == "__main__":
    # read graph
    # create T_c
    # run ILP

    nodes=15
    p_list = [0.342]
    lambda_list = []

    graph = nx.read_gpickle("/home/zceeysa/Desktop/TYP_Code/networktoolbox-master/scripts/Yashvir/test_data/fig2.gpickle")
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    assert type(graph) == nx.classes.graph.Graph
    network = nt.Network.OpticalNetwork(graph)
    print("starting wavelength requirement ILP")
    time_start = time.perf_counter()

    data = network.rwa.static_ILP(min_wave=True, max_time=10800, e=20, k=20, threads=20,
                                node_file_start=0.01)
    time_taken = time.perf_counter()-time_start
    print("time: {}s".format(time_taken))
    lambda_list.append((graph.number_of_edges(), data["objective"], data["status"], time_taken))


with open("/home/zceeysa/Desktop/TYP_Code/networktoolbox-master/scripts/Yashvir/test_data/ILP-fig_2.txt", 'w') as f:
    f.write("E \t lambda \t status \t time taken\n")
    for E, objective, status, _time in lambda_list:

        f.write("{} \t {} \t {} \t {}".format(E, objective, status, _time))
        f.write("\n")