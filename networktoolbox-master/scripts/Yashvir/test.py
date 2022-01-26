import NetworkToolkit as nt
import networkx as nx
import time

if __name__ == "__main__":
    # read graph
    # create T_c
    # run ILP
    nodes=15
    p_list = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    lambda_list = []
    for p in p_list:
        for i in range(1,8):
            graph = nx.read_gpickle("../TYP_code/networktoolbox-master/scripts/Yashvir/Data/graph_{}-{}_{}.gpickle".format(nodes,i,p))
            graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
            assert type(graph) == nx.classes.graph.Graph
            network = nt.Network.OpticalNetwork(graph)
            print("starting wavelength requirement ILP")
            time_start = time.perf_counter()
            

            data = network.rwa.static_ILP(min_wave=True, max_time=3600, e=20, k=20, threads=2,
                                        node_file_start=0.01)
            time_taken = time.perf_counter()-time_start
            print("time: {}s".format(time_taken))
            lambda_list.append((graph.number_of_edges(), data["objective"], data["status"], time_taken))
        with open("../TYP_code/networktoolbox-master/scripts/Yashvir/Data/ILP-{}-results.txt".format(i), 'w') as f:
            f.write("E \t lambda \t status \t time taken\n")
            for E, objective, status, time in lambda_list:

                f.write("{} \t {} \t {} \t {}".format(E, objective, status, time))
                f.write("\n")

