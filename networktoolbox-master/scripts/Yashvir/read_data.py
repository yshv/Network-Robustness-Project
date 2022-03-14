import numpy as np
import networkx as nx

gradients = []
alge_list = []

start = 1
finish = 100

for i in range(start, finish +1):

    with open("../TYP_code/networktoolbox-master/scripts/Yashvir/15_36_ER_data/ILP-results-{}.txt".format(i)) as f:
        lines = f.readlines()[1:]


        lambda_list = []
        edge_list = []
        counter = 1

        for line in lines:
            edge = float(line[0:line.index(" ")])
            edge_list.append(edge)
            line = line[line.index(" ")+3:]
            lam = float(line[0:line.index(" ")])
            lambda_list.append(lam)

        slope, intercept = np.polyfit(edge_list, lambda_list,1)
        
        gradients.append(slope)

for i in range(start, finish +1):
    graph = nx.read_gpickle("/Users/yashvirsangha/Desktop/Third_Year_Project/TYP_Code/networktoolbox-master/scripts/Yashvir/15_36_ER_Data/36({})-0_0.342.gpickle".format(i))
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    alge_list.append(nx.algebraic_connectivity(graph))

file = open("networktoolbox-master/scripts/Yashvir/lambda_alge.txt", "w")
for index in range(len(gradients)):
    file.write(str(gradients[index]) + " " + str(alge_list[index]) + "\n")
file.close()




