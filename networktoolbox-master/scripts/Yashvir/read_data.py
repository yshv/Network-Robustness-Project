import numpy as np

for i in range(1, 11):

    with open("../TYP_code/networktoolbox-master/scripts/Yashvir/15_36_BA_data/ILP-results-{}.txt".format(i)) as f:
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
        print(slope)




