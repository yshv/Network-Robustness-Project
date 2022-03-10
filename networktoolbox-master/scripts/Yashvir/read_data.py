


with open("../TYP_code/networktoolbox-master/scripts/Yashvir/15_36_BA_data/ILP-results-1.txt") as f:
    lines = f.readlines()[1:]

count = 0

lamda = []
edges = []

for line in lines:
    count += 1 
    edges.append(line[0:line.index(" ")])
    line = line[line.index(" ")+3:]
    lamda.append(line[0:line.index(" ")])






