def read_demand(load,set_id, path="/home/uceeatz/Code/test/networktoolbox/scripts/dynamic-simulation/data"):
    filename = "{}/SNR-BA 100-node demand load_{}_{}.txt".format(path,load, set_id)
    with open(filename,'r') as f:
        lines = f.readlines()
        traffic_id = []
        sn = []
        dn = []
        establish = []

        for ind,line in enumerate(lines):
            if ind>=4:
                data = line.split()
    #             print(data)
                traffic_id.append(int(data[0]))
                sn.append(int(data[1]))
                dn.append(int(data[2]))
                establish.append(int(data[3]))

    demand_data = {"id": traffic_id, "sn": sn, "dn": dn, "establish": establish}
    
    return demand_data


if __name__ == "__main__":
    load = 1
    set_id = 8    
    demand_data = read_demand(load, set_id)      

    print(demand_data.keys())