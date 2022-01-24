import pymongo
import NetworkToolkit.Tools as Tools
import pandas as pd
import networkx as nx
import NetworkToolkit as nt
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
import ray
import numpy as np

user = "robin_admin"
pwd = "Focker_12UCL!"
port = 27019
# print("using port {} for database".format(port))

# Initial topology data dictionary template
topology_data_template = {"connectivity": 0,
                          "topology vector": 0,
                          "D FF_kSP": 0,
                          "D ILP": 0,
                          "D upper bound": 0,
                          "D lower bound": 0,
                          "algebraic connectivity": 0,
                          "cheeger constant": 0,
                          "total network capacity ILP": 0,
                          "total network capacity FF_kSP": 0,
                          "chromatic number": 0,
                          "RWA density ILP": 0,
                          "RWA density FF_kSP": 0,
                          "h": 0,
                          "topology data": {},
                          "RWA config": {},
                          "capacity": {},
                          "capacity lower": {},
                          "capacity upper": {},
                          "capacity FF_kSP": 0,
                          "nodes": 0,
                          "edges": 0,
                          "author": "Robin Matzner"}


def insert_graph(graph, db_name, collection_name, k=None,
                 scaling_factor=None, node_data=None,
                 use_pickle=False,
                 **kwargs):
    """
    Method to insert a graph into the database.
    :param graph:           Graph to save - nx.Graph()
    :param db_name:         Name of database to save to - string
    :param collection_name: Name of collection to save to - string
    :param k:               k value to save - int
    :param scaling_factor:  Scaling factor to save - float
    :param node_data:       Whether to save node data - boolean
    :param **kwargs:        Other arguments to save
    """
    # index = len(list(Database.read_data("Topology_Data", "topology_data",{})))
    # print(index)
    # Client to connect to database, change port number to whatever your tunnel is using
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    alpha_ACMN = lambda N_ACMN, L_ACMN: (2 * L_ACMN) / (
            N_ACMN * (N_ACMN - 1))  # lambda function for connectivity
    graph_data = nx.to_dict_of_dicts(graph)
    # print(graph_data)
    nt.Tools.graph_to_database_topology(graph)
    if use_pickle:
        graph_data = pickle.dumps(graph_data)
    else:
        graph_data = {
            str(y): {str(z): str(graph_data[y][z]) for z in graph_data[y].keys()} for y
            in graph_data.keys()}
    topology_dict = topology_data_template.copy()
    topology_dict["topology data"] = graph_data
    # print(graph_data)
    topology_dict["connectivity"] = alpha_ACMN(len(list(graph.nodes)),
                                               len(list(graph.edges)))
    topology_dict["nodes"] = len(list(graph.nodes()))
    topology_dict["edges"] = len(list(graph.edges()))
    topology_dict["topology vector"] = nt.Tools.create_binary_topology_vector(
        graph).tolist()
    topology_dict["mean k"] = k
    if scaling_factor != None:
        topology_dict["scaling factor"] = scaling_factor
    if node_data == True:
        node_data = nt.Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=use_pickle)
        topology_dict["node data"] = node_data
        # print(node_data)
    for key, value in kwargs.items():
        # print(key)
        # print(type(value))
        topology_dict[key] = value

    # if name != None:
    #    topology_dict["name"] = name
    # print(topology_dict)
    # print(type(topology_dict))
    # print(topology_dict)


    Tools.assert_python_types(topology_dict)


    insert_data(db_name, collection_name, topology_dict)
    client.close()


def insert_data(db_name, collection_name, data_dic):
    """
    Method to insert data into database.
    :param db_name:         database name to insert document into.
    :param collection_name: collection to insert document into.
    :parm data_dic:         dictionary of data to insert
    :return: insert id
    """
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    db = client[db_name]
    data = db[collection_name]
    inserted_tag = data.insert_one(data_dic)
    client.close()
    return inserted_tag


def find_dict_argv(*args):
    """
        Method to read data and return a pymongo query object.
        :param db_name:         database name to find document from.
        :param collection_name: collection to find document from.
        :param max_count: maximum amount of records to return
        :param argv: conditions for query in form "value1 <= $param$ >= value2" or
        $param$ == value1
        :return: pymongo query
        """
    import re
    find_dict = {}
    for arg in args:
        node_ind = re.search("\$([^']*)\$", arg).start()

        node = re.findall("\$([^']*)\$", arg)[0]
        # find_statement = find_statement.replace("$"+node+"$"+" ", "")
        # print(find_statement[:2])
        # cond = find_statement[:2]
        conditions = re.findall("[<>=^\s]", arg)
        conditions_ind = re.search("[<>=^\s]", arg).start()

        numbers = re.findall("[-+]?\d+\.\d+|\d+", arg)
        if len(numbers) == 0:
            numbers = arg.split()[-1]

        conds = []
        i = 0
        for ind, char in enumerate(conditions):
            if char == ">" or char == "<":
                if conditions_ind < node_ind and i == 0:
                    if conditions[ind] + conditions[ind + 1] == "> ":
                        cond_key = "$lt"
                        i += 1
                    elif conditions[ind] + conditions[ind + 1] == ">=":
                        cond_key = "$lte"
                        i += 1
                    elif conditions[ind] + conditions[ind + 1] == "< ":
                        cond_key = "$gt"
                        i += 1
                    elif conditions[ind] + conditions[ind + 1] == "<=":
                        cond_key = "$gte"
                        i += 1
                else:
                    if conditions[ind] + conditions[ind + 1] == "> ":
                        cond_key = "$gt"
                    elif conditions[ind] + conditions[ind + 1] == ">=":
                        cond_key = "$gte"
                    elif conditions[ind] + conditions[ind + 1] == "< ":
                        cond_key = "$lt"
                    elif conditions[ind] + conditions[ind + 1] == "<=":
                        cond_key = "$lte"
            elif char == "=":
                if conditions[ind] + conditions[ind + 1] == "==":
                    cond_key = "e"
                    conds.append(cond_key)
                    break
                else:
                    continue

            else:
                continue

            conds.append(cond_key)

        if len(conds) == 1 and conds[0] == 'e':
            try:
                find_dict[node] = float(numbers[0])
            except:
                find_dict[node] = numbers

            continue

        else:
            find_dict[node] = {}
        for ind, key in enumerate(conds):
            find_dict[node][key] = float(numbers[ind])

    return find_dict


def read_data(db_name, collection_name, *args, find_dic=None, max_count=1000000, skip=0, parralel=False):
    """
    Method to read data and return a pymongo query object.
    :param db_name:         database name to find document from.
    :param collection_name: collection to find document from.
    :parm find_dic:         dictionary including query.
    :return: pymongo query
    """
    # print(*argv)
    # Client to connect to database, change port number to whatever your tunnel is using
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    if not find_dic:
        find_dic = find_dict_argv(*args)
    db = client[db_name]
    data = db[collection_name]
    find_results = data.find(find_dic).limit(max_count).skip(skip)
    client.close()
    if parralel:
        return [find_results]
    else:
        return find_results


def delete_collection(db_name, collection_name):
    """
    Method to delete a collection.
    :param db_name:         database name to delete collection from.
    :param collection_name: collection to delete.
    :return: None
    """
    # ClientClient to connect to database, change port number to whatever your tunnel is using
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    db = client[db_name]
    collection = db[collection_name]
    collection.drop()
    client.close()


def read_data_into_pandas(db_name, collection_name, find_dic=None, max_count=1000000,
                          names=None, *args):
    """
    Method to read a query from the database into a pandas dataframe.
    :param db_name:         name of database to search
    :param collection_name: name of collection to search
    :param find_dic:        dictionary query to use for search, leave None if searching for names
    :param max_count:       maximum count of query results to return
    :param names:           name of topolgies to search for - can be a list or singular value
    :return:                pandas DataFrame object containing search results
    """
    # Client to connect to database, change port number to whatever your tunnel is using
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    if names != None:
        try:
            names_length = len(names)
            find_dic = {"name": {"$in": names}}
        except:
            find_dic = {"name": names}

    results = read_data(db_name, collection_name, find_dic=find_dic, max_count=max_count,
                        *args)
    df = pd.DataFrame(list(results))
    client.close()
    return df


def read_topology_dataset_list(db_name, collection_name, *args, find_dic=None,
                               node_data=False,
                               max_count=1000000, parralel=False, list_comprehension=False,
                               use_pickle=False, skip=0, workers=5):
    """
    Method that reads the database depending on the dictionary given and then returns the (graph, _id)
    tuple which allows for efficient graph computations without complex string manipulation.
    :param db_name:         database name to find data from
    :param collection_name: collection name to find data from
    :param find_dic:        {"properties to search for":<property value or range dict>}}
                            e.g. {"connectivity":0.35} - allow graphs with connectivies of 0.35
                            {"connectivity": {"$gt":0.35, "$lt":0.5}} - allow graphs with connectivities between 0.35 and 0.5
    :param node_data:       Boolean to know whether node_data should be read as well (
                            different mechanism)
    :param max_count:       maxmimum amount of graphs to return
    :return:                graph_list
    """
    # Client to connect to database, change port number to whatever your tunnel is using
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)

    results = read_data(db_name, collection_name, *args, find_dic=find_dic,
                        max_count=max_count, skip=skip)

    graph_list = []
    # graphs = [Tools.read_database_topology(result["topology data"]) for result in results]
    # if node_data:
    #     attributes = [nx.set_node_attributes(graph, Tools.read_database_node_data(result["node data"])) for
    #                   graph, result in
    #                   zip(graphs, results)]
    # graph_list = [(graph, result["_id"]) for (graph, result) in zip(graphs,results)]
    if list_comprehension:
        if node_data:
            graph_list = [(Tools.read_database_topology(result["topology data"], node_data=result["node data"]),
                           result["_id"]) for result in results]
        else:
            graph_list = [(Tools.read_database_topology(result["topology data"]), result["_id"]) for result in results]
    else:
        if parralel:
            if node_data:
                ray.init()

                apply_node_data = lambda result, args: (
                Tools.read_database_topology(result["topology data"], node_data=result["node data"]),
                result["_id"], *(result[arg] for arg in args))
                apply_node_data = ray.remote(apply_node_data)
                graph_list = ray.get(
                    [apply_node_data.remote(result, args) for result in tqdm(results, desc="reading data")])

                ray.shutdown()

                # graph_list = Parallel(n_jobs=workers)(delayed(apply_node_data)(results[start_stop[ind]:start_stop[ind+1]], args) for ind in tqdm(range(int(max_count/workers))))
            else:
                ray.init()
                apply_topology_data = lambda result, args: (
                Tools.read_database_topology(result["topology data"]), result["_id"], *(result[arg] for arg in args))
                apply_topology_data = ray.remote(apply_topology_data)
                graph_list = ray.get(
                    [apply_topology_data.remote(result, args) for result in tqdm(results, desc="reading data")])
                ray.shutdown()
            # split_data = lambda result, args: (result["topology data"], result["node data"], result["_id"], *(result[arg] for arg in args))
            # results = Parallel(n_jobs=workers)(delayed(split_data)(result, args) for result in results)
            # results = [(result["topology data"], result["node data"], result["_id"], *(result[arg] for arg in args)) for result in
            #            results]
            # print(type(results[0]))
            # graph_list = Parallel(n_jobs=workers)(delayed(read_topology_data)(item, node_data) for item in results)
        else:
            if node_data:
                graph_list = [(Tools.read_database_topology(result["topology data"], node_data=result["node data"]),
                               result["_id"], *(result[arg] for arg in args)) for result in
                              tqdm(results, desc="reading data")]
            else:
                graph_list = [(Tools.read_database_topology(result["topology data"]), result["_id"],
                               *(result[arg] for arg in args)) for result in tqdm(results, desc="reading data")]

            # for result in results:
            #     graph = Tools.read_database_topology(result["topology data"])
            #     # if type(result["topology data"]) is dict:
            #     #     graph = Tools.read_database_topology(result["topology data"], use_pickle=False)
            #     # else:
            #     #     graph = Tools.read_database_topology(result["topology data"], use_pickle=True)
            #     # graph = Tools.read_database_topology(result["topology data"], use_pickle=use_pickle)
            #     if node_data:
            #         nx.set_node_attributes(graph, Tools.read_database_node_data(result["node data"]))
            #         # if type(result["node data"]) is dict:
            #         #     nx.set_node_attributes(graph, Tools.read_database_node_data(result["node data"], use_pickle=False))
            #         # else:
            #         #     nx.set_node_attributes(graph, Tools.read_database_node_data(result["node data"], use_pickle=True))
            #
            #     additional_items = []
            #     for item in args:
            #         additional_items.append(result[item])
            #
            #     graph_id = result["_id"]
            #     # print(args)
            #     graph_list.append((graph, graph_id, *additional_items))

    client.close()
    return graph_list


def read_topology_data(result, node_data):
    # return 0
    topology_result = result[0]
    node_result = result[1]
    _id_result = result[2]
    if len(result) > 3:
        args = result[3:]
    else:
        args = ()
    print(_id_result)
    if type(topology_result) is dict:
        graph = Tools.read_database_topology(topology_result, use_pickle=False)
    else:
        graph = Tools.read_database_topology(topology_result, use_pickle=True)
    # graph = Tools.read_database_topology(result["topology data"], use_pickle=use_pickle)
    if node_data:
        if type(node_result) is dict:
            nx.set_node_attributes(graph, Tools.read_database_node_data(node_result, use_pickle=False))
        else:
            nx.set_node_attributes(graph, Tools.read_database_node_data(node_result, use_pickle=True))

    return (graph, _id_result, *args)


def update_data_with_id(db_name, collection_name, _id, newvals):
    """
    Method to update data given it's unique id (normally used with graph lists whih
    reutrns a graph and id which can be used to update values.
    :param db_name:         Database where to update data
    :param collection_name: Collection where to update data
    :param _id:             _id to use for the data.
    :param newvals:         newvals to which to assign data e.g. {"$set": {"S": S}}
    :return: None
    """
    # Client to connect to database, change port number to whatever your tunnel is using
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    db = client[db_name]
    data = db[collection_name]
    data.update_one({"_id": _id}, newvals)
    client.close()
    return


def convert_topology_to_pickle(db_name, collection_name,
                               new_collection_name,
                               *args, find_dic={},
                               max_count=100000000):
    """

    :param db_name:
    :param collection_name:
    :param new_collection_name:
    :param find_dic:
    :return:
    """

    results = read_data(db_name, collection_name, *args, find_dic=find_dic,
                        max_count=max_count)

    for item in results:
        data_dic = {}
        for key in item:
            if key == "topology data":
                graph = Tools.read_database_topology(item["topology data"], use_pickle=False)
                nx.set_node_attributes(graph, Tools.read_database_node_data(item["node data"], use_pickle=False))
                graph_data = nx.to_dict_of_dicts(graph)
                graph_data = pickle.dumps(graph_data)
                data_dic["topology data"] = graph_data

            elif key == "node data":
                node_data = Tools.node_data_to_database(dict(graph.nodes.data()), use_pickle=True)
                data_dic["node data"] = node_data
            else:
                data_dic[key] = item[key]
        insert_data(db_name, new_collection_name, data_dic)


if __name__ == "__main__":
    print("starting...")
    graph_list = read_topology_dataset_list("Topology_Data", "ER",
                                            {"nodes": 8, "mean k": 3})[:100]


