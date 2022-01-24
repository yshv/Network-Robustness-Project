
import logging
import math
import random
from itertools import islice
import networkx as nx
import numpy as np
import pandas as pd

def get_m_cut(graph, cost=5):
    """
    This method implements a heuristic to find the limiting cut in a network and the associated minimum wavelengths that occur with it.
    [reference: Wischik 1996]
    :param cost:    The cost by which to increase the most congested links - int
    :return:        m_cut
    :rtype: int
    """
    m_cut_graph = graph.copy()  # copy graph to not alter global variables for other methods
    m_cut_dict = {}  # dictionary to store weights for routing
    graph_edges = m_cut_graph.edges()  # edges of the graph
    for edge in graph_edges:  # initialising m_cut weight to 1 for each edge
        m_cut_dict[edge] = {
            "m_cut": 1}
        m_cut_dict[(edge[0], edge[1])] = {"m_cut": 1}
        nx.set_edge_attributes(self.RWA_graph, m_cut_dict)
    i = 0  # setting i to 0 as to skip the first iteration for old links
    while True:
        unique_shortest_paths = self.get_shortest_dijikstra_all(
            m_cut=True)  # get shortest unique paths weighted by the m_cut weight
        # print("unique shortest paths: {}".format(unique_shortest_paths))
        for path in unique_shortest_paths:  # add congestion for all unique node pair paths
            add_congestion(path=path[2], use_alternate_graph=True, alternate_graph=m_cut_graph)
        most_loaded_links = get_max_cong_link(m_cut_graph)  # get the max congestion links
        for edge in most_loaded_links:  # increase the cost
            graph[edge[0]][edge[1]]["m_cut"] += cost
        sorted_highest_cost_edges = get_highest_cost_m_cut_vals(graph, len(graph_edges))

        i += 1  # increment the loop and go back to finding shortest weighted paths
        if i == 1000:
            break
    C = 0
    for i in range(0, len(sorted_highest_cost_edges)):
        m_cut_graph.remove_edge(sorted_highest_cost_edges[i][0][0], sorted_highest_cost_edges[i][0][
            1])
        C += 1
        if nx.number_connected_components(m_cut_graph) > 1:
            break

    H = list(nx.connected_components(m_cut_graph))

    # from this we can get the size of nodes on each subgraph and the number of links that produce the cut C and calculate m_cut -> lambda_LL
    # print(len(H[0]))
    # print(len(H[1]))

    m_cut = math.ceil((len(H[0]) * len(H[1])) / C)  # m_cut = (K*N\K)/C

    return m_cut
    
def get_highest_cost_m_cut_vals(graph, amount):
    """
    Method to get the highest cost m cut values.
    :param graph:       Graph for which to use the m_cut values from - nx.Grap()
    :param amount:      Amount of values to get - int
    :return:            Sorted m cut values from highest to lowest - list
    """
    def sort_cost(elem):
        return elem[1]

    m_cut_vals = []
    edges = graph.edges()
    for edge in edges:
        m_cut_vals.append((edge, graph[edge[0]][edge[1]]["m_cut"]))
    m_cut_sorted = sorted(m_cut_vals.copy(), reverse=True, key=sort_cost)
    m_cut_sorted_amount = m_cut_sorted[0:amount - 1]

    return m_cut_sorted_amount
    
def get_shortest_dijikstra_all(graph, weighted=True, m_cut=False):
        """
        This method gets the shortest dijikstra paths for all node pairs. It uses the networkx library.
        :param graph:       Graph which to find shortest paths for - nx.Graph()
        :param weighted:    Whether to use the weights of graph - boolean
        :param m_cut:       For use with m_cut function - boolean
        :return:            Returns shortest unique paths
        :rtype:             list - [(S, D, [shortest path])]
        """
        if m_cut:
            shortest_paths = dict(nx.all_pairs_dijkstra_path(graph.copy(), weight="m_cut"))
        elif weighted and not m_cut:
            shortest_paths = dict(nx.all_pairs_dijkstra_path(graph.copy()))  # get shortest paths using nx
        elif not weighted:
            shortest_paths = dict(nx.all_pairs_dijkstra_path(graph.copy(), weight="None"))

        clone = shortest_paths.copy()  # copy shortest paths list
        shortest_unique_paths = []  # create new list to store unique lists
        for item in shortest_paths:
            for other in shortest_paths[item]:

                sort = sorted(clone[item][other])  # sort them in terms of length

                if other == item:  # if they are the same node pass
                    pass
                elif sort in shortest_unique_paths:  # if duplicated pass
                    pass
                else:
                    shortest_unique_paths.append((item, other, shortest_paths[item][other]))  # append the unique path

        shortest_unique_paths = remove_duplicates_in_paths(shortest_unique_paths)  # remove the rest of duplicates

        # logging.debug("shortest_unique_paths: {}".format(shortest_unique_paths))
        return shortest_unique_paths
        
def remove_duplicates_in_paths(paths):
        """
        This method takes in a set of paths and removes duplicate elements within this path.
        :param paths:   Takes in paths in format list((S, D, [path]), ...)
        :return:        List of paths
        :rtype:         List - [(S, D, [shortest path])]
        """
        for path in paths:  # iterate through paths
            for i in range(len(paths) - 1):
                logging.debug(path[2][-1])
                logging.debug(paths[i][2][-1])
                if path[2][0] == paths[i][2][-1] and path[2][-1] == paths[i][2][0] and path[2] != paths[i][
                    2]:  # compare all paths S and D pairs and remove duplicates

                    paths.remove(paths[i])
        return paths
        
def djikstras_shortest_paths(graph):  # CAREFUL: only gets for all edges
        """
        This method is out of date and is not used in the object...CAREFUL only gets the shortest path for all edges...
        :param graph:       Graph to use - nx.Graph()
        :return:            List of shortest paths - list
        """
        edges = graph.edges()
        shortest_paths = list(
            map(lambda edges: (edges[0], edges[1], nx.dijkstra_path(graph, edges[0], edges[1])), edges))
        shortest_paths = get_shortest_dijikstra_all()
        return shortest_paths
        
def sort_length(elem):  
        """
        This method is to sort k_shortest paths by length of the path(hops)
        :param elem:    The path to sort (S, D, [path])
        :return:        Length of that element
        :rtype:         int
        """
        return len(elem[2])
        
def sort_cost(elem):  # function to sort k_shortest paths by cost input format: ((s, d), cost, [path])
        """
        This method is to sort k_shortest paths by the weight of the element - cost
        :param elem:    The shortest path to sort (S, D, cost, [path])
        :return:        Cost
        :rtype:         float
        """
        return elem[1]
        
def path_cost(graph, path, weight=None):
        """
        Method to calculate a singular cost of a path, with weight or just length of path.
        :param graph:   Graph for which you want to use (depending on which weights you want to use) - nx.Graph()
        :param path:    Path which you want to evaluate - list
        :param weight:  Whether to use weights of graph or not - Boolean
        :return:        Cost of path
        :rtype:         float or int (weight or no weight)
        """
        pathcost = 0
        for i in range(len(path)):
            if i > 0:
                if weight != None:
                    logging.debug(graph.has_edge(path[i - 1], path[i]))
                    pathcost += graph.get_edge_data(path[i - 1], path[i])["weight"]
                else:
                    # just count the number of edges
                    pathcost += 1
        return pathcost
        
def add_congestion(graph, path, use_alternate_graph=False, alternate_graph=None):
        """
        This method adds congestion to all edges affected by the given path.
        :param graph:               Graph to add congestion to - nx.Graph()
        :param path:                Path on which to add congestion - list - [path]
        :param use_alternate_graph: Whether to use an alternate graph - DEPRECATED
        :param alternate_graph:     Alternate graph to use - DEPRECATED
        :return:                    Graph
        :rtype:                     nx.Graph()
        """
        if use_alternate_graph:
            path_edges = nodes_to_edges(path)
            for edge in path_edges:
                alternate_graph[edge[0]][edge[1]]["congestion"] += 1
            return alternate_graph
        path_edges = nodes_to_edges(path)
        for edge in path_edges:
            graph[edge[0]][edge[1]]["congestion"] += 1
        return graph
        
def remove_congestion(graph, path, use_alternate_graph=False, alternate_graph=None):
        """
        This methods removes congestion from all edges affected by the given path.
        :param graph:       Graph to remove congestion from - nx.Graph()
        :param path:        Path on which to add congestion - list - [path]
        :return:            Graph
        :rtype:             nx.Graph()
        """
        if use_alternate_graph:
            path_edges = nodes_to_edges(path)
            for edge in path_edges:
                alternate_graph[edge[0]][edge[1]]["congestion"] += 1
            return alternate_graph
        path_edges = nodes_to_edges(path)
        for edge in path_edges:
            graph[edge[0]][edge[1]]["congestion"] -= 1
        return graph
        
        
def nodes_to_edges(nodes):
        """
        Method to convert a path into a list of edges.
        :param nodes:       Path to convert - [path]
        :return:            List of edges - list
        :rtype:             [(edge), ..., (edge)]
        """
        edges = []
        for i in range(0, len(nodes) - 1):
            edges.append((nodes[i], nodes[i + 1]))
        return edges
        
def k_SP_to_list_of_paths(k_SP):
        """
        Method to convert the self.equal_cost_paths of form ((s, d), [[path1],...,[pathn]]), ...) to [[path1], ..., [pathn]]
        :param k_SP:    k_SP in format as stated above
        :return:        return the list of paths for the graph as stated above
        """
        list_SP = []
        for item in k_SP:
            for path in item[1]:
                list_SP.append(path)

        return list_SP

def k_shortest_paths(graph, source, target, k, weight=None):
    """
    Method to calculate the k-shortest paths for a source target destination - weighted or not.
    :param graph:   nx.Graph for which to calculate the k-shortest paths for.
    :param source:  source node to calculate for
    :param target:  target node to calculate for
    :param k:       k paths to return
    :param weight:  whether to weight or not
    :return:        list of paths
    """
    return list(islice(nx.shortest_simple_paths(graph, source, target, weight=weight), k))
def get_k_shortest_paths_MNH(graph, e=None, k=1, weighted=None):
    """
    Updated method to calculate the k-shortest paths for all node-pairs in a graph.
    :param graph:       nx.graph for which to calculate them
    :param e:           amount of paths to return
    :param weighted:    whether to weight the graphs
    :return:            return [((s,d),[[path],...,[path]])]
    """
    k_sp = []
    # Iterate through node pairs (+1 indexed nodes)
    for i in range(1,len(graph)+1):
        for j in range(1, len(graph)+1):
            if j < i:
                pass
            elif j == i:
                pass
            else:
                k_sp_paths = k_shortest_paths(graph, i, j, k, weight=weighted)
                # k_sp_paths = list(filter(lambda path: True if len(path)==e else False))
                k_sp.append(((i,j), k_sp_paths))

    # min(k_sp,key=lambda x: min(x[1])
    # min_path_len = len(min(min(k_sp, key=lambda x: len(min(x[1], key=lambda y: len(y))))[1], key=lambda z: len(z)))
    if e is not None:
        k_sp = list(map(lambda k_sp: ((k_sp[0][0], k_sp[0][1]), list(
            filter(lambda x: True if len(x) <= len(min(k_sp[1], key=lambda x: len(x)))+e else False, k_sp[1]))), k_sp))

    return k_sp

def get_k_shortest_paths_MNH_deprecated(graph, e=0, weighted=False, limit=100):
        """
        DEPRICATED - New version of method above...
        This method gets the k_shortest paths for MNH (Minimum Number of Hops). It uses the Yens algorithm.
        :graph:             Graph to use for k shortest paths - nx.Graph()
        :param e:           Length longer than MNH to allow - int
        :param weighted:    Whether to use graph weights - boolean
        :param limit:       Limit of k - int
        :return:            List of equal cost paths - list
        """
        shortest_paths_all = get_shortest_dijikstra_all(graph, weighted=weighted)  #
        # get all shortest paths for node pairs

        k = 2  # initialising k as 2
        k_shortest_ordered_min = []  # list to hold the equal cost paths for all nodes

        def sort_length(elem):  # function to sort k_shortest by length of path
            return len(elem[2])

        for path in shortest_paths_all:  # for all node pairs
            k = 2

            while k <= limit:  # k < set limit in parameters
                k_shortest = yens_k_shortest_paths(graph, k, path)  # find the k_shortest paths

                k_shortest = sorted(k_shortest, key=sort_length)  # sort these according to length of paths
                k_shortest = list(map(lambda x: x[2], k_shortest))  # take only the paths ([[path] ...]

                if len(k_shortest[-1]) > len(k_shortest[
                                                 0]):  # if first value (shortest path) is smaller in length that path[k] filter the list for paths only as long as shortest path and break
                    y = len(k_shortest[0]) + e

                    k_shortest = list(filter(lambda x: len(x) <= y, k_shortest))

                    if graph.has_edge(k_shortest[0][0], k_shortest[0][-1]) and [k_shortest[0][0],
                                                                                         k_shortest[0][
                                                                                             -1]] not in k_shortest:
                        k_shortest.append([k_shortest[0][0], k_shortest[0][-1]])

                    k_shortest_ordered_min.append(((path[0], path[1]), k_shortest))

                    break
                elif len(k_shortest[-1]) <= len(k_shortest[0]) + e or len(k_shortest[-1]) == len(k_shortest[0]):
                    k *= 2
                if k >= limit:
                    k_shortest_ordered_min.append(((path[0], path[1]),
                                                   k_shortest))  # if path[k] is same length as shortest path increment k by a factor of 2

            # logging.debug("k_shortest_ordered_min: {}".format(k_shortest_ordered_min))

        return k_shortest_ordered_min
        
        

def get_k_shortest_paths_SNR(graph, bandwidth=0.9, limit=50):
    """
    This method gets the k shortest paths with SNR weighting.
    :param graph:       The graph to use for k shortest paths - nx.Graph()
    :param bandwidth:   The relaxation on the highest SNR path. (0.9-90% throughput of best SNR path is enough - float
    :param limit:       Limit of k - int
    :return:            Sorted k shortest paths
    :rtype:             list - [(S, D, cost, [path], ...]
    """

    shortest_paths_all = get_shortest_dijikstra_all(weighted=True)
    k_shortest_ordered_min = []

    # choose best SNR path(smallest NSR) and calculate throughput x
    for path in shortest_paths_all:
        k = 2
        while k <= limit:
            k_shortest = yens_k_shortest_paths(graph, k,
                                                    path)  # get k_shortest paths with yens algorithm for link lengths
            k_shortest = update_path_cost(graph, k_shortest,
                                               weight="NSR")  # update path costs for NSR
            k_shortest = sorted(k_shortest, key=sort_cost,
                                reverse=True)  # sort according to best SNR (smallest NSR is best SNR)
            k_shortest_paths = list(
                map(lambda x: x[2], k_shortest))  # get the k_paths instead of cost and other info

            capacity = calculate_capacity_lightpath(
                (1 / k_shortest[-1][1]))  # taking inverse of smallest NSR to calculate capacity with SNR
            C1 = capacity * bandwidth  # criteria C1 - 90% of capacity of best SNR link
            worst_capacity = calculate_capacity_lightpath((1 / k_shortest[0][1]))
            if worst_capacity >= C1:  # is largest NSR (worst SNR) above 90% threshold
                # keep going k *= 2  # if not then it orders values that are above C1 and appends the equal cost paths, otherwise it increments k*=2 and tries again
                if 2 * k > limit:
                    k_shortest_paths = list(
                        filter(lambda x: calculate_capacity_lightpath((1 / x[1])) > C1, k_shortest))
                    k_shortest_paths = list(map(lambda x: x[2], k_shortest_paths))
                    if [(path[0], path[1])] not in k_shortest_paths and graph.has_edge(path[0], path[1]):
                        k_shortest_paths.append([path[0], path[1]])
                    k_shortest_ordered_min.append(((path[0], path[1]), k_shortest_paths))

                k *= 2
            else:
                # filter values of k_shortest paths for only bigger than C2
                logging.debug("done")
                logging.debug("answer: {}".format(((path[0], path[1]), k_shortest_paths)))
                k_shortest_paths = list(
                    filter(lambda x: calculate_capacity_lightpath((1 / x[1])) > C1, k_shortest))
                if len(k_shortest_paths) == 0:
                    # k_shortest_paths.append(k_shortest_paths[2][-1])
                    logging.debug("empty path!!! k_shortest: {}".format(k_shortest_paths))
                k_shortest_paths = list(map(lambda x: x[2], k_shortest_paths))
                if [(path[0], path[1])] not in k_shortest_paths and graph.has_edge(path[0], path[1]):
                    k_shortest_paths.append([path[0], path[1]])
                k_shortest_ordered_min.append(((path[0], path[1]), k_shortest_paths))
                break
        logging.debug("k_shortest_ordered_min: {}".format(k_shortest_ordered_min))
    return k_shortest_ordered_min

def yens_k_shortest_paths(graph, k, shortest_path):
        """
        This method is the base Yen's k shortest path algorithm, to be used in conjunction with SNR k shortest paths, or MNH shortest paths.
        :param k:               Value of k - int
        :param shortest_path:   Shortest path to be evaluated - list
        :return:                k shortest paths
        :rtype:                 list - [(S, D, cost, [path], ...]
        """
        # get the shortest paths for the whole graph
        k_shortest = []  # array for the k shortest paths of the function to return

        source = shortest_path[2][0]  # source of original shortest path
        destination = shortest_path[2][-1]  # destination of original shortest path
        A = [shortest_path[
                 2]]  # initialising k-shortest path list holding the first shortest path[[k-shortest-path] ... []]

        k_shortest.append(
            ((source, destination), path_cost(graph, shortest_path[2], weight=True), shortest_path[
                2]))  # append initial shortest path to the final value

        for _k in range(1, k):
            graph_copy = graph.copy()  # copy input graph
            try:
                for i in range(len(A[_k - 1]) - 1):

                    spurNode = A[_k - 1][i]  # spur node is retrieved from the previous k-shortest path (k-1)
                    rootpath = A[_k - 1][
                               :i]  # the root path is the sequence of nodes leading from the source to the spur node
                    removed_edges = []  # empty array to hold removed edges to add back in later
                    for path in A:
                        if len(path) - 1 > i and rootpath == path[
                                                             :i]:  # if the rootpath is the same as in another path in k-shortest paths
                            edge = (path[i], path[i + 1])  # the edge is removed
                            edges = nodes_to_edges(rootpath)
                            if not graph_copy.has_edge(edge[0], edge[1]):
                                continue
                            removed_edges.append(
                                (path[i], path[i + 1],
                                 graph.copy().get_edge_data(path[i], path[i + 1])["weight"]))
                            for item in edges:
                                removed_edges.append(
                                    (item[0], item[1], graph.copy().get_edge_data(item[0], item[1])["weight"]))
                                graph_copy.remove_edge(item[0], item[1])

                            graph_copy.remove_edge(path[i], path[i + 1])

                    try:
                        spurpath = nx.dijkstra_path(graph_copy, spurNode,
                                                    destination)  # the new shortest path is then found
                        total_path = rootpath + spurpath  # the total path of the new path is addition of rootpath and spurpath
                        total_path_cost = path_cost(graph.copy(), total_path,
                                                         weight=True)  # calculating cost of new path
                        k_shortest.append(((shortest_path[0], shortest_path[1]), total_path_cost, total_path))
                        A.append(total_path)
                        for removed_edge in removed_edges:  # adding in the removed edges again
                            graph_copy.add_weighted_edges_from([removed_edge]
                                                               )

                    except Exception as err:
                        break
            except Exception as e:
                break

        return k_shortest
        

def get_k_e(graph, k_shortest_paths_only):
        """
        
        :param k_shortest_paths_only:
        :return:
        """
        edges = graph.edges()
        # print(edges)
        k_e = np.zeros((len(k_shortest_paths_only), len(edges)))

        for path in k_shortest_paths_only:
            links = nodes_to_edges(path)
            for link in links:
                if link in edges:
                    # print(link)
                    # print(edges)
                    path_ind = k_shortest_paths_only.index(path)
                    try:
                        link_ind = list(edges).index(link)
                    except:
                        link_ind = list(edges).index((link[1], link[0]))
                    k_e[path_ind][link_ind] = 1
        return k_e
        
def get_I(graph, k_SP):
        """
        This method gets the I variable for the ILP, where I(j element path) is 1 if link j is in path p, otherwise 0. Evaluated for all k shortest paths and node pairs.
        :return: returns the set I
        :rtype: list like
        """

        E = list(graph.edges())
        I = [[[0 for e in range(len(E))] for k in range(len(z[1]))] for z in k_SP]
        #print(I)
        for z in k_SP:
            for k in z[1]:
                for e in list(graph.edges()):
                    path = nodes_to_edges(k)
                    # print(e)
                    # print(path)
                    # print(z[1])
                    if e in path:
                        z_i = k_SP.index(z)
                        k_i = z[1].index(k)
                        e_i = E.index(e)
                        #print("i[{}][{}][{}]".format(z_i, k_i, e_i))
                        I[z_i][k_i][e_i] = 1
                    elif (e[1], e[0]) in path:
                        z_i = k_SP.index(z)
                        k_i = z[1].index(k)
                        #print((e[1], e[0]))
                        e_i = E.index((e))
                        I[z_i][k_i][e_i] = 1
                    else:
                        pass
        return I
        
def get_most_loaded_path_link(graph, path):
    """
    This method returns the most loaded path link in a graph
    :param graph:   Graph to use - nx.Graph()
    :param path:    Path of which to scan the graphs edges - list
    :return:        Most congested link
    :rtype: tuple
    """
    path_edges = nodes_to_edges(path)
    cong = list(map(lambda x: graph.get_edge_data(x[0], x[1])["congestion"], path_edges))
    most_cong_link = path_edges[cong.index(max(cong))]
    return most_cong_link

def get_most_loaded_path_link_cong(graph, path):
    """
    This method returns the maximum congestion along a path.
    :param graph:   Graph to use - nx.Graph()
    :param path:    Path of which to scan the graphs edges - list
    :return:        Max congestion along path
    :type: int
    """
    path_edges = nodes_to_edges(path)
    cong = list(map(lambda x: graph.get_edge_data(x[0], x[1])["congestion"], path_edges))
    cong_max = max(cong)
    return cong_max

def get_sum_congestion_path(graph, path):
    """
    This method returns the sum of congestion over a path in the graph.
    :param graph:       Graph to use - nx.Graph()
    :param path:        Path for which to sum the congestion of the graph - list
    :return:            Sum of congestion over path
    :rtype: int
    """
    path_edges = nodes_to_edges(path)
    cong = list(map(lambda x: graph.get_edge_data(x[0], x[1])["congestion"], path_edges))
    cong_sum = sum(cong)
    return cong_sum

def check_congestion(graph, path_old, path_new):
    """
    Method to check whether new path reduces congestion compared to old path.
    :param graph:       Graph to use - nx.Graph()
    :param path_old:    Old path being used - list
    :param path_new:    New path to be analysed - list
    :return:            Whether to replace or not - boolean
    """

    replace = False
    most_cong_link_old = get_most_loaded_path_link(graph, path_old)
    most_cong_link_new = get_most_loaded_path_link(graph, path_new)
    logging.debug("most cong link old: {} most cong link new: {}".format(most_cong_link_old, most_cong_link_new))
    logging.debug(
        "old cong: {} new cong: {} ".format(
            graph[most_cong_link_old[0]][most_cong_link_old[1]]["congestion"],
            graph[most_cong_link_new[0]][most_cong_link_new[1]]["congestion"]))
    if graph[most_cong_link_old[0]][most_cong_link_old[1]]["congestion"] > \
            graph[most_cong_link_new[0]][most_cong_link_new[1]]["congestion"] + 1:
        logging.debug(
            "new congestion: {}".format(
                graph[most_cong_link_new[0]][most_cong_link_new[1]]["congestion"] + 1))
        replace = True

    else:
        replace = False
    return replace

def init_congestion_zero(graph):
    """
    Method to initialise congestion to zero on all links.
    :param graph:   Graph to initialise on - nx.Graph()
    :return:
    """
    copy_graph = graph.copy()
    edges = graph.edges()
    for edge in edges:
        graph[edge[0]][edge[1]]["congestion"] = 0

def update_congestion(graph, LA=None, rwa_assignment=None):
    """
    Method to upadate the congestion of a graph based on a lightpath assignment.
    :param graph:           Graph to use - nx.Graph()
    :param LA:              If using light paths, lightpaths to add congestion - list
    :param rwa_assignment:  If using full RWA assignment - dict
    :return:                New graph with updated congestion - nx.Graph()
    """
    copy_graph = graph.copy()
    edges = graph.edges()
    for edge in edges:
        graph[edge[0]][edge[1]]["congestion"] = 0
    if LA:
        for path in LA:
            logging.debug(path)
            add_congestion(graph, path[2])
    elif rwa_assignment:
        for key in rwa_assignment.keys():
            for path in rwa_assignment[key]:
                add_congestion(graph,path)
    return graph
def check_rwa_validity(rwa):
    """

    :param rwa:
    :return:
    """
    for key in rwa.keys():
        for path in rwa[key]:
            edges = nodes_to_edges(path)
            for s,d in edges:
                edge_count = 0
                for _path in rwa[key]:
                    path_edges = nodes_to_edges(_path)
                    edge_count += path_edges.count((s,d))
                    edge_count += path_edges.count((d,s))
                # print("edge count: {}".format(edge_count))
                    if edge_count >= 2:
                        pass
                        # print(s,d)
                        # print(path_edges)
                        # print(rwa[key])
                        # exit()
                    assert edge_count < 2





def get_max_cong_link(graph):
    """
    Method to return the maximum congested link in a graph.
    :param graph:   Graph to use - nx.Graph()
    :return:        List of links with max congestion - list
    """
    
    max_congestion = get_max_cong(graph)
    max_cong_links = list(filter(lambda x: graph[x[0]][x[1]]["congestion"] == max_congestion, edges))

    return max_cong_links

def get_max_cong(graph):
    """
    Method to get the maximum congestion number in a link in a graph.
    :param graph:   Graph to use - nx.Graph()
    :return:        Maximum congestion in graph - int
    """
    congestion = []
    edges = graph.edges()
    for edge in edges:
        congestion.append(graph[edge[0]][edge[1]]["congestion"])

    logging.debug(congestion)
    max_congestion = max(congestion)

    return max_congestion

def get_average_congestion(graph):
    """
    Method to find average congestion in a graph.
    :param graph:   Graph to use - nx.Graph()
    :return:        Average congestion within graph - float
    """

    edges = graph.edges()
    sum_congestion = 0
    for edge in edges:
        sum_congestion += graph[edge[0]][edge[1]]["congestion"]
    avg_congestion = sum_congestion / len(edges)
    return avg_congestion

def get_connection_requests_k_shortest_paths(k_SP, connection_requests):
    """
    Method to create list of k shortest paths based on the TM given by connection requests.

    :param k_SP:
    :param connection_requests:
    :return:
    :rtype:
    """

    s_d_pairs = k_SP.copy()
   # s_d_pairs = [(s_d_pairs[i][0][0], s_d_pairs[i][0][1], s_d_pairs[i][1], i) for i in range(len(s_d_pairs))]
    s_d_pairs = list(
        map(lambda x: (x[0][0], x[0][1], x[1]), s_d_pairs))  # converting s-d pairs without tuple for sd

    s_d_pairs = np.asarray(s_d_pairs)  # taking as numpy array
    s_d_pairs_with_traffic = np.zeros(
        (3,))  # creating numpy array for s-d based on traffic matrix TODO: better way to do this? -07/02/2020
    for item in s_d_pairs:
        s_d_pairs_item = np.tile(item, (int(connection_requests[item[0] - 1, item[1] - 1]),
                                        1))  # repeating the s-d pairs based on demand matrix

        s_d_pairs_with_traffic = np.vstack(
            (s_d_pairs_with_traffic, s_d_pairs_item))  # stacking this on to the final s-d with traffic array

    s_d_pairs_with_traffic = np.delete(s_d_pairs_with_traffic, 0,
                                        0)  # delete the first zero item (initialised element) TODO: better way to do this? -07/02/2020
    return s_d_pairs_with_traffic






        
        
        
        
        
        
        
        
        
        
        
        

