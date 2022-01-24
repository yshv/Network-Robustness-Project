import NetworkToolkit as nt


def decide_gate_nodes_set(subgraph_i, subgraph_j, T_c, num_nodes):
    """
    Method to decide the gate nodes between two subgraphs in the HTD process.
    :param subgraph_i: subgraph for which the gate nodes are to be chosen
    :param subgraph_j: subgraph for which the target traffic is for
    :param T_c: normalised traffic matrix in terms of connections between the two subgraphs
    :param num_nodes: number of nodes to choose for - has to be less or the same as len(subgraph_i)
    :return: set of gate nodes from subgraph_i - list
    """

    # assert that the number of nodes to be
    # chosen is not larger than the subgraph
    assert num_nodes <= len(subgraph_i)
    traffic_sum = []
    for node_i in subgraph_i.nodes:
        traffic = 0
        for node_j in subgraph_j.nodes:
            traffic += T_c[node_i - 1, node_j -1]
        traffic_sum.append((node_i, traffic))
    traffic_sum_sorted = traffic_sum.sort(reverse=True, key=lambda x: x[1])
    gate_nodes = list(map(lambda x: x[0], traffic_sum_sorted))[:num_nodes]
    return gate_nodes

def decide_gate_nodes_all(graph, subgraphs, T_c):
    """
    Method to decide gate nodes of all subgraphs.
    :param graph: original graph (set of nodes) to design the network for
    :param subgraphs: set of individual subgraphs for which to do the inter and intra network designs
    :param T_c: normalised traffic matrix in terms of connections for the design process
    :return: list of gate nodes for the different subnetworks - list of list
    """
    pass

def determine_inter_subnetwork_gate_edges_set(subgraph_i, subgraph_j, gatenodes_i, gatenodes_j):
    """
    Method to determine which gate nodes are connected to each other between two seperate subgraphs.
    :param subgraph_i: one of the subnetworks
    :param subgraph_j: the other subnetwork
    :param gatenodes_i: gate nodes to be used in subgraph_i and to be connected to some gate nodes in subgraph_j
    :param gatenodes_j: gate nodes to be used in subgraph_j and to be connected to some gate nodes in subgraph_j
    :return: edges of original nodes to be connected - list(tuples)
    """
    pass

def determine_inter_subnetwork_gate_edges_all(graph, subgraphs, T_c):
    """
    Method to determine all the edges to connect the different gate nodes between the different subnetworks.
    :param graph: original graph (set of nodes) to design the network for
    :param subgraphs: set of individual subgraphs for which to do the inter and intra network designs
    :param T_c: normalised traffic matrix in terms of connections for the design process
    :return: all edges defining the connections between the intersubnetwork problem - list(tuples)
    """
    pass



if __name__ == "__main__":
    pass