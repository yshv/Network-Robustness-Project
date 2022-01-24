


class ML():
    def __init__(self, graph, channels, channel_bandwidth):
        self.graph = graph
        self.channels = channels
        self.channel_bandwidth = channel_bandwidth

    def static_DRL_LA(self):
        pass
            # TODO: implement DRL reverse engineered heuristic for LA - Jose Suarez Valera - Deep-RMSA: A Deep-Reinforcement-Learning Routing, Modulation and Spectrum Assignment Agent for Elastic Optical Networks

    def static_DRL_PLI_LA(self):
        pass
        # TODO: implement DRL reverse engineered heuristic for LA with NLI heuristic

    def message_passing_routing(self, traffic_matrix_connection_requests,Q_layer=1, ntrial=1, niter=1000):
        """
        Method to calculate
        :param traffic_matrix_connection_requests:
        :return:
        """
        import mp_module as mp
        N = len(self.graph)
        edges = list(self.graph.edges())
        num_edges = len(edges)

        pairs = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(int(traffic_matrix_connection_requests[i,j])):
                    pairs.append((i+1,j+1))

        rwa = mp.mp(N,edges,pairs,Q_layer,ntrial,niter)
        return rwa
