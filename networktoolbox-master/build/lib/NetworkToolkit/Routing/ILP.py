from . import Tools
from ..PhysicalLayer import  PhysicalLayer
import math
# import gurobi
import numpy as np
from .. import Database
from .. import Tools as ntTools
from cffi import FFI
ffi = FFI()
CData = ffi.CData

class ILP():
    def __init__(self, graph, channels, channel_bandwidth):
        self.graph = graph
        self.channels = channels
        self.channel_bandwidth = channel_bandwidth

        self.SNR_list = None


    def maximise_uniform_set_demands(self):
        pass

    def minimise_congestion(self, T, e=0, k=1, solver_name="GRB", max_time=1000):
        import mip.model as mip
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)  # k shortest paths
        W = self.channels  # number of wavelengths
        K = len(k_SP)  # number of node pairs
        E = len(list(self.graph.edges()))  # number of edges
        edges = list(self.graph.edges)  # list of edges

        # ILP object model
        ILP = mip.Model('ILP', solver_name=solver_name)
        ILP.verbose = 0

        # Variables
        delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[_e][1], edges[_e][0]) in
                           Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                     _e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
            len(k_SP))]

        delta = [[[ILP.add_var(var_type="B") for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]
        epsilon = [ILP.add_var(var_type="I") for e in range(E)]
        # print("T: {}".format(T))
        # print("kSP: {}".format(k_SP))
        T_c = [T[z[0][0] - 1, z[0][1] - 1] for z in k_SP]
        # print("T_c: {}".format(T_c))

        # u_w = [ILP.add_var(var_type=mip.BINARY) for w in range(W)]

        # ILP.objective = mip.minimize(np.mean([mip.xsum(u_w[w] for w in range(W))] for e in E]))
        ILP.objective = mip.minimize(mip.xsum(epsilon[e] for e in range(E)))

        # Constraint 1
        for _e in range(E):
            for w in range(W):
                ILP.add_constr(mip.xsum(delta[i][k][w] * delta_i[i][k][_e] for i in range(len(k_SP)) for k in range(len(k_SP[i][1]))) <= 1)

        # constraint 2:
        for z in range(K):
            ILP.add_constr(mip.xsum(delta[z][k][w] for k in range(len(k_SP[z][1])) for w in range(W)) == T_c[z])

        # constraint 3:
        for e in range(E):
            ILP.add_constr(epsilon[e] >= mip.xsum(delta[z][k][w] * delta_i[z][k][e] for z in range(K) for k in range(
                len(
                k_SP[z][1])) for w in range(W)))
        # Optimise
        ILP.emphasis = 2
        ILP.optimize(max_seconds=max_time)
        if ILP.num_solutions:
            print("objective values: {}".format(ILP.objective_values))
        rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
        # print(ILP.objective_values)
        # print(rwa_assignment)
        if len(ILP.objective_values) < 1:
            return True
        return rwa_assignment
    def maximise_throughput(self,T, e=0, k=1, solver_name="GRB", max_time=1000):

        import mip.model as mip
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)  # k shortest paths
        W = self.channels  # number of wavelengths
        K = len(k_SP)  # number of node pairs
        E = len(list(self.graph.edges()))  # number of edges
        edges = list(self.graph.edges)  # list of edges
        # Add estimate of worst case SNR for every path found (full occupation)
        # pl = PhysicalLayer(self.graph, self.channels, self.channel_bandwidth)
        # pl.add_wavelengths_full_occupation(channels_full=self.channels)
        # pl.add_uniform_launch_power_to_links(self.channels)
        # pl.add_non_linear_NSR_to_links(channels_full=self.channels,
        #                                channel_bandwidth=self.channel_bandwidth)

        # ILP object model
        ILP = mip.Model('ILP', solver_name=solver_name)
        ILP.verbose = 0

        # Variables
        # SNR_list = pl.get_SNR_k_SP(self.channels, k_SP)
        SNR_list = self.SNR_list
        C = [[float(self.channel_bandwidth*np.log2(1+SNR_list[i][1][k])) for k in range(
                                len(SNR_list[i][1]))] for i in range(len(SNR_list))]

        delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[_e][1], edges[_e][0]) in
                           Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                     _e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
            len(k_SP))]

        delta = [[[ILP.add_var(var_type="B") for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]
        # print("T: {}".format(T))
        # print("kSP: {}".format(k_SP))
        T_c = [T[z[0][0] - 1, z[0][1] - 1] for z in k_SP]
        # print("T_c: {}".format(T_c))

        # u_w = [ILP.add_var(var_type=mip.BINARY) for w in range(W)]

        ILP.objective = mip.maximize(mip.xsum(C[i][k]*delta[i][k][w] for i in range(K) for k in range(len(k_SP[i][
                                                                                                              1]))
                                              for w in range(W)))

        # Constraint 1
        for _e in range(E):
            for w in range(W):
                ILP.add_constr(mip.xsum(delta[i][k][w] * delta_i[i][k][_e] for i in range(
                    len(k_SP)) for k in range(len(k_SP[i][1]))) <= 1)

        # # constraint 2:
        # for z in range(K):
        #     for k in range(0, len(k_SP[z][1])):
        #         for w in range(W):
        #             ILP.add_constr(u_w[w] >= delta[z][k][w])
        #             ILP.add_constr(u_w[w] <= 1)
        #             # ILP += u_w[w] <= 1

        # constraint 3:
        for z in range(K):
            ILP.add_constr(mip.xsum(delta[z][k][w] for k in range(len(k_SP[z][1])) for w in range(W)) == T_c[z])

        # Optimise
        ILP.emphasis = 2
        ILP.optimize(max_seconds=max_time)
        # if ILP.num_solutions:
        #     print("objective values: {}".format(ILP.objective_values))
        rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
        # print(ILP.objective_values)
        # print(rwa_assignment)
        if len(ILP.objective_values) < 1:
            return True
        return rwa_assignment

    def minimise_wavelengths_used(self,T, e=0, k=1, solver_name="GRB", max_time=1000):
        """
        Method to route a connection request matrix via ILP with minimum amount of wavelengths used.

        :return: rwa_assignment, objective_values
        """
        import mip.model as mip
        from mip import Model, MAXIMIZE, CBC, INTEGER, OptimizationStatus
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k) # k shortest paths
        W = self.channels  # number of wavelengths
        K = len(k_SP)  # number of node pairs
        E = len(list(self.graph.edges()))  # number of edges
        edges = list(self.graph.edges)  # list of edges

        # ILP object model
        ILP = mip.Model('ILP', solver_name=solver_name)
        ILP.verbose = 0

        # Variables
        delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[_e][1], edges[_e][0]) in
                           Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                     _e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
            len(k_SP))]

        delta = [[[ILP.add_var(var_type="B") for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]
        # print("T: {}".format(T))
        # print("kSP: {}".format(k_SP))
        T_c = [T[z[0][0]-1, z[0][1]-1] for z in k_SP]
        # print("T_c: {}".format(T_c))

        u_w = [ILP.add_var(var_type="B") for w in range(W)]

        ILP.objective = mip.minimize(mip.xsum(u_w[w] for w in range(W)))

        # Constraint 1
        for _e in range(E):
            for w in range(W):
                ILP.add_constr(mip.xsum(delta[i][k][w] * delta_i[i][k][_e] for i in range(
                    len(k_SP)) for k in range(len(k_SP[i][1]))) <= 1)

        # constraint 2:
        for z in range(K):
            for k in range(0, len(k_SP[z][1])):
                for w in range(W):
                    ILP.add_constr(u_w[w] >= delta[z][k][w])
                    ILP.add_constr(u_w[w] <= 1)
                    # ILP += u_w[w] <= 1

        # constraint 3:
        for z in range(K):
            ILP.add_constr(mip.xsum(delta[z][k][w] for k in range(len(k_SP[z][1])) for w in range(W)) == T_c[z])

        # Optimise
        ILP.emphasis = 2
        ILP.optimize(max_seconds=max_time)
        # if ILP.num_solutions:
        #     print("objective values: {}".format(ILP.objective_values))
        rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
        # print(ILP.objective_values)
        # print(rwa_assignment)
        if len(ILP.objective_values) <1:
            return True
        return rwa_assignment
    def maximise_connection_demand(self, T_c=None, e=1, k=5,
                                   solver_name="GRB", max_time=1000, _id=0, node_file_start=0.1, threads=10,
                                   emphasis=2, max_gap=1e-4, max_solutions=100,
                                   node_file_dir="/scratch/datasets/gurobi/nodefiles"):
        """

        :param e:
        :param k:
        :param solver_name:
        :param max_time:
        :param _id:
        :param node_file_start:
        :param threads:
        :return:
        """
        import mip.model as mip
        import mip as MIP
        # Find k shortest paths for the graph given e
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        if T_c is not None:
            print("T_c ACTIVE")
            T_c = [T_c[item[0][0]-1][item[0][1]-1] for item in k_SP]
            for item in T_c:
                assert item is not np.NaN
                assert item != 0

        unique_paths = Tools.get_shortest_dijikstra_all(self.graph)

        W = self.channels  # number of wavelengths
        K = len(k_SP)  # number of node pairs
        E = len(list(self.graph.edges()))  # number of edges
        edges = list(self.graph.edges)  # list of edges

        # ILP object model
        ILP = mip.Model('ILP', solver_name=solver_name)
        ILP.verbose = 0
        ILP.threads = threads
        ILP.solver.set_dbl_param("NodeFileStart", node_file_start)

        # ILP.solver._model.setParam("NodeFileDir", "/scratch/datasets/gurobi/nodefiles")
        # gurobi.setParam("NodeFileStart", node_file_start)
        def set_str_param(param: str, value: str):
            MIP.gurobi.ffi.cdef("""int GRBsetstrparam(GRBenv *env, const char *paramname, const char *value); """,
                                override=True)

            print(MIP.gurobi.grblib)
            GRBsetstrparam = MIP.gurobi.grblib.GRBsetstrparam
            env = MIP.gurobi.GRBgetenv(ILP.solver._model)
            # error = GRBsetdblparam(env, param.encode("utf-8"), float(value))
            error = GRBsetstrparam(env, param.encode("utf-8"), value.encode("utf-8"))
            if error != 0:
                raise MIP.gurobi.ParameterNotAvailable(
                    "Error setting gurobi double param {}  to {}".format(param, value)
                )

        set_str_param("NodeFileDir", node_file_dir)
        # set_str_param("SolFiles", "/scratch/datasets/gurobi/solutions/{}".format(_id))

        # Variables
        delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[_e][1], edges[_e][0]) in
                           Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                     _e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
            len(k_SP))]

        delta = [[[ILP.add_var(var_type="B") for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]

        M = ILP.add_var(var_type="I")

        # Constraint 1
        for _e in range(E):
            for w in range(W):
                ILP.add_constr(mip.xsum(delta[i][k][w] * delta_i[i][k][_e] for i in range(
                    len(k_SP)) for k in range(len(k_SP[i][1]))) <= 1)

        # Constraint 2 - Replace ceiling function with +0.9999

        for i in range(len(k_SP)):
            ILP.add_constr(mip.xsum(delta[i][k][w] for k in range(
                len(k_SP[i][1])) for w in range(W)) <= M*T_c[i]+0.9999)
        for i in range(len(k_SP)):
            ILP.add_constr(mip.xsum(delta[i][k][w] for k in range(
                len(k_SP[i][1])) for w in range(W)) >= M*T_c[i])

        # Constraint 3
        ILP.add_constr(M >= 0)

        # Objective
        ILP.objective = M
        ILP.sense = "MAX"

        # Optimise
        ILP.emphasis = emphasis
        ILP.max_solutions = max_solutions
        ILP.max_mip_gap = max_gap
        status = ILP.optimize(max_seconds=max_time)
        if ILP.num_solutions:
            print("objective values: {}".format(ILP.objective_values))
        rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)

        data = {"rwa": rwa_assignment,
                "objective": ILP.objective_values[0],
                "status": status,
                "gap": ILP.gap}

        return data
    def maximise_uniform_connection_demand(self, e=1, k=5,
                                          solver_name="GRB", max_time=1000, _id=0, node_file_start=0.1, threads=10):
        """
        Method to maximise the uniform bandiwidth that is able to be routed.

        :param e:       Argument for find k-shortest paths + len(e) paths - int
        :solver_name:   Solver to use to solve the ILP problem  - string
        :return:        Objective value
        :rtype:         float
        """
        import mip.model as mip
        import mip as MIP
        # Find k shortest paths for the graph given e
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)

        unique_paths = Tools.get_shortest_dijikstra_all(self.graph)

        W = self.channels                       # number of wavelengths
        K = len(k_SP)                           # number of node pairs
        E = len(list(self.graph.edges()))       # number of edges
        edges = list(self.graph.edges)          # list of edges

        # ILP object model
        ILP = mip.Model('ILP', solver_name=solver_name)
        ILP.verbose = 0
        ILP.threads = threads
        ILP.solver.set_dbl_param("NodeFileStart", node_file_start)

        # ILP.solver._model.setParam("NodeFileDir", "/scratch/datasets/gurobi/nodefiles")
        # gurobi.setParam("NodeFileStart", node_file_start)
        def set_str_param(param: str, value: str):
            MIP.gurobi.ffi.cdef("""int GRBsetstrparam(GRBenv *env, const char *paramname, const char *value); """,
                                override=True)

            print(MIP.gurobi.grblib)
            GRBsetstrparam = MIP.gurobi.grblib.GRBsetstrparam
            env = MIP.gurobi.GRBgetenv(ILP.solver._model)
            # error = GRBsetdblparam(env, param.encode("utf-8"), float(value))
            error = GRBsetstrparam(env, param.encode("utf-8"), value.encode("utf-8"))
            if error != 0:
                raise MIP.gurobi.ParameterNotAvailable(
                    "Error setting gurobi double param {}  to {}".format(param, value)
                )

        set_str_param("NodeFileDir", "/scratch/datasets/gurobi/nodefiles")
        set_str_param("SolFiles", "/scratch/datasets/gurobi/solutions/{}".format(_id))

        # Variables
        delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[_e][1], edges[_e][0]) in
                           Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                        _e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
                        len(k_SP))]

        delta = [[[ILP.add_var(var_type="B") for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]

        M = ILP.add_var(var_type="I")

        #Constraint 1
        for _e in range(E):
            for w in range(W):
                ILP.add_constr(mip.xsum(delta[i][k][w]*delta_i[i][k][_e] for i in range(
                    len(k_SP)) for k in range(len(k_SP[i][1])))  <= 1)

        # Constraint 2

        for i in range(len(k_SP)):
            ILP.add_constr(mip.xsum(delta[i][k][w] for k in range(
                                            len(k_SP[i][1])) for w in range(W)) == M)

        # Constraint 3
        ILP.add_constr(M >= 0)

        #Objective
        ILP.objective = M
        ILP.sense = "MAX"

        #Optimise
        ILP.emphasis = 2
        ILP.optimize(max_seconds=max_time )
        if ILP.num_solutions:
            print("objective values: {}".format(ILP.objective_values))
        rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
        return rwa_assignment, ILP.objective_values[0]

    def minimise_total_path_length(self, T_c=None, e=1, k=5,
                                   solver_name="GRB", max_time=1000, _id=0, node_file_start=0.1, threads=10,
                                   emphasis=2, max_gap=1e-4, max_solutions=100,
                                   node_file_dir="/scratch/datasets/gurobi/nodefiles"):
        """

        :param e:
        :param k:
        :param solver_name:
        :param max_time:
        :param _id:
        :param node_file_start:
        :param threads:
        :return:
        """
        import mip.model as mip
        import mip as MIP
        # Find k shortest paths for the graph given e
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        if T_c is not None:
            print("T_c ACTIVE")
            T_c = [T_c[item[0][0] - 1][item[0][1] - 1] for item in k_SP]
            for item in T_c:
                assert item is not np.NaN
                assert item != 0

        unique_paths = Tools.get_shortest_dijikstra_all(self.graph)

        W = self.channels  # number of wavelengths
        K = len(k_SP)  # number of node pairs
        E = len(list(self.graph.edges()))  # number of edges
        edges = list(self.graph.edges)  # list of edges

        # ILP object model
        ILP = mip.Model('ILP', solver_name=solver_name)
        ILP.verbose = 0
        ILP.threads = threads
        ILP.solver.set_dbl_param("NodeFileStart", node_file_start)

        # ILP.solver._model.setParam("NodeFileDir", "/scratch/datasets/gurobi/nodefiles")
        # gurobi.setParam("NodeFileStart", node_file_start)
        def set_str_param(param: str, value: str):
            MIP.gurobi.ffi.cdef("""int GRBsetstrparam(GRBenv *env, const char *paramname, const char *value); """,
                                override=True)

            print(MIP.gurobi.grblib)
            GRBsetstrparam = MIP.gurobi.grblib.GRBsetstrparam
            env = MIP.gurobi.GRBgetenv(ILP.solver._model)
            # error = GRBsetdblparam(env, param.encode("utf-8"), float(value))
            error = GRBsetstrparam(env, param.encode("utf-8"), value.encode("utf-8"))
            if error != 0:
                raise MIP.gurobi.ParameterNotAvailable(
                    "Error setting gurobi double param {}  to {}".format(param, value)
                )

        set_str_param("NodeFileDir", node_file_dir)
        # set_str_param("SolFiles", "/scratch/datasets/gurobi/solutions/{}".format(_id))

        # Variables
        delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[_e][1], edges[_e][0]) in
                           Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                     _e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
            len(k_SP))]

        delta = [[[ILP.add_var(var_type="B") for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]

        M = ILP.add_var(var_type="I")

        # Constraint 1
        for _e in range(E):
            for w in range(W):
                ILP.add_constr(mip.xsum(delta[i][k][w] * delta_i[i][k][_e] for i in range(
                    len(k_SP)) for k in range(len(k_SP[i][1]))) <= 1)

        # Constraint 2 - route a connection for each entry of T^C_z

        for i in range(len(k_SP)):
            ILP.add_constr(mip.xsum(delta[i][k][w] for k in range(
                len(k_SP[i][1])) for w in range(W)) == T_c[i])

        # Constraint 3
        ILP.add_constr(M >= 0)

        # Objective
        ILP.objective = mip.xsum(delta[i][k][w]*len(k_SP[i][1][k]) for i in range(len(k_SP))
                                 for k in range(len(k_SP[i][1])) for w in range(W))
        ILP.sense = "MIN"

        # Optimise
        ILP.emphasis = emphasis
        ILP.max_solutions = max_solutions
        ILP.max_mip_gap = max_gap
        status = ILP.optimize(max_seconds=max_time)
        if ILP.num_solutions:
            print("objective values: {}".format(ILP.objective_values))
        rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)

        data = {"rwa": rwa_assignment,
                "objective": ILP.objective_values[0],
                "status": status,
                "gap": ILP.gap}

        return data

    def maximise_uniform_bandwidth_demand(self, e=1, k=5,
                                          solver_name="GRB", max_time=1000, T=0,
                                          shortest_paths_only=False, capacity_constraint=False,
                                          threads=4, node_file_start=0.1, collection="ilp-test", db="Topology_Data",
                                          _id=0, emphasis=0, node_file_dir="/scratch/datasets/gurobi/nodefiles",
                                          solution_file_dir="/scratch/datasets/gurobi/solutions/{}",
                                          c_type="C",
                                          verbose=0,
                                          mip_gap=1e-4):
        """
        Method to maximise the uniform bandiwidth that is able to be routed.

        :param e:       Argument for find k-shortest paths + len(e) paths - int
        :solver_name:   Solver to use to solve the ILP problem  - string
        :return:        Objective value
        :rtype:         float
        """
        import mip.model as mip
        import mip as MIP
        # print(_id)
        # Find k shortest paths for the graph given e
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)

        # Add estimate of worst case SNR for every path found (full occupation)
        pl = PhysicalLayer(self.graph,self.channels, self.channel_bandwidth)
        pl.add_wavelengths_full_occupation(channels_full=self.channels)
        pl.add_uniform_launch_power_to_links(self.channels)
        pl.add_non_linear_NSR_to_links(channels_full=self.channels,
                                       channel_bandwidth=self.channel_bandwidth)
        unique_paths = Tools.get_shortest_dijikstra_all(self.graph)
        if shortest_paths_only:
            SNR_list = pl.get_SNR_shortest_path_node_pair(self.channels, unique_paths)
        else:
            SNR_list = pl.get_SNR_k_SP(self.channels, k_SP)

        W = self.channels                       # number of wavelengths
        K = len(k_SP)                           # number of node pairs
        E = len(list(self.graph.edges()))       # number of edges
        edges = list(self.graph.edges)          # list of edges

        # ILP object model
        ILP = mip.Model('ILP', solver_name=solver_name)
        ILP.solver.set_dbl_param("NodeFileStart", node_file_start)
        # ILP.solver._model.setParam("NodeFileDir", "/scratch/datasets/gurobi/nodefiles")
        # gurobi.setParam("NodeFileStart", node_file_start)
        def set_str_param(param: str, value: str):
            MIP.gurobi.ffi.cdef("""int GRBsetstrparam(GRBenv *env, const char *paramname, const char *value); """,
                                override=True)

            # print(MIP.gurobi.grblib)
            GRBsetstrparam = MIP.gurobi.grblib.GRBsetstrparam
            env = MIP.gurobi.GRBgetenv(ILP.solver._model)
            # error = GRBsetdblparam(env, param.encode("utf-8"), float(value))
            error = GRBsetstrparam(env, param.encode("utf-8"), value.encode("utf-8"))
            if error != 0:
                raise MIP.gurobi.ParameterNotAvailable(
                    "Error setting gurobi double param {}  to {}".format(param, value)
                )
        if node_file_dir is not None:
            set_str_param("NodeFileDir", node_file_dir)
        if solution_file_dir is not None:
            set_str_param("SolFiles", solution_file_dir.format(_id))


        ILP.verbose = verbose
        ILP.threads = threads
        ILP.max_mip_gap=mip_gap

        # Variables
        delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[_e][1], edges[_e][0]) in
                           Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                        _e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
                        len(k_SP))]

        delta = [[[ILP.add_var(var_type="B", name='rwa-w-{}-k-{}-z-{}'.format(w,p,k)) for w in range(W)] for p in range(
            len(k_SP[k][1]))] for k in range(K)]

        ######################

        # def callback( p_model: CData, where: int) -> int:
        #     if where == gurobi.Callback.MIPSOL:
        #         delta_x=[[[0 for w in range(W)] for p in range(len(k_SP[k][1]))] for k in range(K)]
        #         for k in range(K):
        #             for p in range(len(k_SP[k][1])):
        #                 for w in range(W):
        #                     delta_x[k][p][w] = ILP.solver.var_get_x(delta[k][p][w])
        #         rwa = self.convert_delta_to_rwa_assignment(delta_x, k_SP, W)
        #         rwa = ntTools.write_database_dict(rwa)
        #         Database.update_data_with_id(db, collection, _id, newvals={"$set":{"ILP intermediate rwa":rwa}})
        # MIP.gurobi.GRBsetcallbackfunc(ILP.solver._model, *callback, ffi.NULL)
        ################

        if T==0:
            assert (len(self.graph)*(len(self.graph)-1))/2 == len(k_SP)
            T = [1/(len(self.graph)*(len(self.graph)-1)) for k in range(len(k_SP))]
        if shortest_paths_only:
            C = [[float(self.channel_bandwidth * np.log2(1 + SNR_list[i][2])) for k in range(
                len(k_SP[i][1]))] for i in range(len(SNR_list))]
        else:
            C = [[float(self.channel_bandwidth*np.log2(1+SNR_list[i][1][k])) for k in range(
                                len(SNR_list[i][1]))] for i in range(len(SNR_list))]

        c = ILP.add_var(var_type=c_type)
        # assert type(delta) is not None
        # assert type(delta_i) is not None
        # assert type(k_SP) is not None
        # print("k_SP: {}".format(type(k_SP)))
        # print("delta: {}".format(type(delta)))
        # print("delta_i: {}".format(type(delta_i)))
        # print("C_i: {}".format(type(C)))
        # print("T_i: {}".format(type(T)))
        #Constraint 1
        for _e in range(E):
            for w in range(W):
                ILP.add_constr(mip.xsum(delta[i][k][w]*delta_i[i][k][_e] for i in range(
                    len(k_SP)) for k in range(len(k_SP[i][1])))  <= 1)

        # Constraint 2

        for i in range(len(k_SP)):
            ILP.add_constr(mip.xsum(delta[i][k][w] * C[i][k] - c*T[i] for k in range(
                                            len(k_SP[i][1])) for w in range(W)) >= 0)
        if capacity_constraint:
            print("CAPACITY CONSTRAINT ACTIVE")
            for i in range(len(k_SP)):
                ILP.add_constr(mip.xsum(delta[i][k][w] * C[i][k] - c*T[i] for k in range(
                                                len(k_SP[i][1])) for w in range(W)) <= max(C[i]))
        # Constraint 3
        # ILP.add_constr(c >= 0)

        #Objective
        ILP.objective = c
        ILP.sense = "MAX"


        #Optimise
        ILP.emphasis = emphasis

        status = ILP.optimize(max_seconds=max_time )
        print("status: {}".format(status.value))
        # print("status: {}".format(status(0)))
        # print("status: {}".format(status(4)))
        # print("status: {}".format(str(status)))

        if ILP.num_solutions:
            print("objective values: {}".format(ILP.objective_values))
        for i in range(len(k_SP)):
            # print("achieved value: {}".format(
            a = sum([delta[i][k][w].x * C[i][k] - ILP.objective_values[0] * T[i] for k in range(len(k_SP[i][1]))for w in range(W)])
            #          for w in range(W)])))
            # if a <=0:
            #     print(a)
            assert a >= -0.1

        rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
        data = {"rwa":rwa_assignment,
                "objective":ILP.objective_values[0],
                "status":status,
                "gap":ILP.gap}
        return data




    def convert_delta_to_rwa_assignment(self, delta, k_SP, W, numeric_delta=False):
        if not numeric_delta:
            delta_int = [
                [[delta[i][k][w].x for w in range(W)] for k in
                 range(len(k_SP[i][1]))]
                for i in range(len(k_SP))]
        else: delta_int =delta
        rwa_assignment = {w: [] for w in range(W)}
        for i in range(len(k_SP)):
            for k in range(len(k_SP[i][1])):
                for w in range(W):
                    if delta_int[i][k][w] == 1:
                        rwa_assignment[w].append(k_SP[i][1][k])

        return rwa_assignment


    def static_ILP(self, max_time=600, min_wave=False, max_D=False, solver_name="GRB", e=1, k=5,
                   threads=10, node_file_start=0.1, node_file_dir="/scratch/datasets/gurobi/nodefiles"):
        """
        This method implements a static ILP solution for the k-shortest equal cost paths in a optical transport network.
        Make sure to call k-shortest paths (MNH or SNR) to get the k-shortest paths before
        calling this method.
        :return: Nothing, assigns a global RWA to wavelengths and lightpaths variables.
        :rtype: None
        """
        #from mip.model import *

        #k_SP = Tools.
        k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        # k_shortest_paths_only = []
        # for path in self.equal_cost_paths:
        #     # print(path[1])
        #     k_shortest_paths_only += path[1]
        import mip.model as mip
        import mip as MIP
        W = self.channels
        K = len(k_SP)
        # Z = int((len(self.graph.nodes()) * (len(self.graph.nodes()) - 1)) / 2)  # (N*(N-1))/2
        # Z = len(k_SP)
        E = len(list(self.graph.edges()))
        edges = list(self.graph.edges)

        # Create Model
        if solver_name == "GRB" or solver_name == "CBC":
            ILP = mip.Model('ILP', solver_name=solver_name)
            ILP.verbose = 0
            ILP.threads = threads
            ILP.solver.set_dbl_param("NodeFileStart", node_file_start)

            # ILP.solver._model.setParam("NodeFileDir", "/scratch/datasets/gurobi/nodefiles")
            # gurobi.setParam("NodeFileStart", node_file_start)
            def set_str_param(param: str, value: str):
                MIP.gurobi.ffi.cdef("""int GRBsetstrparam(GRBenv *env, const char *paramname, const char *value); """,
                                    override=True)

                print(MIP.gurobi.grblib)
                GRBsetstrparam = MIP.gurobi.grblib.GRBsetstrparam
                env = MIP.gurobi.GRBgetenv(ILP.solver._model)
                # error = GRBsetdblparam(env, param.encode("utf-8"), float(value))
                error = GRBsetstrparam(env, param.encode("utf-8"), value.encode("utf-8"))
                if error != 0:
                    raise MIP.gurobi.ParameterNotAvailable(
                        "Error setting gurobi double param {}  to {}".format(param, value)
                    )

            set_str_param("NodeFileDir", node_file_dir)
            if min_wave:
                # Variables
                # print(k_SP)
                # delta = [[[ILP.add_var(var_type="BINARY") for w in range(W)] for k in range(len(z[1]))] for z in
                #          k_SP]
                delta = [[[ILP.add_var(var_type="B") for w in range(W)] for p in range(
                    len(k_SP[k][1]))] for k in range(K)]
                # print(np.shape(delta[1]))
                u_w = [ILP.add_var(var_type="B") for w in range(W)]
                I = [[[1 if (edges[e] in Tools.nodes_to_edges(k_SP[i][1][k]) or (edges[e][1], edges[e][0]) in
                                   Tools.nodes_to_edges(k_SP[i][1][k])) else 0 for
                             e in range(E)] for k in range(len(k_SP[i][1]))] for i in range(
                    len(k_SP))]
                delta_int = [
                    [[delta[z][k][w].obj for w in range(W)] for k in range(0, len(k_SP[z][1]))]
                    for z in
                    range(len(k_SP))]
                # print(delta_int)
                ILP.objective = mip.minimize(mip.xsum(u_w[w] for w in range(W)))

                # constraint 1:
                for z in range(K):
                    ILP += mip.xsum(
                        delta[z][k][w] for w in range(0, W) for k in range(0, len(k_SP[z][1]))) == 1

                # constraint 2:
                for z in range(K):
                    for k in range(0, len(k_SP[z][1])):
                        for w in range(W):
                            ILP += u_w[w] >= delta[z][k][w]
                            ILP += u_w[w] <= 1
                # constraint 3:
                for z in range(K):
                    for k in range(0, len(k_SP[z][1])):
                        for w in range(W):
                            ILP += delta[z][k][w] >= 0

                # constraint 4:
                for j in range(E):
                    for w in range(W):
                        ILP += mip.xsum(delta[z][k][w] * I[z][k][j] for z in range(K) for k in
                                    range(0, len(k_SP[z][1]))) <= 1
                for w in range(W):
                    ILP += (u_w[w] >= 0)

                status = ILP.optimize(max_seconds=max_time)
                if ILP.num_solutions >= 1:
                    delta_int = [
                        [[delta[z][k][w].x for w in range(W)] for k in range(0, len(k_SP[z][1]))]
                        for z in range(len(k_SP))]
                    # print("equal cost paths: {}".format(k_SP))
                    # print("I: {}".format(I))
                    # print("delta: {}".format(delta_int))
                    u_w_int = [u_w[w].x for w in range(W)]
                    # print("u_w: {}".format(u_w_int))
                    # print("N_lambda: {}".format(ILP.objective_values))
                    ILP_sum_1 = [sum(
                        [sum([delta[z][k][w].obj for w in range(0, W)]) for k in
                         range(0, len(k_SP[z][1]))])
                        for z in range(0, len(k_SP))]
                    # print(ILP_sum_1)
                    self.wavelengths = {w: [] for w in range(W)}
                    for z in range(len(k_SP)):
                        for k in range(len(k_SP[z][1])):
                            for w in range(W):
                                if delta_int[z][k][w] == 1:
                                    self.wavelengths[w].append(k_SP[z][1][k])
                    # print("wavelengths: {}".format(self.wavelengths))
                    self.wavelength_max = 0
                    for key in self.wavelengths.keys():
                        if self.wavelengths[key]:
                            self.wavelength_max += 1
                    self.N_lambda = ILP.objective_values[0]
                    print("objective value: {}".format(ILP.objective_values[0]))
                    data = {"rwa": self.wavelengths,
                            "objective": ILP.objective_values[0],
                            "status": status,
                            "gap": ILP.gap}
                    return data
                else:
                    return None, None
            elif max_D:
                D = math.floor(156 / self.m_cut) + 50
                # D=100
                delta = [
                    [[[ILP.add_var(var_type="B") for w in range(W)] for k in range(len(self.equal_cost_paths[z][1]))]
                     for
                     z in range(
                        len(self.equal_cost_paths))] for d in range(D)]
                D_i = [ILP.add_var(var_type="B") for d in range(D)]
                I = self.get_I()
                ILP.objective = mip.maximize(mip.xsum(D_i[d] for d in range(D)))
                for d in range(D):
                    for z in range(len(self.equal_cost_paths)):
                        ILP += mip.xsum(
                            delta[d][z][k][w] for w in range(W) for k in range(len(self.equal_cost_paths[z][1]))) <= 1
                for d in range(D):
                    ILP += D_i[d] >= 0
                # constraint 1
                for d in range(D):
                    ILP += mip.xsum(
                        delta[d][z][k][w] for w in range(W) for z in
                        range(len(self.equal_cost_paths)) for k in range(len(self.equal_cost_paths[z][1]))) == len(
                        self.equal_cost_paths) * D_i[d]
                # constraint 2

                for d in range(D):
                    for z in range(len(self.equal_cost_paths)):
                        ILP += mip.xsum(
                            delta[d][z][k][w] for w in range(W) for k in range(len(self.equal_cost_paths[z][1]))) == \
                               D_i[d]
                # constraint 3
                for d in range(D):
                    for z in range(len(self.equal_cost_paths)):
                        for k in range(len(self.equal_cost_paths[z][1])):
                            for w in range(W):
                                ILP += delta[d][z][k][w] >= 0
                # constraint 4
                for w in range(W):
                    for j in range(E):
                        ILP += mip.xsum(delta[d][z][k][w] * I[z][k][j] for z in
                                    range(len(self.equal_cost_paths)) for k in range(len(self.equal_cost_paths[z][1]))
                                    for d
                                    in range(D)) <= 1

                for d in range(D):
                    ILP += D_i[d] <= 1
                    ILP += D_i[d] >= 0
                # ILP.emphasis = 1
                ILP.optimize()
                print("N_lambda: {}".format(ILP.objective_values))
                D_i_vales = [D_i[d].x for d in range(D)]
                print(D_i_vales)
                delta_values = [
                    [[[delta[d][z][k][w].x for w in range(W)] for k in range(len(self.equal_cost_paths[z][1]))] for z in
                     range(len(self.equal_cost_paths))] for d in range(D)]
                self.wavelengths = {w: [] for w in range(W)}
                print(self.wavelengths)
                for d in range(D):
                    for z in range(len(self.equal_cost_paths)):
                        for k in range(len(self.equal_cost_paths[z][1])):
                            for w in range(W):
                                if delta_values[d][z][k][w] == 1:
                                    self.wavelengths[w].append(self.equal_cost_paths[z][1][k])
                self.wavelength_max = list(self.wavelengths.keys())[-1]
                self.N_lambda = ILP.objective_values[0]

                print("wavelengths: {}".format(self.wavelengths))
        elif solver_name == "CPLEX":
            import docplex.mp.model as dp
            ILP = dp.Model()
            D = math.floor(156 / self.m_cut) + 10
            delta = [
                [[[ILP.binary_var() for w in range(W)] for k in range(len(self.equal_cost_paths[z][1]))] for
                 z in range(
                    len(self.equal_cost_paths))] for d in range(D)]
            D_i = [ILP.binary_var() for d in range(D)]
            I = self.get_I()
            ILP.minimize(ILP.sum(D_i[d] for d in range(D)))
            for d in range(D):
                for z in range(len(self.equal_cost_paths)):
                    ILP.add_constraint(ILP.sum(
                        delta[d][z][k][w] for w in range(W) for k in range(len(self.equal_cost_paths[z][1]))) <= 1)

            for d in range(D):
                ILP.add_constraint(D_i[d] >= 0)

            # constraint 1
            for d in range(D):
                ILP.add_constraint(ILP.sum(delta[d][z][k][w] for w in range(W) for z in
                                           range(len(self.equal_cost_paths)) for k in
                                           range(len(self.equal_cost_paths[z][1]))) == len(
                    self.equal_cost_paths) * D_i[d])
            # constraint 2

            for d in range(D):
                for z in range(len(self.equal_cost_paths)):
                    ILP.add_constraint(
                        ILP.sum(delta[d][z][k][w] for w in range(W) for k in range(len(self.equal_cost_paths[z][1]))) == \
                        D_i[d])
            # constraint 3
            for d in range(D):
                for z in range(len(self.equal_cost_paths)):
                    for k in range(len(self.equal_cost_paths[z][1])):
                        for w in range(W):
                            ILP.add_constraint(delta[d][z][k][w] >= 0)
            # constraint 4
            for w in range(W):
                for j in range(E):
                    ILP.add_constraint(ILP.sum(delta[d][z][k][w] * I[z][k][j] for z in
                                               range(len(self.equal_cost_paths)) for k in
                                               range(len(self.equal_cost_paths[z][1])) for d
                                               in range(D)) <= 1)

            for d in range(D):
                ILP.add_constraint(D_i[d] <= 1)
                ILP.add_constraint(D_i[d] >= 0)
            solution = ILP.solve()
            solution.print_information()