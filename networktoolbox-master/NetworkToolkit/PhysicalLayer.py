import logging

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# import NetworkToolkit.Topology as Topology
from NetworkToolkit.ISRSGNmodel import ISRSGNmodel, todB
from NetworkToolkit.Routing.Tools import add_congestion

#logging.basicConfig(level=26)


class PhysicalLayer():
    def __init__(self, graph, channels, channel_bandwidth):
        self.graph = graph
        self.channels = channels
        self.channel_bandwidth = channel_bandwidth
        self.assign_physical_wavelengths(channels=channels)
        self.to_norm = lambda x: (10 ** (x / 10))
        self.to_dB = lambda x: 10 * np.log10(x)

    def assign_physical_wavelengths(self, channels=156):
        """

        :return:
        """

        #B_cband = 5 * 10 ** 12
        #baud_rate = 32 * 10 ** 9
        #baud_rate = 32 * 10 ** 9
        #channels = int(B_cband / baud_rate)
        Cband_width = 40.049
        channel_spacing = Cband_width / channels
        wavelengths_physical = []
        for i in range(channels):
            wavelengths_physical.append((1530 + i * channel_spacing)*1e-9)
        self.wavelengths_physical = wavelengths_physical

    def add_SNR(self, channels, N, NF=10 ** (4 / 10), a=0.2, L=80, h=6.62607004 * 10 ** (-34),
                f=(3 * 10 ** 8) / (1550 * 10 ** (-9)), B_ref=32 * 10 ** 9, _lambda=1550 * 10 ** (-9), D=17, gamma=1.3):
        gain = a * L
        g = 10 ** (gain / 10)
        P_ase = NF * (g - 1) * h * f * B_ref
        eta = self.get_eta(B_ref, _lambda, a, L, D, gamma, channels)

        if N == 0:
            N = 1
        P_opt = (P_ase / 2 * eta) ** (1 / 3)
        # print(10*np.log10(P_opt))
        SNR_opt = (P_opt / ((N * P_ase) + (N * eta * P_opt ** 3)))

        # SNR_opt = 20*np.log10(SNR_opt)
        # print(20*np.log10(SNR_opt))

        NS_ratio = 1 / SNR_opt
        if NS_ratio < 1:
            logging.debug("++++++++++++++++++++++++++++++++++++infinite SNR++++++++++++++++++++++++++++++++++")
            logging.debug("NSR: {} P_opt: {} N: {} P_ase: {} eta: {}".format(NS_ratio, P_opt, N, P_ase, eta))
            pass
        return NS_ratio

    def calculate_capacity_lightpath(self, path_cost, Bref=32 * 10 ** 9):
        return 2 * Bref * np.log2(1 + path_cost)

    def calculate_throughput_path(self, path):
        pass

    def add_SNR_links(self):
        graph_links = self.graph.edges.data("weight")  # [(u, v, weight)....]
        SNR_links = list(map(lambda x: (x[0], x[1], self.add_SNR(channels=156, N=x[2])), graph_links))
        logging.debug(SNR_links)
        self.graph.add_weighted_edges_from(SNR_links)


    def get_eta(self, B_ref, _lambda, a, L, D, gamma, channels):
        c = 3 * 10 ** (8)
        a = a / 2 / 4.3463 / 10 ** (3)
        L = L * 1000
        Leff = (1 - np.exp(-2 * a * L)) / (2 * a)
        beta2 = -D * (10 ** (-12) / 10)
        gamma = gamma / (10 ** 3)
        BWtot = B_ref * channels
        eta = (2 / 3) ** 3 * gamma ** 2 * Leff * Leff * 2 * a * B_ref * np.arcsinh(
            (1 / 2) * np.pi ** 2 * abs(beta2) / 2 / a * BWtot ** 2) / (np.pi * abs(beta2 * (B_ref ** 3)))
        return eta

    def get_SPM_n(self, COI, gamma, bandwidth, alpha, D):
        """
        This method calculates the self phase modulation (SPM) noise coefficient for a given channel of interest(COI)

        :param frequency_i: frequency of COI
        :param gamma: non-linear-coefficient
        :param bandwidth: bandwidth of channel (32Gbaud channels - same bandwidth)
        :param C_r: slope of the linear regression of the normalized Raman gain spectrum
        :param alpha: attenuation coefficient
        :param beta_2: GVD parameter
        :param beta_3: slope of GVD parameter
        :return: SPM coefficient
        ":rtype: float
        """

        frequency_i = (3 * 10 ** 8) / self.wavelengths_physical[COI]  # getting frequency of COI
        S = D / (35 * 10 ** (-9))  # getting the slope of the dispersion parameter
        beta_2 = -self.wavelengths_physical[COI] ** 2 / 2 * np.pi * 3 * 10 ** 8  # calculating GVD parameter
        beta_3 = (S - ((4 * np.pi * 3 * 10 ** 8) / self.wavelengths_physical[COI] ** 3) * beta_2) / (
                (2 * np.pi * 3 * 10 ** 8) / self.wavelengths_physical[COI] ** 2)  # calculating slope of GVD

        phi_i = 12 * np.pi ** 2 * (beta_2 + 2 * np.pi * beta_3 * frequency_i)
        T_i = 2  # T_i for C-band at C_r = 0 due to assumption of ISRS to be neglible
        # The rest is the implementation of equation (10) from "A Closed-Form Approximation of the Gaussian
        # Noise Model in the Presence of Inter-Channel Stimulated Raman Scattering, Daniel Semrau"
        # Check notes from 7/11/19
        A = (gamma ** 2) / (bandwidth ** 2)
        B = (np.pi * (T_i ** 2 - (4 / 9))) / (alpha * phi_i)
        C = np.arcsinh((bandwidth ** 2 * phi_i) / 16 * alpha)
        D = (bandwidth ** 2) / (9 * alpha ** 2)

        n_SPM = (16 / 27) * A * (B * C + D)
        return n_SPM

    def get_XPM_n(self, COI, edge, gamma, bandwidth, alpha, D):
        """
        This method calculates the cross phase modulation (XPM) for a given COI and edge in a graph with other
        interfering channels. It does it for variably loaded channels.

        :param COI: Channel of interest (integer - corresponds to the wavelengths used for channel)
        :param edge: Link of graph that is being calculated for
        :param graph: graph that is being considered (nx.Graph())
        :param gamma: non-linear-coefficient
        :param bandwidth: bandwidth of channel (32Gbaud channels - same bandwidth)
        :param C_r: slope of the linear regression of the normalized Raman gain spectrum
        :param alpha: attenuation coefficient
        :param launch_power: power at which channel is launched (dBm)
        :param beta_2: GVD parameter
        :param beta_3: slope of GVD parameter
        :return: XPM coefficient
        :rtype: float
        """
        graph = self.graph
        lambda_ref = 1550e-9
        XPM_contr = 0  # Initialise the XPM contribution as 0
        S = D / (35 * 10 ** (-9))
        frequency = lambda wavelength: (3 * 10 ** 8) / (wavelength)  # function to return the frequency of a wavelength

        wavelengths = graph[edge[0]][edge[1]]["wavelengths"]  # get the wavelengths from the link of interest
        interfering_wavelengths = wavelengths.copy()
        interfering_wavelengths.remove(COI)  # remove the COI to get purely the interfering wavelengths
        interfering_wavelengths_physical = list(map(lambda x: self.wavelengths_physical[x],
                                                    interfering_wavelengths))  # assign physical wavelengths to the interfering wavelengths

        COI_wavelength = self.wavelengths_physical[COI]  # assign the physical wavelength to the channel of interest
        interfering_frequency = list(map(lambda x: frequency(x),
                                         interfering_wavelengths_physical))  # assign frequencies to the interfering wavelengths
        COI_frequency = frequency(COI_wavelength)  # assign frequency to the COI wavelength
        # The rest is the implementation of equation (11) from "A Closed-Form Approximation of the Gaussian
        # Noise Model in the Presence of Inter-Channel Stimulated Raman Scattering, Daniel Semrau"
        # Check notes from 7/11/19
        P_tot = sum(graph[edge[0]][edge[1]]["launch_powers"])
        logging.debug("wavelengths physical: {}".format(interfering_wavelengths_physical))
        launch_power_interfering = list(
            map(lambda x: graph[edge[0]][edge[1]]["launch_powers"][x],
                interfering_wavelengths))  # getting the launch power for each interfering wavelength
        launch_power_COI = graph[edge[0]][edge[1]]["launch_powers"][COI]
        alpha_bar = alpha
        C_r = 0  # 0.028 *10**9
        S = 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9

        for k in range(len(interfering_wavelengths)):
            T_k = 2 - (interfering_frequency[k] * P_tot * C_r) / (
                alpha)  # C_r is 0 for C-band with assumption of ISRS being neglible
            T_k = (alpha + alpha_bar - interfering_frequency[k] * P_tot * 0) ** 2
            beta_2 = abs((-D * (
                lambda_ref) ** 2) / (
                                 2 * np.pi * 3 * 10 ** 8))  # Calculate GVD dispersion parameter
            # beta_3 = (S-((4*np.pi*3*10**8)/COI_wavelength**3)*beta_2)/((2*np.pi*3*10**8)/COI_wavelength**2)
            beta_3 = lambda_ref ** 2 / (2 * np.pi * (3 * 10 ** 8)) ** 2 * (
                    lambda_ref ** 2 * S + 2 * lambda_ref * D)
            # beta_3 = (S - ((4 * np.pi * 3 * 10 ** 8) / COI_wavelength ** 3) * beta_2) / (
            #       (2 * np.pi * 3 * 10 ** 8) / COI_wavelength ** 2)  # Calculate GVD slope
            phi_i_k = 2 * np.pi ** 2 * (interfering_frequency[k] - COI_frequency) * (
                    beta_2 + np.pi * beta_3 * (COI_frequency + interfering_frequency[k]))

            B = (gamma ** 2 / (bandwidth * phi_i_k * alpha_bar * (
                    2 * alpha + alpha_bar)))  # ((launch_power_interfering[k] / launch_power_COI)
            C = (T_k - alpha ** 2) / alpha
            D = np.arctan((bandwidth * phi_i_k) / alpha)
            E = (((alpha + alpha_bar) ** 2 - T_k)) / (alpha + alpha_bar)
            F = np.arctan((bandwidth * phi_i_k) / (alpha + alpha_bar))
            XPM_interferer = abs(B * (C * D + E * F))
            XPM_contr += XPM_interferer
            logging.debug("XPM interferer: {}".format(XPM_interferer))

            logging.debug(
                "XPM : {} B_2: {} phi_i_k:{} B: {} C: {} D: {} E: {} F: {}".format(XPM_contr, beta_3, phi_i_k, B, C, D,
                                                                                   E, F))
        n_XPM = (32 / 27) * XPM_contr
        logging.debug("XPM: {}".format(n_XPM))
        return n_XPM

    def get_non_linear_coefficient(self, edge, gamma=1.2 / 1e3, bandwidth=32 * 10 ** 9,
                                   alpha=0.2 / 4.343 / 1e3, D=17 * 1e-12 / 1e-9 / 1e3,
                                   coherence_factor=0, Plot=False, name="ISRS", channels_full =156):
        """
        This method calculates the SPM and XPM coefficients and returns the total non-linear coefficient.

        :param D: Dispersion parameter
        :param spans: amount of spans the link has
        :param COI: Channel of interest (integer - corresponds to the wavelengths used for channel)
        :param edge: Link of graph that is being calculated for
        :param graph: graph that is being considered (nx.Graph())
        :param gamma: non-linear-coefficient
        :param bandwidth: bandwidth of channel (32Gbaud channels - same bandwidth)
        :param C_r: slope of the linear regression of the normalized Raman gain spectrum
        :param alpha: attenuation coefficient
        :param beta_2: GVD parameter
        :param beta_3: slope of GVD parameter
        :param coherence_factor: (...)
        :return: non-linear coefficient
        :rtype: float
        """
        graph = self.graph
        wavelengths = graph[edge[0]][edge[1]]["wavelengths"]
        wavelengths = np.asarray(wavelengths)
        self.spacing = bandwidth
        self.channels_full = channels_full
        self.channels = len(wavelengths)
        self.n = int(graph[edge[0]][edge[1]]["weight"])
        if self.n ==0:
            self.n = 1
        self.launch_powers_shape = 10 ** (0 / 10) * 0.001 * np.ones([self.channels, self.n])
        logging.debug("shape of wavelengths: {}".format(wavelengths))

        self.launch_powers = np.asarray(list(map(lambda x: graph[edge[0]][edge[1]]["launch_powers"][x], wavelengths)))
        #print("launch_powers_used: {}".format(self.launch_powers))
        self.launch_powers = np.tile(self.launch_powers, (self.n, 1)).transpose()
        logging.debug("launch power shape: {} original: {}".format(np.shape(self.launch_powers), np.shape(self.launch_powers_shape)))
        self.channel_parameters = {
            'fi': np.repeat(np.reshape(
                (wavelengths - (self.channels_full - 1) / 2) * self.spacing
                , [-1, 1]), self.n, axis=1),  # center frequencies of WDM channels (relative to reference frequency)
            'n': self.n,  # number of spans
            'Bch': np.tile(self.channel_bandwidth, [self.channels, self.n]),  # channel bandwith
            'RefLambda': 1550e-9,  # reference wavelength
            'D': 17 * 1e-12 / 1e-9 / 1e3 * np.ones(self.n),  # dispersion coefficient      (same) for each span
            'S': 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9 * np.ones(self.n),
            # dispersion slope            (same) for each span
            'Att': 0.2 / 4.343 / 1e3 * np.ones([self.channels, self.n]),
            # attenuation coefficient     (same) for each channel and span
            'Cr': 0 * np.ones([self.channels, self.n]),  # 0.028 / 1e3 / 1e12 * np.ones([self.channels, self.n]),
            # Raman gain spectrum slope   (same) for each channel and span
            'gamma': 1.2 / 1e3 * np.ones(self.n),  # nonlinearity coefficient    (same) for each span
            'Length': 80 * 1e3 * np.ones(self.n),  # fiber length                (same) for each span
            'coherent': 1  # NLI is added coherently across multiple spansP_tot
        }
        self.channel_parameters['Att_bar'] = self.channel_parameters['Att']
        P_NLI, eta = ISRSGNmodel(Att=self.channel_parameters["Att"], Att_bar=self.channel_parameters["Att_bar"],
                                 Cr=self.channel_parameters["Cr"], Pch=self.launch_powers,
                                 fi=self.channel_parameters["fi"], Bch=self.channel_parameters["Bch"],
                                 Length=self.channel_parameters["Length"], D=self.channel_parameters["D"],
                                 S=self.channel_parameters["S"],
                                 gamma=self.channel_parameters["gamma"], RefLambda=self.channel_parameters["RefLambda"])
        logging.debug("eta_1_span: {}".format(P_NLI))
        # SPM_n = self.get_SPM_n(COI, gamma, bandwidth, alpha, D)
        # XPM_n = self.get_XPM_n(COI, edge, graph, gamma, bandwidth, alpha, D)
        # logging.info("spans: {}".format(spans["weight"]))
        # non_linear_coefficient = SPM_n * spans["weight"] ** (1 + coherence_factor) + XPM_n * spans["weight"]
        # logging.info("SPM: {} XPM: {} dBm eta: {} dB".format(10 * np.log(SPM_n) / np.log10(10),
        #                                                    10 * np.log10(XPM_n) / np.log10(10),
        #                                                    10 * np.log10(non_linear_coefficient) / np.log10(10)))
        #Plotting.plot_bar_channels(y=Plotting.concatenate_empty_spectrum(P_NLI, edge, graph), file_name="P_NLI_{}_{}".format(edge[0],edge[1]))
        if Plot:
            frequencies = np.arange(channels_full)
            for i in range(channels_full):
                if frequencies[i] not in wavelengths:
                    frequencies[i]=0
                elif frequencies[i] in wavelengths:
                    frequencies[i]= eta[np.where(wavelengths==i)]
            logging.debug(frequencies)

            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
            plt.rcParams['text.usetex'] = True

            plt.title('The NLI coefficient for all channels.')
            plt.xlabel(r'Channel Number $f_i$')
            plt.ylabel(r'NLI coefficient $\eta_1$ $\bigg[\mathrm{dB}\big(\frac{1}{\mathrm{W}^2}\big)\bigg]$')
            logging.debug("len(cp): {} len(eta): {}".format(len(self.channel_parameters['fi']), len(eta)))
            plt.bar(np.arange(channels_full), todB(frequencies))
            plt.savefig("{}_{}_{}.png".format(name, edge[0], edge[1]))
            plt.close()
        return P_NLI, eta

    def get_SNR_shortest_path_node_pair(self, channels, unique_paths):
        NSR_path_sum = []
        SNR = []
        for node_pair in unique_paths:
            edges = self.nodes_to_edges(node_pair[2])
            NSR_sum = 0
            for edge in edges:
                NSR_sum += self.graph[edge[0]][edge[1]]["NSR"][int(channels / 2)]
            SNR.append((node_pair[0], node_pair[1], 1 / NSR_sum))
        return SNR

    def get_SNR_k_SP(self, channels, k_SP):
        """
        Method to get the SNR of a list in form [(s,d, [paths])] and return in form of
        [(s,d, [SNR list])].
        :param channels:    Amount of channels to use. - int
        :param k_SP:        k shortest paths to use - list((s, d, [paths]))
        :return:            list of SNR values in same form as k_SP
        :rtype:             list
        """
        import copy
        SNR_list = copy.deepcopy(k_SP)
        for i, node_pair in enumerate(k_SP):
            # print(node_pair)
            for k, path in enumerate(node_pair[1]):
                edges = self.nodes_to_edges(path)
                NSR_sum = 0
                for edge in edges:
                    NSR_sum += self.graph[edge[0]][edge[1]]["NSR"][int(channels / 2)]
                SNR_list[i][1][k] = 1/NSR_sum

        return SNR_list

    def get_SNR_uniform(self, edge, gamma=1.2, channel_bandwidth=32 * 10 ** 9, C_r=0, alpha=0.2, D=18, Plot=False, name="test"):
        """

        :param edge:
        :param graph:
        :param gamma:
        :param channel_bandwidth:
        :param C_r:
        :param alpha:
        :param D:
        :param Plot:
        :param name:
        :return:
        """
        graph = self.graph
        wavelengths = np.arange(156)
        P_ase = list(map(lambda x: self.get_P_ase(x), wavelengths))
        P_NLI, non_linear_coefficient = self.get_non_linear_coefficient(edge, graph)
        NSR = list(map(lambda x: 1 / (graph[edge[0]][edge[1]]["launch_powers"][wavelengths.index(x)] / (
                P_ase[wavelengths.index(x)] + P_NLI[wavelengths.index(x)])), wavelengths))

    def get_SNR_P_ase(self, edge, gamma=1.2, channel_bandwidth=32 * 10 ** 9, C_r=0, alpha=0.2, D=18, Plot=True, name="test"):
        graph = self.graph
        wavelengths = graph[edge[0]][edge[1]]["wavelengths"]
        wavelengths_physical = list(map(lambda x: self.wavelengths_physical[x], wavelengths))
        P_ase = list(map(lambda x: self.get_P_ase(3e8/x, graph[edge[0]][edge[1]]["weight"]), wavelengths_physical))
        NSR = list(map(lambda x: 1 / (graph[edge[0]][edge[1]]["launch_powers"][x] / (
                P_ase[wavelengths.index(x)])), wavelengths))
        return NSR

    def get_SNR_non_linear(self, edge, gamma=1.2, channel_bandwidth=32 * 10 ** 9, C_r=0, alpha=0.2, D=18, Plot=False, name="test", channels_full = 156):
        """
        This method calculates the SNR for a given channel on a given link in a given graph.

        :param Plot:
        :param COI: Channel of interest
        :param edge: link in graph
        :param graph: graph to use
        :param gamma: non-linear coefficient
        :param channel_bandwidth: bandwidth of channel (32 Gbaud)
        :return: SNR value
        :rtype: float
        """
        graph = self.graph
        wavelengths = graph[edge[0]][edge[1]]["wavelengths"]
        wavelengths_physical = list(map(lambda x: self.wavelengths_physical[x], wavelengths))
        P_ase = list(map(lambda x: self.get_P_ase(3e8/x, graph[edge[0]][edge[1]]["weight"]), wavelengths_physical))
        P_NLI, non_linear_coefficient = self.get_non_linear_coefficient(edge, graph, Plot=Plot, bandwidth=channel_bandwidth, channels_full=channels_full)
        logging.debug("launch_powers: {}".format(len(graph[edge[0]][edge[1]]["launch_powers"])))
        logging.debug("wavelengths: {}".format(wavelengths))
        NSR = list(map(lambda x: 1 / (graph[edge[0]][edge[1]]["launch_powers"][x] / (
                P_ase[wavelengths.index(x)] + P_NLI[wavelengths.index(x)])), wavelengths))
        #SNR = list(map(lambda x: (graph[edge[0]][edge[1]]["launch_powers"][x] / (
          #      P_ase[wavelengths.index(x)] + P_NLI[wavelengths.index(x)])), wavelengths))

        #print("launch powers: {} P_ase: {} P _ NLI: {} ".format(graph[1][12]["launch_powers"][15:20], P_ase[15:20], P_NLI[15:20]))
        #print(todB(list(map(lambda x: 1/x, NSR))))
        logging.debug("NSR: {}".format(NSR))
        #print("plotting")
        if Plot:
            frequencies = np.arange(156)
            for i in range(156):
                if i not in wavelengths:
                    frequencies[i] = 0
                elif i in wavelengths:
                    logging.debug("frequency: {} wavelengths: {} ".format(frequencies[i], wavelengths))
                    frequencies[i] = 1/NSR[wavelengths.index(i)]
            logging.debug(frequencies)

            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
            plt.rcParams['text.usetex'] = True

            plt.title('The SNR values for all channels in link ({}, {}).'.format(edge[0], edge[1]))
            plt.xlabel(r'Channel Number $f_i$')
            plt.ylabel(r'SNR  $\eta_1$ $\bigg[\mathrm{dB}\bigg]$')

            plt.bar(np.arange(156), todB(frequencies))
            # plt.plot(self.channel_parameters['fi'] * 1e-12, todB(eta), ls='--')
            plt.savefig("{}.png".format(name))
            plt.close()

        return NSR

    def get_P_ase(self, f, L, NF=10 ** (4 / 10), h=6.62607004 * 10 ** (-34), B_ref=32 * 10 ** 9):
        """
        This method gets the amplified spontaneous emission noise.

        :param NF: Noise function
        :param g: gain
        :param h: plancs constant
        :param f: frequency of wavelength
        :param B_ref: bandwidth of channel
        :return: amplified spontaneous emission power
        :rtype: float
        """
        gain_dB = 0.2 * L
        gain_norm = 10 ** (gain_dB / 10)
        P_ase = NF * (gain_norm - 1) * h * f * B_ref
        # P_ase = NF *  h * f * B_ref
        return P_ase

    def get_non_linear_LOGON(self, beta2_n, gamma_n=1.2 * (10 ** (-3)), alpha_n=0.2 / (10 ** (3)), a_n=0.2 * 80,
                             bandwidth_WDM=32 * 10 ** 9):
        """
        This method gets the non_linear contribution via LOGON, approximating NSR for optimising launch powers.

        :param beta2_n:
        :param gamma_n:
        :param alpha_n:
        :param a_n:
        :param bandwidth_WDM:
        :return:
        """
        alpha_n = self.to_norm(alpha_n)
        L_eff_n = (1 - np.exp(-2 * alpha_n * 80000)) / (2 * alpha_n)
        A = (16 / 27)
        B = (alpha_n * (gamma_n ** 2) * (L_eff_n ** 2)) / (np.pi * beta2_n)  # 10^-3 * 10^-6 * 10^3 / 10^-26
        C = (np.pi ** 2) / (4 * alpha_n)
        D = beta2_n * bandwidth_WDM ** 2
        E = np.arcsinh(C * D)
        #print("A: {} B: {} C: {} D: {} E: {} L_eff: {}".format(A, B, C, D, E, L_eff_n))
        non_linear = A * B * E
        # non_linear = a_n * (16 / 27) * ((alpha_n * gamma_n ** 2 * L_eff_n ** 2) / (np.pi * beta2_n)) * np.arcsinh(
        #   ((np.pi ** 2) / (4 * alpha_n)) * beta2_n * bandwidth_WDM ** 2)
        return non_linear

    def assign_LOGON_launch_powers(self, edge):
        graph = self.graph
        wavelengths = graph[edge[0]][edge[1]]["wavelengths"]
        eta = graph[edge[0]][edge[1]]["eta"]
        for i in range(len(wavelengths)):
            eta_i = eta[i]
            P_ase = self.get_P_ase(f= (3e8)/(self.wavelengths_physical[wavelengths[i]]), L=graph[edge[0]][edge[1]]["weight"])
            P_opt = (P_ase/(2*eta_i))**(1/3)
            logging.debug("P_ase:{} eta_i: {} P_opt: {}".format(P_ase, eta_i, P_opt))
            graph[edge[0]][edge[1]]["launch_powers"][wavelengths[i]] = P_opt
        logging.debug("launch powers: {}".format(graph[edge[0]][edge[1]]["launch_powers"]))


    def update_launch_powers(self, D=18):
        beta_2 = lambda D, _lambda: (-D * ((10 ** (-12)) / (10 ** (-9)) / (10 ** 3)) * (_lambda * 10 ** (-9)) ** 2) / (
                2 * np.pi * 3 * 10 ** 8)
        #print("Beta2: {}".format(beta_2(D, 1550)))
        non_linear_LOGON = lambda beta_2: self.get_non_linear_LOGON(beta_2)
        #print("non_linear: {} ".format(non_linear_LOGON(beta_2(D, 1550 * 10 ** -9))))
        P_opt = lambda non_linear_LOGON, P_ase: (P_ase / (2 * non_linear_LOGON)) ** (1 / 3)
        launch_power_1550 = P_opt(non_linear_LOGON(beta_2(D, 1550)), self.get_P_ase(3 * 10 ** 8 / 1550 * 10 ** -9))
        launch_powers = list(map(lambda x: (
            P_opt(non_linear_LOGON(beta_2(D, x * 10 ** -9)), self.get_P_ase((3 * 10 ** 8) / (x * 10 ** -9)))),
                                 self.wavelengths_physical))
        self.launch_powers = launch_powers
        # print("hello")
        #print("launch power 1550: {}".format(launch_power_1550))
    def nodes_to_edges(self, nodes):
        """
        Method to convert a path into a list of edges.
        :param nodes: path to convert - [path]
        :return: list of edges
        :rtype: [(edge), ..., (edge)]
        """
        # print(nodes)
        # edges = list(
        #  map(lambda x: (nodes[nodes.index(x)], nodes[nodes.index(x) + 1]) if nodes.index(x) < len(nodes) - 1 else 0,
        #      nodes))
        # edges.pop()
        # print(edges)
        edges = []
        for i in range(0, len(nodes) - 1):
            edges.append((nodes[i], nodes[i + 1]))
        return edges
    def add_wavelengths_to_links(self, wavelengths_dict):
        """

        :return:
        """
        graph = self.graph
        wavelengths = {}
        graph_edges = graph.edges()
        logging.debug(graph_edges)
        logging.debug(wavelengths_dict)
        for edge in graph_edges:
            wavelengths[edge] = {"wavelengths": []}
            wavelengths[(edge[1], edge[0])] = {"wavelengths": []}

        for key in wavelengths_dict:
            for path in wavelengths_dict[key]:
                edges = self.nodes_to_edges(path)
                for edge in edges:
                    wavelengths[edge]["wavelengths"].append(key)
                    wavelengths[(edge[1], edge[0])]["wavelengths"].append(key)
        nx.set_edge_attributes(graph, wavelengths)
        # logging.info(self.RWA_graph[1][8]["wavelengths"])
        return graph

    def add_wavelengths_full_occupation(self, channels_full):
        graph = self.graph
        wavelengths = {}
        graph_edges = graph.edges()
        for edge in graph_edges:
            wavelengths[edge] = {"wavelengths": list(range(channels_full))}
            wavelengths[(edge[1], edge[0])] = {"wavelengths": list(range(channels_full))}
        nx.set_edge_attributes(graph, wavelengths)
        return graph


    def add_uniform_launch_power_to_links(self, channels):
        """

        :param channels:
        :return:
        """
        graph = self.graph
        launch_powers = {}
        graph_edges = graph.edges()
        for edge in graph_edges:
            launch_powers[edge] = {"launch_powers": [(10 ** (-3))] * channels}
            launch_powers[(edge[1], edge[0])] = {"launch_powers": [(10 ** (-3))] * channels}
        logging.debug(launch_powers)
        nx.set_edge_attributes(graph, launch_powers)
        return graph

    def add_non_linear_to_links(self):
        """

        :return:
        """
        graph = self.graph
        # add non linear coefficients to all channels on all links
        eta_n = {}
        graph_edges = graph.edges()
        for edge in graph_edges:
            eta_n[edge] = {"eta": self.get_non_linear_coefficient(edge, graph, Plot=False)[1]}
            eta_n[(edge[1], edge[0])] = {"eta": self.get_non_linear_coefficient(edge, graph, Plot=False)[1]}
        nx.set_edge_attributes(graph, eta_n)
        return graph

    def add_LOGON_launch_power_to_links(self, channels, Plot=False):
        """

        :param channels:
        :param Plot:
        :return:
        """
        graph = self.graph
        # add LOGON optimised launch powers to links
        # add uniform launch power
        # add NLI coefficients
        # add LOGON launch power
        graph = self.add_non_linear_to_links(graph)
        edges = graph.edges()
        for edge in edges:
            self.assign_LOGON_launch_powers(edge, graph)
            # Plotting.plot_bar_channels(self.RWA_graph[edge[0]][edge[1]]["launch_powers"], file_name="laucch_power_{}_{}".format(edge[0], edge[1]))
        return graph

    def add_non_linear_NSR_to_links(self, Plot=False, channels_full=156, channel_bandwidth=32e9):
        """

        :param Plot:
        :return:
        """
        graph = self.graph
        NSR = {}
        graph_edges = graph.edges()
        for edge in graph_edges:
            NSR[edge] = {
                "NSR": self.get_SNR_non_linear(edge, graph, channels_full=channels_full,
                                               channel_bandwidth=channel_bandwidth,
                                               name="SNR_{}_{}".format(edge[0], edge[1]))}
            NSR[(edge[1], edge[0])] = {"NSR": self.get_SNR_non_linear(edge, graph, channels_full=channels_full,
                                                                      channel_bandwidth=channel_bandwidth)}
            nx.set_edge_attributes(graph, NSR)

            if Plot:
                import NetworkToolkit.Plotting as Plotting
                # print(self.RWA_graph[edge[0]][edge[1]]["NSR"])
                SNR = list(map(lambda x: 1 / x, graph[edge[0]][edge[1]]["NSR"]))
                Plotting.plot_bar_channels(Plotting.concatenate_empty_spectrum(SNR, edge, graph),
                                           file_name="Figures/ACMN1_1/SNR_{}_{}".format(edge[0], edge[1]),
                                           variable_x="f_i", title="SNR Spectrum", x_axes_unit="Frequency Channel",
                                           y_axes_unit="dB", x_axes="Channel", y_axes="SNR")
        return graph

    def add__P_ase_NSR_to_links(self, Plot=False):
        graph = self.graph
        NSR = {}
        graph_edges = graph.edges()
        for edge in graph_edges:
            NSR[edge] = {
                "NSR": self.get_SNR_P_ase(edge, graph, Plot=Plot, name="SNR_{}_{}".format(edge[0], edge[1]))}
            NSR[(edge[1], edge[1])] = {"NSR": self.get_SNR_P_ase(edge, graph)}
            nx.set_edge_attributes(graph, NSR)
            if Plot:
                import NetworkToolkit.Plotting as Plotting
                # print(self.RWA_graph[edge[0]][edge[1]]["NSR"])
                SNR = list(map(lambda x: 1 / x, graph[edge[0]][edge[1]]["NSR"]))
                Plotting.plot_bar_channels(Plotting.concatenate_empty_spectrum(SNR, edge, graph),
                                           file_name="Figures/ACMN1_1/SNR_{}_{}".format(edge[0], edge[1]),
                                           variable_x="f_i",
                                           title="SNR Spectrum", x_axes_unit="Frequency Channel", y_axes_unit="dB",
                                           x_axes="Channel", y_axes="SNR")
        return graph

    def get_lightpath_capacities_PLI(self, wavelengths_dict):
        """

        :return:
        """
        node_pair_capacities = {}
        graph = self.graph
        channel_bandwidth =self.channel_bandwidth
        capacity_total = 0
        n = 0
        capacity_matrix = [[0 for j in graph.nodes()] for i in graph.nodes()]
        for key in wavelengths_dict:
            logging.debug(key)
            for path in wavelengths_dict[key]:

                n += 1
                edges = self.nodes_to_edges(path)
                logging.debug("edges: {}".format(edges))
                # logging.info("key: {} {}".format(key, len(self.RWA_graph[4][7]["NSR"])))
                NSR = 0
                for edge in edges:
                    index = graph[edge[0]][edge[1]]["wavelengths"].index(key)
                    NSR += graph[edge[0]][edge[1]]["NSR"][index]
                SNR = 1 / NSR
                logging.debug("SNR:{}".format(10 * np.log10(SNR)))
                capacity = 2 * channel_bandwidth * np.log2(1 + SNR)
                # adding capacities to the node-pair dict
                if (path[0], path[-1]) in node_pair_capacities.keys():
                    node_pair_capacities[(path[0], path[-1])] += capacity
                else:
                    node_pair_capacities[(path[0], path[-1])] = capacity
                # print(np.log2(1 + SNR))
                capacity_total += capacity
                capacity_matrix[path[0] - 1][path[-1] - 1] += capacity
                logging.debug("Capacity: {}".format(capacity / 1e9))
                logging.debug("Capacity Total: {}".format(capacity_total / 1e12))
        #  print(capacity_total / 1e12)
        # print(n)
        #  print(self.N_lambda)
        # print(len(self.lightpath_routes_consecutive_single))

        capacity_average = capacity_total / n
        return capacity_total, capacity_average, node_pair_capacities

    def get_lightpath_capacities_no_PLI(self, wavelength_dic):
        channel_bandwidth = self.channel_bandwidth
        graph = self.graph
        capacity_total = 0
        edges = graph.edges()
        for wavelength in wavelength_dic.keys():
            for path in wavelength_dic[wavelength]:
                capacity_total += 2*channel_bandwidth
        # for edge in edges:
        #     capacity_total += graph[edge[0]][edge[1]]["congestion"] * 2 * channel_bandwidth
        return capacity_total

    def add_congestion_to_links(self, wavelength_dic):
        for wavelength in wavelength_dic.keys():
            for path in wavelength_dic[wavelength]:
                add_congestion(self.graph, path)



if __name__ == "__main__":
    physical_model = PhysicalLayer()

    topology = Topology.Topology("nsf")
    topology.init_nsf()

    # graph = topology.create_ACMN(14, 21, 0.31, "test")
    # physical_model.add_SNR_links(graph)
    # physical_model.calculate_capacity_lightpath(graph, [1, 2, 3, 4], 2000)
    # physical_model.add_SNR_links(graph)
    physical_model.assign_physical_wavelengths()
    physical_model.update_launch_powers()

""" Approach for different traffics and calculating the SNR for that
    def SPM(self, phi_i, T_i, B_i, a, a_bar, gamma):
        SPM = 4 / 9 * gamma ** (2) / B_i ** (2) * np.pi / (phi_i * a_bar * (2 * a + a_bar)) * ((
                T_i - a ** (2) / a * np.arcsinh(phi_i * B_i ** (2) / a / np.pi) + (
                (a + a_bar) ** (2) - T_i / (a + a_bar) * np.arcsinh(
            np.divide(phi_i * B_i ** (2), (a + a_bar)) / np.pi))))

    def XPM(self, Pi, Pk, phi_ik, T_k, B_i, B_k, a, a_bar, gamma):
        XPM = 32 / 27 * np.sum((Pk / Pi) ** 2 * gamma ** 2 / (B_k * phi_ik * a_bar * (2 * a + a_bar)) * (
                (T_k - a ** 2) / a * np.arctan(phi_ik * B_i / a) + ((a + a_bar) ** 2 - T_k) / (
                a + a_bar) * np.arctan(phi_ik * B_i / (a + a_bar))))

    def ISRSGNmodel(self):

        for j in range(0, len(n)):
            for i in range(0, len(fi)):
                (self.SPM(), self.XPM())
"""
