

import matplotlib.pyplot as plt
import numpy as np
from NetworkToolkit.ISRSGNmodel import todB
import networkx as nx
import NetworkToolkit.Tools
import matplotlib.style as style


def concatenate_empty_spectrum(y, edge, graph, channels=156):
    """

    :param y:
    :param edge:
    :param graph:
    :param channels:
    :return:
    """
    frequencies = list(range(156))
    wavelengths = graph[edge[0]][edge[1]]["wavelengths"]
    for i in range(channels):
        if i not in wavelengths:
            frequencies[i] = 0
        elif i in wavelengths:
            #print(y)
            #print(wavelengths)
            #print(y[wavelengths.index(i)])
            frequencies[i] = y[wavelengths.index(i)]
    return frequencies

def plot_bar_channels(y, file_name=None, variable_x=None, title=None, x_axes_unit=None, y_axes_unit=None, x_axes=None, y_axes=None, channels=156):
    """

    :param y:
    :param file_name:
    :param variable_x:
    :param title:
    :param x_axes_unit:
    :param y_axes_unit:
    :param x_axes:
    :param y_axes:
    :param channels:
    :return:
    """
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    plt.rcParams['text.usetex'] = True

    plt.title('{}'.format(title))
    plt.xlabel(r'{}${}$'.format(x_axes, variable_x))
    plt.ylabel(r'SNR  $\bigg[\mathrm{dB}\bigg]$')

    plt.bar(np.arange(channels), todB(y), width=1)
    # plt.bar(np.arange(156), graph[edge[0]][edge[1]]["launch_powers"])
    # plt.plot(self.channel_parameters['fi'] * 1e-12, todB(eta), ls='--')
    plt.savefig("{}.png".format(file_name))
    plt.close()

def plot_channel_link_occupation(graph, file_name="test_plot_event", x_label="Channel ID", y_label="Link ID", Title="Link and Channel Occupation"):
    """

    :param graph:
    :param file_name:
    :param x_label:
    :param y_label:
    :param Title:
    :return:
    """
    edges = graph.edges()
    data = []
    for edge in edges:
        data.append(np.asarray(graph[edge[0]][edge[1]]["wavelengths"]))
    #print(data)
    #data = np.random.gamma(4, size=[60, 50])
    #print(data)
    color = "#940707"
    lineoffsets2 = 1
    linelengths2 = 1
    linewidth = 3
    #fig, axs = plt.plot()
    plt.eventplot(np.asarray(data), colors=color, lineoffsets=lineoffsets2,
                        linelengths=linelengths2, linewidths=linewidth, orientation='horizontal')
    plt.xlabel("{}".format(x_label))
    plt.ylabel("{}".format(y_label))
    plt.title("{}".format(Title))
    plt.show()
    plt.savefig("{}.png".format(file_name))

def plot_graph_pygraphviz(graph, file_name):
    """

    :param graph:
    :param file_name:
    :return:
    """
    edges = graph.edges()
    a_graph = nx.nx_agraph.to_agraph(graph)
    for edge in edges:
        e = a_graph.get_edge(edge[0], edge[1])
        congestion = graph[edge[0]][edge[1]]["congestion"]
        SNR = round(10*np.log10(1/graph[edge[0]][edge[1]]["NSR"][0]),2)
        launch_power = round(10*np.log10(graph[edge[0]][edge[1]]["launch_powers"][0]/1e-3),2)
        distance = graph[edge[0]][edge[1]]["weight"]
        congestion_factor = congestion/156*16.0
        if congestion > 110:
            e.attr['label'] = 'cong: {} \nP: {}dBm\n SNR: {}dB \nspans: {}'.format(congestion, launch_power, SNR,distance)
            e.attr['color'] = 'red'
            e.attr['penwidth'] = congestion_factor
        elif congestion >= 60 and congestion <= 110:
            e.attr['label'] = 'cong: {} \nP: {}dBm\n SNR: {}dB \nspans: {}'.format(congestion, launch_power, SNR,distance)
            e.attr['color'] = 'darkorange'
            e.attr['penwidth'] = congestion_factor
        else:
            e.attr['label'] = 'cong: {} \nP: {}dBm\n SNR: {}dB \nspans: {}'.format(congestion, launch_power, SNR,distance)
            e.attr['color'] = 'green'
            e.attr['penwidth'] = congestion_factor

    a_graph.draw("{}.png".format(file_name), prog='dot')

def get_pos(graph, x="Latitude", y="Longitude"):
    pos = {}
    for node, data in graph.nodes.data():
        pos[node] = [data[x], data[y]]
    return pos

def plot_graph(graph, with_pos=False, x="Latitude", y="Longitude", plt_show=True, **kwargs):
    """

    :param graph:
    :return:
    """

    if with_pos:
        pos = get_pos(graph, x=y, y=x)
        # nx.draw(graph,  pos, **kwargs)
        nx.draw_networkx(graph, pos=pos, **kwargs)
    else:
        # nx.draw(graph, ax=ax, **kwargs)
        nx.draw_networkx(graph, **kwargs)
    if plt_show:
        plt.show()


def plot_histogram(x, x_label, y_label, title):
    """"""
    style.use("seaborn-dark")
    plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.title("{}".format(title))
    plt.xlabel("{}".format(x_label))
    plt.ylabel("{}".format(y_label))
    #plt.xticks(range(3, 13), fontsize=14)
    plt.hist(x,bins=20, density=True, color="#3F5D7D")
    plt.show()
    plt.savefig("{}.png".format(title))


def plot_data(x, y,labels=None,multiple = False, scatter=False, bar=False, line=False,title=None, x_label=None, y_label=None, error_bars=None, mean=None, std=None, error_bar_index=None):
    style.use("seaborn-dark")
    plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    #plt.ylim(120*1e12, 600 *1e12)
    plt.xticks(np.arange(0.21, 0.41, 0.01), fontsize=14)
    #plt.yticks(np.arange(120*1e12, 600*1e12, 30e12), fontsize=14)

    plt.title("{}".format(title))
    plt.xlabel("{}".format(x_label))
    plt.ylabel("{}".format(y_label))
    if scatter:
        if multiple:
            _i = 0
            for i, j, z in zip(x, y, labels):
                _i+=1
                if _i == error_bar_index:
                    j = sliding_mean(j, window=1)
                    plt.fill_between(i, j - std,
                                     j + std, color="#3F5D7D")
                    plt.plot(i, j, color="white", lw=2, label=z, marker="v", markerfacecolor="black")

                    pass
                else:
                    j = sliding_mean(j, window=1)
                    plt.plot(i, j, label=z, marker="^", markerfacecolor="black")
                    plt.legend()
        else:
            plt.plot(x, y)
    plt.show()
    plt.savefig("{}.png".format(title))


def sliding_mean(data_array, window=5):
    data_array = data_array
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return new_list

def plot_graph_google_earth(graph, save_path, network_name, node_size=3, link_size=3):
    import os
    import os.path as op
    from inspect import stack
    import simplekml
    import networkx as nx


    # style of a point in Google Earth
    point_style = simplekml.Style()
    point_style.labelstyle.color = simplekml.Color.white
    point_style.labelstyle.scale = 0
    point_style.iconstyle.scale = node_size
    point_style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'

    #point_style.iconstyle.icon.href = ('https://raw.githubusercontent.com/afourmy/'
    #                                   'pyNMS/master/Icons/default_router.gif')

    line_style = simplekml.Style()
    line_style.linestyle.color = simplekml.Color.red
    line_style.linestyle.width = link_size

    # associates a node name to its geodetic coordinates
    node_coords = {}

    kml = simplekml.Kml()


    for node in graph.nodes:

        try:
            point = kml.newpoint(name=str(node))
            coords = [(
                float(graph.nodes.data()[node]['Longitude']),
                float(graph.nodes.data()[node]['Latitude'])
            )]

            point.coords = coords
            node_coords[node] = coords
            point.style = point_style
        except KeyError:
            continue
    print("node coords: {}".format(node_coords))
    for link in graph.edges():
        print(node_coords[link[0]][0])
        try:
            name = '{} - {}'.format(*link)
            line = kml.newlinestring(name=name)
            line.coords = [
                node_coords[link[0]][0],
                node_coords[link[1]][0]
            ]
            line.style = line_style
        except KeyError:
            continue

    kml.save(op.join(save_path, '{}.kml'.format(network_name)))

if __name__ == "__main__":
    import sys
    if len(sys.argv) >1:
        if "-r1" in sys.argv:
            range_1 = [0.21+i*0.01 for i in range(int(sys.argv[sys.argv.index("-r1") + 1]))]
        if "-r2" in sys.argv:
            range_2 = range(int(sys.argv[sys.argv.index("-r2") + 1]))
        if "-p" in sys.argv:
            if sys.argv[sys.argv.index("-p")+1] == "C":
                parameter = "Capacity"
            elif sys.argv[sys.argv.index("-p")+1] == "D":
                parameter = "N_lambda"
            elif sys.argv[sys.argv.index("-p")+1] == "d":
                parameter = "RWA_density"
        if "-x" in sys.argv:
            x_label = sys.argv[sys.argv.index("-x") + 1]
        if "-y" in sys.argv:
            y_label = sys.argv[sys.argv.index("-y") + 1]
        if "-t" in sys.argv:
            title = sys.argv[sys.argv.index("-t") + 1]
        if "--plot" in sys.argv:
            plot_variables = list(sys.argv[sys.argv.index("--plot") + 1])
            if sys.argv[sys.argv.index("--plot") + 1] == "algorithms":


                data_MNH = Tools.load_network_data("ACMN_{}_node_0.6_network_data_MNH_no_weighting_LAR", "Data/4_20_NODES_0.6", range(4,14), "N_lambda")
                data_ILP = Tools.load_network_data("ACMN_{}_node_0.6_network_data_ILP", "Data/4_20_NODES_0.6", range(4,14), "N_lambda")
                data_FF_kSP = Tools.load_network_data("ACMN_{}_node_0.6_network_data_FF_kSP", "Data/4_20_NODES_0.6", range(4,14), "N_lambda")
                data_FF_static_baroni = Tools.load_network_data("ACMN_{}_node_0.6_network_data_baroni_MNH_optimised", "Data/4_20_NODES_0.6", range(4, 14), "N_lambda")
                data_FF_kSP_mean = list(map(lambda x: sum(x)/len(x),Tools.load_network_data_over_range("ACMN_13_node_{}_{}_network_data_FF_kSP", "Data/13_NODES_0.21_0.41", range_1=range_1,range_2=range_2,parameter=parameter)))
                data_baroni_mean = list(map(lambda x: sum(x)/len(x),Tools.load_network_data_over_range("ACMN_13_node_{}_{}_network_data_baroni_optimised", "Data/13_NODES_0.21_0.41", range_1=range_1,range_2=range_2,parameter=parameter)))
                #data_ILP_mean = list(map(lambda x: sum(x)/len(x),Tools.load_network_data_over_range("ACMN_13_node_{}_{}_network_data_ILP", "Data/13_NODES_0.21_0.41", range_1=range_1,range_2=range_2,parameter="N_lambda")))
                data_MNH_CA_mean = list(map(lambda x: sum(x) / len(x),
                                            Tools.load_network_data_over_range("ACMN_13_node_{}_{}_network_data_MNH_CA",
                                                                               "Data/13_NODES_0.21_0.41",
                                                                               range_1=range_1,
                                                                               range_2=range_2, parameter=parameter)))
                #data_baroni_mean = list(map(lambda x: sum(x) / len(x),
                              #              Tools.load_network_data_over_range("ACMN_13_node_{}_{}_network_data_baroni_optimised",
                                 #                                              "Data/13_NODES_0.21_0.41",
                              #                                                 range_1=[0.21 + i * 0.01 for i in range(20)],
                                   #                                            range_2=range(50), parameter="Capacity")))
                #plot_data([range(4,14), range(4,14), range(4,14), range(4,14)], [data_MNH, data_ILP, data_FF_kSP, data_FF_static_baroni], labels=["kSP-CA-FF", "ILP", "FF-kSP", "Baroni"], multiple=True, scatter=True, title="Demand vs Number of Nodes in ACMN alpha=0.6", x_label="Number of Nodes", y_label="Number of uniform node-pair connections")
                variance_FF = np.std(np.array(Tools.load_network_data_over_range("ACMN_13_node_{}_{}_network_data_FF_kSP", "Data/13_NODES_0.21_0.41", range_1=range_1,range_2=range_2,parameter=parameter)), axis=1)
                data_FF_plus, data_FF_minus = (data_FF_kSP_mean+variance_FF), (data_FF_kSP_mean-variance_FF)
                plot_data([range_1 for j in range(3)], [data_FF_kSP_mean, data_MNH_CA_mean, data_baroni_mean], labels=["FF-kSP", "kSP-CA-FF", "baroni"], multiple=True, scatter=True, x_label=x_label, y_label=y_label, title=title, error_bar_index=1, std=variance_FF, error_bars=True)
                print("variance of FF_kSP: {}".format(variance_FF))
                print("FF_kSP: {}".format(data_FF_kSP_mean))
                print("MNH_CA: {}".format(data_MNH_CA_mean))
                print("Baroni: {}".format(data_baroni_mean))

               # print("ILP: {}".format(data_ILP_mean))
            elif sys.argv[sys.argv.index("--plot")+1] == "baroni_hist":
                data_023 = Tools.load_data(name="0.23_baroni_results", location="Data")
                data_029 = Tools.load_data(name="0.29_baroni_results", location="Data")
                data_035 = Tools.load_data(name="0.35_baroni_results", location="Data")
                data_041 = Tools.load_data(name="0.41_baroni_results", location="Data")
                plot_histogram(data_023["ILP_WR"].tolist(), x_label=x_label, y_label=y_label, title=title.format(0.23))
                plot_histogram(data_029["ILP_WR"].tolist(), x_label=x_label, y_label=y_label, title=title.format(0.29))
                plot_histogram(data_035["ILP_WR"].tolist(), x_label=x_label, y_label=y_label, title=title.format(0.35))
                plot_histogram(data_041["ILP_WR"].tolist(), x_label=x_label, y_label=y_label, title=title.format(0.41))


