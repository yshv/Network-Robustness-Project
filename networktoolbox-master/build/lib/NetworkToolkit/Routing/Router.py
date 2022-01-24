import logging
import math
import random

import networkx as nx
import numpy as np
import pandas as pd

from .Heuristics import Heuristics
from .ML import ML
from .ILP import ILP
logging.basicConfig(level=26)

class RWA(ML, ILP, Heuristics):
    """class that can take in a graph and store RWA methods for this graph.

    :param graph: an input graph generated from Topology
    """

    def __init__(self, graph, channels, channel_bandwidth):
        Heuristics.__init__(self,graph, channels, channel_bandwidth)
        ILP.__init__(self,graph, channels, channel_bandwidth)
        ML.__init__(self,graph, channels, channel_bandwidth)
    