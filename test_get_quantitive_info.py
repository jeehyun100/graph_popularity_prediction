import networkx as nx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import collections
from math import factorial, exp
import numpy as np
import time
from utils.graph_utils import GraphUtils
import os


def main():
    file_edge_path = "./data/edge_list.txt"
    file_node_path = "./data/class_info.txt"
    base_dir = "./plot/"
    g, train_node, test_node = GraphUtils().make_graph(file_edge_path, file_node_path)
    GraphUtils().print_all_graph_quantitatively(g)
    GraphUtils().print_quantitatively_plt(g, base_dir)
    GraphUtils().get_centrality(g)

if __name__ == "__main__":
    if not os.path.isdir("plot"):
        os.makedirs("plot")
    main()