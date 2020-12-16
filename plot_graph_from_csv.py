import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import Counter
from networkx.utils import groups
import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pickle

def make_graph(file_edge, file_node):
    g = nx.read_edgelist(file_edge, create_using=nx.Graph(), nodetype=int)
    #print(g)
    return g

def preprocess_node(file_edge, file_node):

    G = nx.Graph()

    with open(file_node) as f:
        nodes = {int(line.split('\t')[0].rstrip("\n")): int(line.split('\t')[1].rstrip("\n")) for line in f if line.rstrip() != ''}
        # for line in lines:
        #     self.sentences.extend(line.split('. '))
    sorted_nodes = sorted(nodes.items())
    for (id, label) in sorted_nodes:
        # if label == 0:
        #     label = ''
        G.add_node(id, label=label, community=label )

    print("graph complete")

    with open(file_edge) as f:
        edges = [(int(line.split('\t')[0].rstrip("\n")), int(line.split('\t')[1].rstrip("\n"))) for line in f if line.rstrip() != '']
        # for line in lines:
        #     self.sentences.extend(line.split('. '))
    print("Total edges {0}".format(edges))


    for (n1, n2) in edges:
        G.add_edge(n1, n2)
    #G = nx.relabel_nodes(G, nodes)
    #print(g)
    return G

def draw_network(G, base_dir, pos, base_score):
    plt.figure(figsize=(12, 8))
    # position map

    # community index
    communities = [c for c in nx.get_node_attributes(G, 'community').values()]
    communities_dict = nx.get_node_attributes(G, 'community')
    num_communities = max(communities) + 1

    # color map from http://colorbrewer2.org/
    cmap_light = colors.ListedColormap(
        ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6'], 'indexed', num_communities)
    cmap_dark = colors.ListedColormap(
        ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a'], 'indexed', num_communities)

    # edges
    nx.draw_networkx_edges(G, pos)
    node_size = 100
    # nodes
    node_collection = nx.draw_networkx_nodes(
        G, pos=pos, node_color=communities, cmap=cmap_light, node_size=node_size)

    # set node border color to the darker shade
    dark_colors = [cmap_dark(v) for v in communities]
    node_collection.set_edgecolor(dark_colors)

    # Print node labels separately instead
    for n in G.nodes:
        plt.annotate(n,
                     xy=pos[n],
                     textcoords='offset points',
                     horizontalalignment='center',
                     verticalalignment='center',
                     xytext=[0, 2],
                     # color=cmap_dark(communities[n]))
                     color=cmap_dark(communities_dict.get(n)))
    plt.title(str(Counter(communities)) + "_" + str(base_score) )
    plt.axis('off')
    # pathlib.Path("output").mkdir(exist_ok=True)
    #print("Writing network figure to output/karate.png")
    #plt.savefig(base_dir + "com_det_graph_nx_{0}_maxc_{1}.png".format(self.min_cooccurrence, num_communities))
    plt.show()

def load_obj(file_name_w_path ):
    with open(file_name_w_path, 'rb') as f:
        return pickle.load(f)

def save_graphml(graph, file_name_w_path):

    #num_communities = len(set(nx.get_node_attributes(graph, 'community').values()))
    nx.write_graphml(graph, file_name_w_path)


if __name__ == "__main__":
    file_edge_path = "./data/edge_list.txt"
    file_node_path = "./result/10_800_128_SVC_0.905.csv"


    base_dir = "./draw/"
    os.makedirs(base_dir, exist_ok=True)

    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    graphml_name = checkpoint_dir + os.path.splitext(os.path.basename(file_node_path))[0] + ".graphml"

    g1 = make_graph(file_edge_path, file_node_path)
    #pos_file = save_layout_pos(g1)
    pos_file = "./obj/spring_layout_pos.pkl"
    g2 = preprocess_node(file_edge_path, file_node_path)
    # pos = nx.spring_layout(G)
    if pos_file is not None:
        pos = load_obj(pos_file)

    draw_network(g2, base_dir, pos, 1)

    save_graphml(g2, graphml_name)

    # Detect communities label propagation
    """ Library """
    #communities = asyn_lpa_communities(G=g2, pos_file = pos_file, limit_epoch=10,  chk_dir = checkpoint_dir, base_dir = base_dir, weight=None)  # Asynchronous
    # c_iter = community.label_propagation_communities(G=G)   # Semi-synchronous

    # max_k_w = []
    # for c in c_iter:
    #     max_k_w += [c]
    #
    #
    # communities = dict()
    # for i, nodes in enumerate(max_k_w):
    #     communities.update(dict.fromkeys(nodes, i))

    print("done")

    # _communities_dict1 = dict.fromkeys(dcl[0], 0)
    # _communities_dict2 = dict.fromkeys(dcl[1], 1)
    # communities = {**_communities_dict1, **_communities_dict2 }
    #
    # # Add communities attribute into the graph
    #nx.set_node_attributes(g2, communities, 'community')
    #
    # # Draw Community detected graph
    #draw_network(g2, base_dir)
    # pgu.save_graphml(graph_uf2, base_dir, min_cooccurrence)
    #
    # # Print Community detected graph
    # pgu.print_quantitatively(graph_uf2, dcl, base_dir, idx_to_vocab)


#    if limit_epoch < stop_point:
#    total_cont = False

