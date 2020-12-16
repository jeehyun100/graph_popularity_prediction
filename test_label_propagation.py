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
import pandas as pd

from utils.graph_utils import GraphUtils


def save_graphml(graph, base_dir, best_score):
    #num_communities = len(set(nx.get_node_attributes(graph, 'community').values()))
    nx.write_graphml(graph, base_dir + "label_propergation_{0}.graphml".format(best_score))


def save_layout_pos(G):
    pos = nx.spring_layout(G)
    file_name = save_obj(pos, "spring_layout_pos" )
    return file_name


def save_obj(obj, name ):
    with open('./obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return './obj/'+ name + '.pkl'


def load_obj(file_name_w_path ):
    with open(file_name_w_path, 'rb') as f:
        return pickle.load(f)


def asyn_lpa_communities(G, pos_file, limit_epoch, chk_dir, base_dir, weight=None ):
    """Returns communities in `G` as detected by asynchronous label
    propagation.

    The asynchronous label propagation algorithm is described in
    [1]_. The algorithm is probabilistic and the found communities may
    vary on different executions.

    The algorithm proceeds as follows. After initializing each node with
    a unique label, the algorithm repeatedly sets the label of a node to
    be the label that appears most frequently among that nodes
    neighbors. The algorithm halts when each node has the label that
    appears most frequently among its neighbors. The algorithm is
    asynchronous because each node is updated without waiting for
    updates on the remaining nodes.

    This generalized version of the algorithm in [1]_ accepts edge
    weights.

    Parameters
    ----------
    G : Graph

    weight : string
        The edge attribute representing the weight of an edge.
        If None, each edge is assumed to have weight one. In this
        algorithm, the weight of an edge is used in determining the
        frequency with which a label appears among the neighbors of a
        node: a higher weight means the label appears more often.

    Returns
    -------
    communities : iterable
        Iterable of communities given as sets of nodes.

    Notes
    ------
    Edge weight attributes must be numerical.

    References
    ----------
    .. [1] Raghavan, Usha Nandini, Réka Albert, and Soundar Kumara. "Near
           linear time algorithm to detect community structures in large-scale
           networks." Physical Review E 76.3 (2007): 036106.
    """

    if pos_file is not None:
        pos = load_obj(pos_file)
    labels = {n: G._node[n]['label'] for i, n in enumerate(G)}
    all_nodes = list(G)
    node_with_zero = [n for n in all_nodes if G._node[n]['label'] == 0]
    total_cont = True
    number_loop = 0
    best_epoch = 0
    stop_point = 0
    label_dict = dict()
    cont = True

    while total_cont and cont:
        cont = False

        random.shuffle(node_with_zero)
        # Calculate the label for each node
        #end_freq = Counter()
        check_nodes = dict()
        for node in node_with_zero:
            if len(G[node]) < 1:
                continue

            # Get label frequencies. Depending on the order they are processed
            # in some nodes with be in t and others in t-1, making the
            # algorithm asynchronous.
            label_freq = Counter()

            for v in G[node]:
                value = G.nodes[v]['label']
                if labels[v] != 0:
                    label_freq.update({labels[v]: G.edges[v, node][weight]
                                        if weight else 1})
            # Choose the label with the highest frecuency. If more than 1 label
            # has the highest frecuency choose one randomly.
            if len(label_freq) != 0:
                try:
                    max_freq = max(label_freq.values())
                except Exception as e:
                    print(e)
                best_labels = [label for label, freq in label_freq.items()
                               if freq == max_freq]
                new_label = random.choice(best_labels)
                labels[node] = new_label
                # Continue until all nodes have a label that is better than other
                # neighbour labels (only one label has max_freq for each node).
                cont = cont or len(best_labels) > 1
                end_freq = Counter(labels.values())
                check_nodes[node] = cont

                #if (end_freq[0] + end_freq[1]) == 800:
        chk_flg = Counter(check_nodes.values())
        # End of node loop
        if (end_freq[0]) == 0:
            current_epoch = chk_flg[False]
            if current_epoch > best_epoch:
                # Add communities attribute into the graph
                plot_g = G.copy()
                nx.set_node_attributes(plot_g, labels, 'community')
                best_epoch = current_epoch
                # Save best_epoch label
                label_dict[best_epoch] = labels
                #nx.set_node_attributes(G, labels, 'community')
                save_graphml(G, chk_dir, best_epoch)
                print("result : {0}, epoch {1} false_cnt {2}".format(end_freq, number_loop, current_epoch))
                stop_point += 1
                GraphUtils().draw_network(plot_g, base_dir, pos, best_epoch)

                best_dict = {items[0]:items[1] for items in labels.items() if items[0] in node_with_zero}

                #best_df = pd.DataFrame({"node": best_dict.keys(), "value": best_dict.values()})
                result_file = "./result/" + "epoch_{0}_fcnt_{1}_lp_result.txt".format(number_loop, current_epoch)
                GraphUtils().save_result(best_dict, result_file)
                # saving as a CSV file
                #df.to_csv(result_dir, sep='\t', index=False, header=False)

        elif number_loop%1 == 0:
            plot_g = G.copy()
            nx.set_node_attributes(plot_g, labels, 'community')
            print("result : {0}, epoch {1}".format(end_freq, number_loop))
            GraphUtils().draw_network(plot_g, base_dir, pos, best_epoch)
            #chk_flg = Counter(check_nodes.values())
        number_loop += 1
        if number_loop > limit_epoch:
            total_cont = False
        print("cont Counter : {0} epoch {1}".format(chk_flg, number_loop))
    labels_b = label_dict[max(label_dict.keys())]
    # TODO In Python 3.3 or later, this should be `yield from ...`.
    return labels_b


def main():
    file_edge_path = "./data/edge_list.txt"
    file_node_path = "./data/class_info.txt"
    g2 = GraphUtils().preprocess_label_propagation(file_edge_path, file_node_path)

    # position file save for plotting
    #pos_file = save_layout_pos(g1)
    pos_file = "./obj/spring_layout_pos.pkl"

    """ Library """
    communities = asyn_lpa_communities(G=g2, pos_file=pos_file, limit_epoch=100, chk_dir=checkpoint_dir,
                                       base_dir=base_dir, weight=None)  # Asynchronous
    nx.set_node_attributes(g2, communities, 'community')


if __name__ == "__main__":
    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    base_dir = "./draw/"
    os.makedirs(base_dir, exist_ok=True)

    obj_dir = "./obj/"
    os.makedirs(obj_dir, exist_ok=True)

    result_dir = "./result/"
    os.makedirs(result_dir, exist_ok=True)

    main()
