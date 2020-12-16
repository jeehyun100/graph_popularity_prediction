import networkx as nx
import pandas as pd
from collections import Counter
from gensim.models import KeyedVectors
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from stellargraph import StellarGraph
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
import time
import collections
import numpy as np
import matplotlib.colors as colors


class GraphUtils():

    # def __init__(self ):
    #     #print("init")

    def make_graph(self, file_edge, file_node_path):
        G = nx.read_edgelist(file_edge, create_using=nx.Graph(), nodetype=int)

        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        with open(file_node_path) as f:
            nodes = {int(line.split('\t')[0].rstrip("\n")): int(line.split('\t')[1].rstrip("\n")) for line in f if
                     line.rstrip() != ''}

        df_node = pd.DataFrame({"node": nodes.keys(), "values": nodes.values()}, index=nodes.keys())
        train_node = df_node.loc[df_node["values"] != 0]
        test_node = df_node.loc[df_node["values"] == 0]
        return G, train_node, test_node

    def preprocess_label_propagation(self, file_edge, file_node):

        G = nx.Graph()

        with open(file_node) as f:
            nodes = {int(line.split('\t')[0].rstrip("\n")): int(line.split('\t')[1].rstrip("\n")) for line in f if
                     line.rstrip() != ''}
        sorted_nodes = sorted(nodes.items())
        for (id, label) in sorted_nodes:
            G.add_node(id, label=label)
        print("graph complete")

        with open(file_edge) as f:
            edges = [(int(line.split('\t')[0].rstrip("\n")), int(line.split('\t')[1].rstrip("\n"))) for line in f if
                     line.rstrip() != '']
        #print("Total edges {0}".format(edges))

        for (n1, n2) in edges:
            G.add_edge(n1, n2)
        return G


    def preprocessing(self, g, train_node, file_emb_output="./emb/100_900_nede2vec.emb"):

        node_subjects = train_node['values']

        node_subjects = node_subjects.astype(str)
        print(Counter(node_subjects))

        #file_emb_output = "./emb/100_900_nede2vec.emb"
        model = KeyedVectors.load_word2vec_format(file_emb_output)
        node_ids = model.wv.index2word
        node_embeddings = (
            model.wv.vectors
        )  # num
        print("Embedding load success.")

        reinex_node_embedding = pd.DataFrame(node_embeddings, index=map(int, node_ids))
        g_feature_attr = g.copy()

        G = StellarGraph.from_networkx(
            g_feature_attr, node_features=reinex_node_embedding, node_type_default="n", edge_type_default="e"
        )
        print(G.info())

        train_subjects, test_subjects = model_selection.train_test_split(
            node_subjects, train_size=160, test_size=None, stratify=node_subjects
        )
        val_subjects, test_subjects = model_selection.train_test_split(
            test_subjects, train_size=20, test_size=None, stratify=test_subjects
        )

        train_subjects.value_counts().to_frame()

        target_encoding = preprocessing.LabelBinarizer()
        # target_encoding = preprocessing.OneHotEncoder()

        train_targets = target_encoding.fit_transform(train_subjects)
        val_targets = target_encoding.transform(val_subjects)
        test_targets = target_encoding.transform(test_subjects)

        generator = FullBatchNodeGenerator(G, method="gcn")
        train_gen = generator.flow(train_subjects.index, train_targets)
        val_gen = generator.flow(val_subjects.index, val_targets)
        test_gen = generator.flow(test_subjects.index, test_targets)

        all_nodes = node_subjects.index
        all_gen = generator.flow(all_nodes)

        return G, train_gen, train_targets, val_gen, val_targets, test_targets, test_gen, all_gen, generator

    def preprocessing_predict(self, g, test_node, file_emb_output="./emb/100_900_nede2vec.emb"):

        node_subjects = test_node['values']

        node_subjects = node_subjects.astype(str)
        print(Counter(node_subjects))

        #file_emb_output = "./emb/100_900_nede2vec.emb"
        model = KeyedVectors.load_word2vec_format(file_emb_output)
        node_ids = model.wv.index2word
        node_embeddings = (
            model.wv.vectors
        )  # num
        print("Embedding load success.")

        reinex_node_embedding = pd.DataFrame(node_embeddings, index=map(int, node_ids))
        g_feature_attr = g.copy()

        G = StellarGraph.from_networkx(
            g_feature_attr, node_features=reinex_node_embedding, node_type_default="n", edge_type_default="e"
        )
        print(G.info())

        # train_subjects, test_subjects = model_selection.train_test_split(
        #     node_subjects,  stratify=node_subjects #train_size=160, test_size=None,
        # )
        # # val_subjects, test_subjects = model_selection.train_test_split(
        # #     test_subjects, train_size=20, test_size=None, stratify=test_subjects
        # # )

        #train_subjects.value_counts().to_frame()

        #target_encoding = preprocessing.LabelBinarizer()
        # target_encoding = preprocessing.OneHotEncoder()

        # train_targets = target_encoding.fit_transform(train_subjects)
        # val_targets = target_encoding.transform(val_subjects)
        # test_targets = target_encoding.transform(test_subjects)

        generator = FullBatchNodeGenerator(G, method="gcn")
        # train_gen = generator.flow(train_subjects.index, train_targets)
        # val_gen = generator.flow(val_subjects.index, val_targets)
        # test_gen = generator.flow(test_subjects.index, test_targets)

        all_nodes = node_subjects.index
        test_gen = generator.flow(all_nodes)

        return G, test_gen, generator

    def evaluate(self, model, test_gen):
        test_metrics = model.evaluate(test_gen)
        print("\nTest Set Metrics:")
        for name, val in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, val))


    def show_embedding(self, embedding_model, all_gen, train_node, model_type):
        #embedding_model = Model(inputs=x_inp, outputs=x_out)
        emb = embedding_model.predict(all_gen)
        print(emb.shape)

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        transform = TSNE  # or PCA

        X = emb.squeeze(0)
        X.shape

        trans = transform(n_components=2)
        X_reduced = trans.fit_transform(X)
        X_reduced.shape

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=train_node["values"].astype("category").cat.codes,
            cmap="jet",
            alpha=0.7,
        )
        ax.set(
            aspect="equal",
            xlabel="$X_1$",
            ylabel="$X_2$",
            title=f"{transform.__name__} visualization of {model_type} embeddings",
        )

        fig.show()

    def print_all_graph_quantitatively(self, graph):
        start_time = time.time()
        try:
            max_component = max(nx.connected_components(graph), key=len)
            diameter = nx.diameter(graph.subgraph(max_component))
        except Exception as e:
            print(e)
            diameter = 0

        print("all diameter elapse time {0:.6f}".format(time.time() - start_time))
        print("all node Diameter : {0:.6f}".format(diameter))
        print("all avg clustering {0}".format(nx.average_clustering(graph)))

        degreelist_uf2 = [val for (node, val) in graph.degree()]
        print("all Avg. Node Degree : {0:.4f}".format(float(sum(degreelist_uf2)) / nx.number_of_nodes(
            graph)))
        print("all Density : {0:.4f}".format(nx.density(graph)))
        self.print_node_edge(graph, "all")


    def print_node_edge(self, graph, graph_name):
        """
            Graph의 node와 edge를 보여준다
        """
        print("{0} number_of_nodes : {1}".format(graph_name, graph.number_of_nodes()))
        print("{0} number_of_edges : {1}".format(graph_name, graph.number_of_edges()))


    def print_quantitatively_plt(self, graph, base_dir ):
        #for i, kl in enumerate(klt):
        # print("graph number{0}, {1}".format(i, [(idx_to_vocab[i], i) for i in kl]))
        #print("######################{0}#######################".format(i))
        #subgraph = graph.subgraph(kl)
        #self.get_centrality(subgraph, "Cluster {0}".format(i), idx_to_vocab)

        degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
        sum_of_edges = sum(degree_sequence)
        num_of_nodes = nx.number_of_nodes(graph)
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())
        c = sum_of_edges / num_of_nodes
        dgs = [d for i, d in graph.degree()]
        cds = np.zeros(max(dgs) + 1)
        for d in dgs:
            cds[d] += 1

        plt.loglog(deg, np.array(cnt) / float(sum(cnt)), ".b", label="real")
        # try:
        y = [2 * (c ** 2) * d ** (-3) if d != 0 else 0 for d in deg]

        start_point = int(len(deg) * 0.3)
        plt.title("Degree Histogram graph(powlaw)")
        plt.loglog(deg[start_point:], y[start_point:], ".g", label="theoretical")
        plt.ylabel('Pk(Degree Distribution)')
        plt.xlabel('K(depgree)')
        plt.legend()

        filename = base_dir + "pow_law_graph"
        plt.tight_layout()
        plt.savefig(filename + ".png")

        degree_sequence = sorted([d for n, d in graph.degree()])  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        fig, ax = plt.subplots()
        deg_str = list(map(str, deg))
        ax.bar(deg_str, cnt, color="b")

        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        #plt.xticks(rotation=90)
        #ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticks(deg_str[::3])
        ax.set_xticklabels(deg_str[::3], rotation=45)
        #ax.set_title("Every 2nd ticks on x axis")
        #ax.set_xticklabels(deg)
        #
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
        # ax1.plot(x, y)
        # ax1.set_title("Crowded x axis")
        #
        # ax2.plot(x, y)
        # ax2.set_xticks(x[::2])
        # ax2.set_xticklabels(x[::2], rotation=45)
        # ax2.set_title("Every 2nd ticks on x axis")


        # draw graph in inset
        filename = base_dir + "degree_distribute"
        plt.tight_layout()
        plt.savefig(filename + ".png")



    def get_centrality(self, graph, graph_type="term"):
        """
            centrality를 구한다.
            degree, eigenvector, katz, pagerank, closeness betweenness centrality를 구한다.
        """
        c_dict = []

        top_15_sorted_left_eigen_centrality_g = [(n#, round(c, 4)
                                                 ) for (n, c) in
                                                sorted(nx.eigenvector_centrality(graph, tol=1000).items(),
                                                       reverse=True, key=lambda item: item[1])[:15]]
        # top_5_sorted_left_eigen_centrality_g += [(n, round(c, 4)) for (n, c) in
        #                                          sorted(nx.eigenvector_centrality(graph, tol=1000).items(),
        #                                                 reverse=True, key=lambda item: item[1])[-4:]]

        c_dict.append(
            {'grape_type': graph_type, 'type': 'left_eigen', 'centrality': top_15_sorted_left_eigen_centrality_g, })

        top_15_sorted_left_katz_centrality_g = [(n#, round(c, 4)
                                                ) for (n, c) in
                                               sorted(nx.katz_centrality(graph, alpha=0.01).items(),
                                                      reverse=True, key=lambda item: item[1])[:15]]
        # top_5_sorted_left_katz_centrality_g += [(n, round(c, 4)) for (n, c) in
        #                                         sorted(nx.katz_centrality(graph, alpha=0.01).items(),
        #                                                reverse=True, key=lambda item: item[1])[-4:]]

        c_dict.append(
            {'grape_type': graph_type, 'type': 'left_katz', 'centrality': top_15_sorted_left_katz_centrality_g, })

        top_15_in_left_pagerank_g = [(n#, round(c, 4)
                                     ) for (n, c) in
                                    sorted(nx.pagerank(graph).items(),
                                           reverse=True, key=lambda item: item[1])[:15]]
        # top_5_in_left_pagerank_g += [(n, round(c, 4)) for (n, c) in
        #                              sorted(nx.pagerank(graph).items(),
        #                                     reverse=True, key=lambda item: item[1])[-4:]]
        # c_dict.append({'grape_type': graph_type, 'type': 'page_Rank', 'centrality': top_5_in_left_pagerank_g, })

        top_15_in_closeness_g = [(n#, round(c, 4)
                                 ) for (n, c) in
                                sorted(nx.closeness_centrality(graph).items(),
                                       reverse=True, key=lambda item: item[1])[:15]]
        # top_5_in_closeness_g += [(n, round(c, 4)) for (n, c) in
        #                          sorted(nx.closeness_centrality(graph).items(),
        #                                 reverse=True, key=lambda item: item[1])[-4:]]

        c_dict.append({'grape_type': graph_type, 'type': 'in closeness', 'centrality': top_15_in_closeness_g, })

        top_15_betweenness_g = [(n#, round(c, 4)
                                ) for (n, c) in
                               sorted(nx.betweenness_centrality(graph).items(),
                                      reverse=True, key=lambda item: item[1])[:15]]
        # top_5_betweenness_g += [(n, round(c, 4)) for (n, c) in
        #                         sorted(nx.betweenness_centrality(graph).items(),
        #                                reverse=True, key=lambda item: item[1])[-4:]]

        c_dict.append({'grape_type': graph_type, 'type': 'betweenness', 'centrality': top_15_betweenness_g, })

        print("{1} Left Eigen centrality : {0}".format(top_15_sorted_left_eigen_centrality_g, graph_type))

        print("{1} Left Katz centrality : {0}".format(top_15_sorted_left_katz_centrality_g, graph_type))

        print("{1} pageRank : {0}".format(top_15_in_left_pagerank_g, graph_type))
        print("{1} in closeness : {0}".format(top_15_in_left_pagerank_g, graph_type))

        print("{1} betweenness : {0}".format(top_15_betweenness_g, graph_type))
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        node_centrality_set = set(top_15_sorted_left_eigen_centrality_g).union(set(top_15_sorted_left_katz_centrality_g)) \
                              .union(set(top_15_in_left_pagerank_g)).union(set(top_15_in_left_pagerank_g)).union(set(top_15_betweenness_g))
        print(node_centrality_set)
        return c_dict

    def draw_network(self, G, base_dir, pos, base_score):
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
        plt.title(str(Counter(communities)) + "_" + str(base_score))
        plt.axis('off')
        # pathlib.Path("output").mkdir(exist_ok=True)
        # print("Writing network figure to output/karate.png")
        plt.savefig(base_dir + "label_propagation_{0}.png".format(str(Counter(communities))))
        #plt.show()

    def save_result(self, all_df, result_dir):
        df = pd.DataFrame({"node": all_df.keys(), "value": all_df.values()})
        # saving as a CSV file
        df.to_csv(result_dir, sep='\t', index=False, header=False)