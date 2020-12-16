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

from gensim.models import Word2Vec
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score,f1_score
from utils.node2vec import Graph

from utils.graph_utils import GraphUtils


def save_model(obj, file_name_w_path ):
    with open(file_name_w_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return file_name_w_path

def load_model(file_name_w_path ):
    with open(file_name_w_path, 'rb') as f:
        return pickle.load(f)


def learn_embeddings(walks, dimensions, window_size, workers, iter):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    _walks = [map(str, walk) for walk in walks]
    _walks2 = [list(map(str, walk)) for walk in walks]

    embed_model = Word2Vec(_walks2, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers,
                     iter=iter)
    # file_emb_output = emb_dir + str(num_walks) + "_" + str(walk_length) + "_" + file_emb_output
    # model.wv.save_word2vec_format(file_emb_output)

    return embed_model

def emb_save(embed_model, num_walks, walk_length, f1, emb_dir):
    file_emb_output = "nede2vec.emb"
    file_emb_output = emb_dir + str(num_walks) + "_" + str(walk_length) + "_f1_"+ str(f1) +"_"+ file_emb_output
    embed_model.wv.save_word2vec_format(file_emb_output)


def make_random_walk(g1, p, q, num_walks, walk_length):
    G = Graph(g1, False, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    return walks


def get_embedding(walks, dimensions, window_size, workers, iter, emb_load_file):
    #file_emb_output = "nede2vec.emb"
    if emb_load_file == "" or emb_load_file is None :
        emb_model = learn_embeddings(walks, dimensions, window_size, workers, iter)
    else:
        emb_model = KeyedVectors.load_word2vec_format(emb_load_file)
    return emb_model


def preprocessing_data(emb_model, file_node_path):

    with open(file_node_path) as f:
        nodes = {str(int(line.split('\t')[0].rstrip("\n"))): int(line.split('\t')[1].rstrip("\n")) for line in f if
                 line.rstrip() != ''}

    node_ids = emb_model.wv.index2word
    node_embeddings = (
        emb_model.wv.vectors
    )

    df_node = pd.DataFrame(nodes.items())
    train_node = df_node.loc[df_node[1] != 0]
    test_node = df_node.loc[df_node[1] == 0]
    y = list()
    X = list()
    train_val = list()
    for i, node in enumerate(node_ids):
        if node in train_node[0].to_list():
            train_val.append(node)
            X.append(node_embeddings[i])
            y.append(int(train_node.loc[train_node[0] == node][1].values))

    test_val = list()
    test_X = list()
    test_y = list()
    for i, node in enumerate(node_ids):
         if node in test_node[0].to_list():
             test_val.append(node)
             test_X.append(node_embeddings[i])
             test_y.append(int(test_node.loc[test_node[0]==node][1].values))

    X = np.array(X)
    y = np.array(y)
    print(
        "Array shapes:\n X_train = {}\n y_train = {}\n".format(
            X.shape, y.shape
        )
    )
    return X, y, train_val, test_val, test_X, test_y

def cv_model_selection(X, y, num_walks, walk_length, dimensions, model_dir):
    print("Model initailization")
    clf = LogisticRegressionCV(
        Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=300
    )
    svc = SVC(C=0.25, gamma='auto', kernel='linear', class_weight='balanced')
    """
    kernel:{'linear', 'poly', 'rbf', 'sigmoid'}
    """
    model_list = [clf, svc]

    kfold = KFold(n_splits=4, random_state=0, shuffle=True)
    model_sel = list()
    for model in model_list:
        #acc_l = list()
        #recall_l = list()
        f1_l = list()
        for train_index, validate_index in kfold.split(X):
            X_train, X_validate = X[train_index], X[validate_index]
            y_train, y_validate = y[train_index], y[validate_index]
            model.fit(X_train, y_train)
            svc_pred = model.predict(X_validate)
            acc = accuracy_score(y_validate, svc_pred)
            recall = recall_score(y_validate, svc_pred)
            f1 = f1_score(y_validate, svc_pred)
            f1_l.append(f1)
            print(str(model) + "->   Acc: ", acc, "\t Recall: ", round(recall, 2), "\t F1: ", round(f1, 2))

        np_f1_mean = round(np.mean(np.array(f1_l)),3)
        model_name = model_dir + str(num_walks) + "_" + str(walk_length) + "_" + str(dimensions) \
                  + "_" + str(model.__class__.__name__) + "_n2v_" + str(np_f1_mean) +".bin"
        row_dict = {"f1" : np_f1_mean, "model": model, "model_file" : model_name}
        model_sel.append(row_dict)

        save_model(model, model_name)

    df_model_sel = pd.DataFrame(model_sel)
    best_model = df_model_sel.loc[df_model_sel.f1.idxmax()]["model"]
    best_model_file = df_model_sel.loc[df_model_sel.f1.idxmax()]["model_file"]
    best_f1 = df_model_sel.loc[df_model_sel.f1.idxmax()][0]
    print("All model info : \n{0}".format(df_model_sel))
    print("The best model is {0}".format(best_model))
    return best_model, best_model_file, best_f1

def n2v_prediction(best_model, X, test_X, train_val, y, test_val, best_model_file, model_file_load=None):

    if model_file_load is None or model_file_load == "":
        model = best_model
    else:
        # model_name = model_dir + file_name
        model = load_model(model_file_load)

    model_pred_test = model.predict(test_X)

    model_pred_train = model.predict(X)

    acc = accuracy_score(y, model_pred_train)
    f1 = f1_score(y, model_pred_train)
    print("saving model eval -> acc : {0} f1 : {1}".format(acc, f1))

    dict_200 = dict(zip(list(map(int,train_val)), list(y)))
    dict_800 = dict(zip(list(map(int,test_val)), list(model_pred_test)))

    all_df = {**dict_200, **dict_800}

    dict_800 = dict(sorted(dict_800.items()))

    result_file = result_dir + os.path.splitext(os.path.basename(best_model_file))[0]+"_"+ str(f1) + ".txt"
    GraphUtils().save_result(dict_800, result_file)

def main():
    file_edge_path = "./data/edge_list.txt"
    file_node_path = "./data/class_info.txt"

    emb_dir = "./emb/"
    os.makedirs(emb_dir, exist_ok=True)

    model_dir = "./model/"
    os.makedirs(model_dir, exist_ok=True)

    # node vec parameter
    p = 1
    q = 1
    num_walks = 50#40#30#18 #50 #40 #30 #18
    walk_length = 200#300#300 #200 # 500 #400 #200 #100
    dimensions = 128
    window_size = 15
    workers = 8
    iter = 3

    emb_load_file = ""

    # Uncomment it For load embedding
    emb_load_file = "./emb/50_300_f1_0.916_nede2vec.emb"
    #emb_load_file = "./emb/50_200_f1_0.899_nede2vec.emb"
    model_file_load = ""
    # Uncomment it For load model
    model_file_load = "./model/50_300_128_LogisticRegressionCV_n2v_0.916.bin"
    #model_file_load = "./model/50_200_128_LogisticRegressionCV_n2v_0.899.bin"
    best_model_file = model_file_load
    model = None
    walks = None

    # make graph
    g, train_node, test_node = GraphUtils().make_graph(file_edge_path, file_node_path)

    # make random walks
    if emb_load_file is None or emb_load_file == "":
        walks = make_random_walk(g, p, q, num_walks, walk_length)

    # make embedding
    emb_model = get_embedding(walks, dimensions, window_size, workers, iter, emb_load_file)

    # preparing data
    X, y, train_val, test_val, test_X, test_y = preprocessing_data(emb_model, file_node_path)


    if model_file_load is None or model_file_load == "":
        # classifier train
        model, best_model_file, best_f1 = cv_model_selection(X, y, num_walks, walk_length, dimensions, model_dir)

        # embedding model save with parameter
        emb_save(emb_model, num_walks, walk_length, best_f1, emb_dir)


    # prediction
    n2v_prediction(model, X, test_X, train_val, y, test_val, best_model_file, model_file_load )


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