import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from stellargraph import StellarGraph
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import networkx as nx
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import accuracy_score, recall_score,f1_score

from utils.graph_utils import GraphUtils


def make_gcn(train_targets, generator):
    gcn = GCN(
        layer_sizes=[90, 90], activations=["relu", "relu"], generator=generator, dropout=0.5
    )

    x_inp, x_out = gcn.in_out_tensors()
    #predictions = keras.layers.Softmax()(x_out)
    #predictions = keras.layers.()(x_out)
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)

    gcn_model = Model(inputs=x_inp, outputs=predictions)
    gcn_model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.mean_squared_error,
        metrics=["acc"],
    )
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    return gcn_model, embedding_model


def gcn_train(gcn_model, train_gen, val_gen):
    es_callback = EarlyStopping(
        monitor="val_acc", patience=50
    )  # patience is the number of epochs to wait before early stopping in case of no further improvement

    mc_callback = ModelCheckpoint(
        "model/best_gcn_model.h5",
        monitor="val_acc",
        save_best_only=True,
        save_weights_only=True,
    )

    history = gcn_model.fit(
        train_gen,
        epochs=150,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )

    fig = sg.utils.plot_history(history,return_figure=True )
    fig.show()


def all_node_test(model, all_gen, train_node, test_800_gen, test_node ):
    #all_nodes = node_subjects.index
    #all_gen = generator.flow(all_nodes)
    all_predictions = model.predict(all_gen)
    predicted_node = np.where(all_predictions > 0.5, '1', '-1')
    # accuracy_score(node_subjects.value, predicted_node.squeeze())

    score = accuracy_score(train_node['values'].values.astype(str), predicted_node.squeeze())
    f1 = f1_score(train_node['values'].values.astype(int), predicted_node.squeeze().astype(int))

    predictions_800 = model.predict(test_800_gen)
    predicted_800_node = np.where(predictions_800 > 0.5, '1', '-1')
    dict_800 = dict(zip(list(map(int,test_node['node'].values)), list(predicted_800_node.squeeze())))
    result_file = "./result/" + "gcn_result"+"_"+ str(round(f1,2)) + ".txt"
    GraphUtils().save_result(dict_800, result_file)

    print("\nTest train all node: \n \tacc : {0} f1: {1}".format(score, f1))


def main():
    file_emb_output = "./emb/50_300_f1_0.916_nede2vec.emb"
    file_edge_path = "./data/edge_list.txt"
    file_node_path = "./data/class_info.txt"
    g, train_node, test_node = GraphUtils().make_graph(file_edge_path, file_node_path)
    G, train_gen, train_targets, val_gen, val_targets, test_targets, test_gen, all_gen, generator \
        = GraphUtils().preprocessing(g, train_node, file_emb_output)

    Gt, test_800_gen, generator = GraphUtils().preprocessing_predict(g, test_node, file_emb_output)

    gcn_model, embedding_model = make_gcn(train_targets, generator)
    #
    gcn_train(gcn_model, train_gen, val_gen)
    GraphUtils().evaluate(gcn_model, test_gen )
    all_node_test(gcn_model, all_gen, train_node, test_800_gen, test_node )
    GraphUtils().show_embedding(embedding_model, all_gen,train_node, "GCN")


if __name__ == "__main__":
    if not os.path.isdir("model"):
        os.makedirs("model")

    result_dir = "./result/"
    os.makedirs(result_dir, exist_ok=True)

    main()