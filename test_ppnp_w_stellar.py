import networkx as nx
import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer.ppnp import PPNP
from stellargraph.layer.appnp import APPNP

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
#%matplotlib inline
from utils.graph_utils import GraphUtils
from sklearn.metrics import accuracy_score, recall_score,f1_score


def make_ppnp(train_targets, generator):
    ppnp = PPNP(
        layer_sizes=[90, 90, train_targets.shape[-1]],
        activations=["relu", "relu", "relu"],
        generator=generator,
        dropout=0.5,
        kernel_regularizer=keras.regularizers.l2(0.001),
    )

    x_inp, x_out = ppnp.in_out_tensors()
    #predictions = keras.layers.Softmax()(x_out)
    #predictions = keras.layers.()(x_out)
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)

    ppnp_model = Model(inputs=x_inp, outputs=predictions)
    ppnp_model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.mean_squared_error,
        metrics=["acc"],
    )
    embedding_model = Model(inputs=x_inp, outputs=x_out)
    return ppnp_model, embedding_model

def ppnp_train(ppnp_model, train_gen, val_gen):
    es_callback = EarlyStopping(
        monitor="val_acc", patience=50
    )  # patience is the number of epochs to wait before early stopping in case of no further improvement

    mc_callback = ModelCheckpoint(
        "logs/best_ppnp_model.h5",
        monitor="val_acc",
        save_best_only=True,
        save_weights_only=True,
    )

    history = ppnp_model.fit(
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
    result_file = "./result/" + "ppnp_result"+"_"+ str(round(f1,2)) + ".txt"
    GraphUtils().save_result(dict_800, result_file)

    print("\nTest train all node: \n \tacc : {0} f1 : {0}".format(score, f1))


def main():
    file_emb_output = "./emb/18_100_f1_0.916_nede2vec.emb"
    file_edge_path = "./data/edge_list.txt"
    file_node_path = "./data/class_info.txt"
    g, train_node, test_node = GraphUtils().make_graph(file_edge_path, file_node_path)

    G, train_gen, train_targets, val_gen, val_targets, test_targets, test_gen, all_gen, generator \
        = GraphUtils().preprocessing(g, train_node, file_emb_output)

    Gt, test_800_gen, generator = GraphUtils().preprocessing_predict(g, test_node, file_emb_output)

    ppnp_model, embedding_model = make_ppnp(train_targets, generator)
    ppnp_train(ppnp_model, train_gen, val_gen)
    GraphUtils().evaluate(ppnp_model, test_gen )
    all_node_test(ppnp_model, all_gen, train_node, test_800_gen, test_node  )
    GraphUtils().show_embedding(embedding_model, all_gen,train_node, 'PPNP')

if __name__ == "__main__":
    if not os.path.isdir("logs"):
        os.makedirs("logs")

    main()