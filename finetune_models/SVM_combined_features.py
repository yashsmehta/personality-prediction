import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
import re
import pickle
import time
from datetime import timedelta
import pandas as pd
from pathlib import Path
from sklearn import svm

import utils.gen_utils as utils
import utils.dataset_processors as dataset_processors
import utils.linguistic_features_utils as feature_utils
from sklearn.model_selection import StratifiedKFold


def get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer):
    """Read data from pkl file and prepare for training."""

    file = open(
        inp_dir + dataset + "-" + embed + "-" + embed_mode + "-" + mode + ".pkl", "rb"
    )
    data = pickle.load(file)
    orders, data_x, data_y = list(zip(*data))
    file.close()

    # alphaW is responsible for which BERT layer embedding we will be using
    if layer == "all":
        alphaW = np.full([n_hl], 1 / n_hl)

    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1

    # just changing the way data is stored (tuples of minibatches) and getting the output for the required layer of BERT using alphaW
    inputs = []
    targets = []
    author_ids = []
    n_batches = len(data_y)
    print(len(orders))

    for ii in range(n_batches):
        inputs.extend(np.einsum("k,kij->ij", alphaW, data_x[ii]))
        targets.extend(data_y[ii])
        author_ids.extend(orders[ii])
    print("inputs shape: ", np.array(inputs).shape)
    print("author_ids shape: ", np.array(author_ids).shape)
    inputs = pd.DataFrame(np.array(inputs))
    inputs["order"] = author_ids
    inputs = inputs.set_index(["order"])
    full_targets = pd.DataFrame(np.array(targets))
    full_targets["order"] = author_ids
    full_targets = full_targets.set_index(["order"])
    if dataset == "essays":
        dump_data = dataset_processors.load_essays_df("../data/essays/essays.csv")
        trait_labels = ["EXT", "NEU", "AGR", "CON", "OPN"]

    elif dataset == "kaggle":
        dump_data = dataset_processors.load_Kaggle_df("../data/kaggle/kaggle.csv")
        trait_labels = ["E", "N", "F", "J"]

    _, _, _, other_features_df = feature_utils.get_psycholinguist_data(
        dump_data, dataset, feature_flags
    )
    inputs, full_targets = merge_features(inputs, other_features_df, full_targets)

    return inputs, full_targets, trait_labels


def merge_features(embedding, other_features, trait_labels):
    """Merge BERT and Psychologic features."""
    if dataset == "essays":
        orders = pd.read_csv("../data/essays/author_id_order.csv").set_index(["order"])
        df = pd.merge(embedding, orders, left_index=True, right_index=True).set_index(
            ["user"]
        )
    else:
        df = embedding
    df = pd.merge(df, other_features, left_index=True, right_index=True)
    data_arr = df[df.columns[: -len(trait_labels)]].values
    targets_arr = df[df.columns[-len(trait_labels) :]].values
    return data_arr, targets_arr


def classification(X_train, X_test, y_train, y_test, file_name):
    """Run classification algorithm (SVM)"""
    """ (commented lines can save SVM model) """
    classifier = svm.SVC(gamma="scale")
    classifier.fit(X_train, y_train)
    # model_name = file_name + '.joblib'
    # joblib.dump(classifier, model_name)
    acc = classifier.score(X_test, y_test)
    return acc


def training(inputs, full_targets, trait_labels):
    """Train MLP model for each trait on 10-fold corss-validtion."""
    n_splits = 10
    expdata = {}
    expdata["acc"], expdata["trait"], expdata["fold"] = [], [], []

    for trait_idx in range(full_targets.shape[1]):
        # convert targets to one-hot encoding
        targets = full_targets[:, trait_idx]
        expdata["trait"].extend([trait_labels[trait_idx]] * n_splits)
        expdata["fold"].extend(np.arange(1, n_splits + 1))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        k = -1
        for train_index, test_index in skf.split(inputs, targets):
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            k += 1
            acc = classification(
                x_train,
                x_test,
                y_train,
                y_test,
                "SVM-" + dataset + "-" + embed + "-" + str(k) + "_t" + str(trait_idx),
            )
            print(acc)
            expdata["acc"].append(acc)
    print(expdata)
    df = pd.DataFrame.from_dict(expdata)
    return df


def logging(df, log_expdata=True):
    """Save results and each models config and hyper parameters."""
    (
        df["network"],
        df["dataset"],
        df["lr"],
        df["batch_size"],
        df["epochs"],
        df["model_input"],
        df["embed"],
        df["layer"],
        df["mode"],
        df["embed_mode"],
        df["jobid"],
    ) = (
        network,
        dataset,
        lr,
        batch_size,
        epochs,
        MODEL_INPUT,
        embed,
        layer,
        mode,
        embed_mode,
        jobid,
    )

    pd.set_option("display.max_columns", None)
    print(df.head(5))

    # save the results of our experiment
    if log_expdata:
        Path(path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path + "expdata.csv"):
            df.to_csv(path + "expdata.csv", mode="a", header=True)
        else:
            df.to_csv(path + "expdata.csv", mode="a", header=False)


if __name__ == "__main__":
    (
        inp_dir,
        dataset,
        lr,
        batch_size,
        epochs,
        log_expdata,
        embed,
        layer,
        mode,
        embed_mode,
        jobid,
    ) = utils.parse_args()
    print("{} : {} : {} : {} : {}".format(dataset, embed, layer, mode, embed_mode))
    n_classes = 2
    features_dim = 123
    MODEL_INPUT = "combined_features"
    path = "explogs/"
    network = "SVM"
    np.random.seed(jobid)
    tf.random.set_seed(jobid)

    nrc, nrc_vad, readability, mairesse = [True, True, True, True]
    feature_flags = [nrc, nrc_vad, readability, mairesse]

    start = time.time()

    if re.search(r"base", embed):
        n_hl = 12
        features_dim += 768

    elif re.search(r"large", embed):
        n_hl = 24
        features_dim += 1024

    inputs, full_targets, trait_labels = get_inputs(
        inp_dir, dataset, embed, embed_mode, mode, layer
    )
    df = training(inputs, full_targets, trait_labels)
    logging(df, log_expdata)
