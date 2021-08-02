import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import pickle
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import utils.gen_utils as utils


(
    inp_dir,
    dataset_type,
    network,
    lr,
    batch_size,
    epochs,
    seed,
    write_file,
    embed,
    layer,
    mode,
    embed_mode,
) = utils.parse_args()
n_classes = 2
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

start = time.time()


def classification(X_train, X_test, y_train, y_test, file_name):
    model_name = "LR-models/" + file_name + "-mean.joblib"
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    # joblib.dump(classifier, model_name)
    acc = classifier.score(X_test, y_test)
    return acc


if __name__ == "__main__":
    if embed == "bert-base":
        pretrained_weights = "bert-base-uncased"
        n_hl = 12
        hidden_dim = 768

    file = open(
        "../"
        + inp_dir
        + dataset_type
        + "-"
        + embed
        + "-"
        + embed_mode
        + "-"
        + mode
        + ".pkl",
        "rb",
    )

    data = pickle.load(file)
    data_x, data_y = list(zip(*data))
    file.close()

    # alphaW is responsible for which BERT layer embedding we will be using
    if layer == "all":
        alphaW = np.full([n_hl], 1 / n_hl)

    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1

    # just changing the way data is stored (tuples of minibatches) and getting the output for the required layer of BERT using alphaW
    # data_x[ii].shape = (12, batch_size, 768)
    inputs = []
    targets = []

    n_batches = len(data_y)

    for ii in range(n_batches):
        inputs.extend(np.einsum("k,kij->ij", alphaW, data_x[ii]))
        targets.extend(data_y[ii])

    if mode == "docbert":
        length_list = open("../num_sobdocuments_200.txt").read()
        length_list = length_list[1:-1].split(",")
        inputs_tmp = []
        for length in length_list:
            mean_vector = np.mean(inputs[: int(length)], axis=0)
            del [inputs[: int(length)]]
            inputs_tmp.append(mean_vector)
        inputs = inputs_tmp

    inputs = np.array(inputs)
    full_targets = np.array(targets)
    for trait_idx in range(full_targets.shape[1]):
        targets = full_targets[:, trait_idx]
        acc_list = []
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        k = 0
        for train_index, test_index in kf.split(inputs):
            X_train, X_test = inputs[train_index], inputs[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            acc = classification(
                X_train,
                X_test,
                y_train,
                y_test,
                "LR-"
                + dataset_type
                + "-"
                + embed
                + "-"
                + str(k)
                + "_t"
                + str(trait_idx)
                + "-"
                + embed_mode
                + "-"
                + mode,
            )
            print(acc)
            acc_list.append(acc)
            k += 1
        total_acc = np.mean(acc_list)
        print("trait: ", trait_idx)
        print("total_acc: ", total_acc)

        if write_file:
            results_file = (
                "results/LR_"
                + dataset_type
                + "_"
                + embed
                + "_t"
                + str(trait_idx)
                + "_results"
                + " - "
                + embed_mode
                + " - "
                + mode
                + ".txt"
            )
            file = open(results_file, "w")
            file.write("10 fold accs: " + str(acc_list) + "\n")
            file.write("total acc: " + str(total_acc))
            file.close()
