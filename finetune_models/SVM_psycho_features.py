import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import csv
import pickle
import time
import pandas as pd
import tensorflow as tf
import re
import preprocessor as p
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from scipy.io import arff
import joblib
from sklearn import svm

import sys

sys.path.insert(0, '/nfs/ghome/live/yashm/Desktop/research/personality/utils')

import utils.gen_utils as utils
import utils.dataset_processors as dataset_processors
import utils.linguistic_features_utils as feature_utils

inp_dir, dataset, lr, batch_size, epochs, log_expdata, embed, layer, mode, embed_mode, jobid = utils.parse_args()

features_dim = 123
# embed = 'psycholinguist features'
layer = ''
path = 'explogs/'
n_classes = 2
MODEL_INPUT = 'psycholinguist_features'
network = 'SVM'
print(network)

nrc, nrc_vad, readability, mairesse = [True, True, True, True]
feature_flags = [nrc, nrc_vad, readability, mairesse]

np.random.seed(jobid)
tf.random.set_seed(jobid)

start = time.time()

def classification(X_train, X_test, y_train, y_test, file_name):
    classifier = svm.SVC(gamma="scale")
    classifier.fit(X_train, y_train)
    # model_name = file_name + '.joblib'
    # joblib.dump(classifier, model_name)
    acc = classifier.score(X_test, y_test)
    return acc

if __name__ == "__main__":
    if dataset == 'essays':
        dump_data = dataset_processors.load_essays_df('../data/essays/essays.csv')
        trait_labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']
    elif dataset == 'kaggle':
        dump_data = dataset_processors.load_Kaggle_df('../data/kaggle/kaggle.csv')
        trait_labels = ['E', 'N', 'F', 'J']
    print('dataset loaded! Getting psycholinguistic features...')
    inputs, full_targets, feature_names, _ = feature_utils.get_psycholinguist_data(dump_data, dataset, feature_flags)
    inputs = np.array(inputs)
    full_targets = np.array(full_targets)

    print(inputs.shape)
    print(full_targets.shape)
    print(feature_names)
    print('starting k-fold cross validation...')
    
    n_splits = 10
    fold_acc = {}
    expdata = {}
    expdata['acc'], expdata['trait'], expdata['fold'] = [], [], []

    for trait_idx in range(full_targets.shape[1]):
        # convert targets to one-hot encoding
        targets = full_targets[:, trait_idx]
        n_data = targets.shape[0]

        expdata['trait'].extend([trait_labels[trait_idx]] * n_splits)
        expdata['fold'].extend(np.arange(1, n_splits + 1))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        k = -1
        acc_list = []
        for train_index, test_index in skf.split(inputs, targets):
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            k += 1
            acc = classification(x_train, x_test, y_train, y_test,
                             'SVM-' + dataset + '-' + embed + '-' + str(k) + "_t" + str(trait_idx))
            print(acc)
            expdata['acc'].append(acc)


    print (expdata)

    df = pd.DataFrame.from_dict(expdata)

    df['network'], df['dataset'], df['lr'], df['batch_size'], df['epochs'], df['model_input'], df['embed'], df['layer'], \
    df['mode'], df['embed_mode'], df['jobid'] = network, \
                                                dataset, lr, batch_size, epochs, MODEL_INPUT, embed, layer, mode, embed_mode, jobid

    pd.set_option('display.max_columns', None)
    print(df.head(5))

    # save the results of our experiment
    if (log_expdata):
        Path(path).mkdir(parents=True, exist_ok=True)
        if (not os.path.exists(path + 'expdata.csv')):
            df.to_csv(path + 'expdata.csv', mode='a', header=True)
        else:
            df.to_csv(path + 'expdata.csv', mode='a', header=False)
