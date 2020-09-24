import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
import re
import pickle
import time
from datetime import timedelta
import pandas as pd

import utils.linguistic_features_utils as feature_utils

def merge_features(embedding, other_features, full_targets):
    orders = pd.read_csv('data/essays/author_id_order.csv').set_index(['order'])
    df = pd.merge(embedding, orders, left_index=True, right_index=True).set_index(['user'])
    df = pd.merge(df, other_features, left_index=True, right_index=True)
    # df = pd.merge(df, full_targets, left_index=True, right_index=True)
    # df = df.drop(['index'], axis=1)
    data_arr = df[df.columns[:-len(trait_labels)]].values
    targets_arr = df[df.columns[-len(trait_labels):]].values
    return data_arr, targets_arr


def prepare_model_inputs(meta_data):
    inp_dir, dataset, lr, batch_size, epochs, log_expdata, MODEL_INPUT, embed, layer, mode, embed_mode, jobid = meta_data

    if (MODEL_INPUT == 'LM_features'):
        if (re.search(r'base', embed)):
            n_hl = 12
            features_dim = 768

        elif (re.search(r'large', embed)):
            n_hl = 24
            features_dim = 1024

        file = open(inp_dir + dataset + '-' + embed + '-' + embed_mode + '-' + mode + '.pkl', 'rb')

        data = pickle.load(file)
        orders, data_x, data_y = list(zip(*data))
        file.close()

        # alphaW is responsible for which BERT layer embedding we will be using
        if (layer == 'all'):
            alphaW = np.full([n_hl], 1 / n_hl)

        else:
            alphaW = np.zeros([n_hl])
            alphaW[int(layer) - 1] = 1

        inputs, targets, author_ids = [], [], []
        n_batches = len(data_y)

        for ii in range(n_batches):
            inputs.extend(np.einsum('k,kij->ij', alphaW, data_x[ii]))
            targets.extend(data_y[ii])
            author_ids.extend(orders[ii])

        inputs = np.array(inputs)
        full_targets = np.array(targets)

        print('inputs shape: ', np.array(inputs).shape)
        print('author_ids shape: ', np.array(author_ids).shape)

    elif (MODEL_INPUT == 'psycholinguistic_features'):
        features_dim = 123

    elif (MODEL_INPUT == 'combined_features'):
        inputs = pd.DataFrame(np.array(inputs))
        inputs['order'] = author_ids
        inputs = inputs.set_index(['order'])
        full_targets = pd.DataFrame(np.array(targets))
        full_targets['order'] = author_ids
        full_targets = full_targets.set_index(['order'])
        
        other_features_df = feature_utils.get_psycholinguist_data(dump_data, dataset, feature_flags)
        
        inputs, full_targets = merge_features(inputs, other_features_df, full_targets)

        if (re.search(r'base', embed)):
            n_hl = 12
            features_dim = 768 + 123

        elif (re.search(r'large', embed)):
            n_hl = 24
            features_dim = 1024 + 123


    return inputs, full_targets, n_hl, features_dim