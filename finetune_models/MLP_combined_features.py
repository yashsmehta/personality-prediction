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

import utils.gen_utils as utils
import utils.dataset_processors as dataset_processors
import utils.linguistic_features_utils as feature_utils
from sklearn.model_selection import StratifiedKFold

inp_dir, dataset, lr, batch_size, epochs, log_expdata, embed, layer, mode, embed_mode, jobid = utils.parse_args()
print('{} : {} : {} : {} : {}'.format(dataset, embed, layer, mode, embed_mode))
n_classes = 2
features_dim = 123
network = 'MLP'
np.random.seed(jobid)
tf.random.set_seed(jobid)

nrc, nrc_vad, readability, mairesse = [True, True, True, True]
feature_flags = [nrc, nrc_vad, readability, mairesse]

start = time.time()

def merge_features(embedding, other_features, full_targets):
    if dataset == 'essays':
        orders = pd.read_csv('data/essays/author_id_order.csv').set_index(['order'])
        df = pd.merge(embedding, orders, left_index=True, right_index=True).set_index(['user'])
    else:
        df = embedding
    df = pd.merge(df, other_features, left_index=True, right_index=True)
    # df = pd.merge(df, full_targets, left_index=True, right_index=True)
    # df = df.drop(['index'], axis=1)
    data_arr = df[df.columns[:-len(trait_labels)]].values
    targets_arr = df[df.columns[-len(trait_labels):]].values
    return data_arr, targets_arr


if (re.search(r'base', embed)):
    n_hl = 12
    features_dim += 768

elif (re.search(r'large', embed)):
    n_hl = 24
    features_dim += 1024

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

# just changing the way data is stored (tuples of minibatches) and getting the output for the required layer of BERT using alphaW
# data_x[ii].shape = (12, batch_size, 768)
inputs = []
targets = []
author_ids = []

n_batches = len(data_y)
print(len(orders))

for ii in range(n_batches):
    inputs.extend(np.einsum('k,kij->ij', alphaW, data_x[ii]))
    targets.extend(data_y[ii])
    author_ids.extend(orders[ii])

print('inputs shape: ', np.array(inputs).shape)
print('author_ids shape: ', np.array(author_ids).shape)

inputs = pd.DataFrame(np.array(inputs))
inputs['order'] = author_ids
inputs = inputs.set_index(['order'])
full_targets = pd.DataFrame(np.array(targets))
full_targets['order'] = author_ids
full_targets = full_targets.set_index(['order'])

if dataset == 'essays':
    # dump_data = pd.read_csv('data/essays/essays.csv', index_col='#AUTHID')
    dump_data = dataset_processors.load_essays_df('data/essays/essays.csv')
    trait_labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

elif dataset == 'kaggle':
    # dump_data = pd.read_csv('data/kaggle/kaggle.csv', index_col='id')
    dump_data = dataset_processors.load_Kaggle_df('data/kaggle/kaggle.csv')
    trait_labels = ['E', 'N', 'F', 'J']

other_features_df = feature_utils.get_psycholinguist_data(dump_data, dataset, feature_flags)
inputs, full_targets = merge_features(inputs, other_features_df, full_targets)

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
    for train_index, test_index in skf.split(inputs, targets):
        x_train, x_test = inputs[train_index], inputs[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        # converting to one-hot embedding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
        model = tf.keras.models.Sequential()

        # define the neural network architecture
        model.add(tf.keras.layers.Dense(50, input_dim=features_dim, activation='relu'))
        # model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(n_classes))

        k += 1
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['mse', 'accuracy'])

        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(x_test, y_test), verbose=0)

        # if(k==0):
        #     print(model.summary())

        # print('\nacc: ', history.history['accuracy'])
        print('val acc: ', history.history['val_accuracy'])
        # print('loss: ', history.history['loss'])
        # print('val loss: ', history.history['val_loss'])
        expdata['acc'].append(100 * max(history.history['val_accuracy']))

print(expdata)

df = pd.DataFrame.from_dict(expdata)

df['network'], df['dataset'], df['lr'], df['batch_size'], df['epochs'], df['embed'], df['layer'], df['mode'], df[
    'embed_mode'], df['jobid'] = network, \
                                 dataset, lr, batch_size, epochs, embed, layer, mode, embed_mode, jobid

pd.set_option('display.max_columns', None)
print(df.head(5))

# save the results of our experiment
if (log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if (not os.path.exists(path + 'expdata.csv')):
        df.to_csv(path + 'expdata.csv', mode='a', header=True)
    else:
        df.to_csv(path + 'expdata.csv', mode='a', header=False)
