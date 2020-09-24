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
from utils.train_helper import prepare_model_inputs

inp_dir, dataset, lr, batch_size, epochs, log_expdata, MODEL_INPUT, embed, layer, mode, embed_mode, jobid = utils.parse_args()
meta_data = [inp_dir, dataset, lr, batch_size, epochs, log_expdata, MODEL_INPUT, embed, layer, mode, embed_mode, jobid]

print('{} : {} : {} : {} : {}'.format(dataset, embed, layer, mode, embed_mode))
n_classes = 2
network = 'MLP'
np.random.seed(jobid)
tf.random.set_seed(jobid)

nrc, nrc_vad, readability, mairesse = [True, True, True, True]
feature_flags = [nrc, nrc_vad, readability, mairesse]

start = time.time()

if dataset == 'essays':
    # dump_data = pd.read_csv('data/essays/essays.csv', index_col='#AUTHID')
    dump_data = dataset_processors.load_essays_df('data/essays/essays.csv')
    trait_labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

elif dataset == 'kaggle':
    # dump_data = pd.read_csv('data/kaggle/kaggle.csv', index_col='id')
    dump_data = dataset_processors.load_Kaggle_df('data/kaggle/kaggle.csv')
    trait_labels = ['E', 'N', 'F', 'J']

inputs, full_targets, n_hl, features_dim = prepare_model_inputs(meta_data)

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
