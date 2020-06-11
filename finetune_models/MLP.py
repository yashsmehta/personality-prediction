import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import csv
import re
import pickle
import time
from datetime import timedelta
import pandas as pd

import utils.gen_utils as utils

inp_dir, dataset_type, _, lr, batch_size, epochs, seed, write_file, embed, layer, mode, embed_mode = utils.parse_args()
print('{} : {} : {} : {} : {}'.format(dataset_type, embed, layer, mode, embed_mode))
n_classes = 2
np.random.seed(seed)
tf.random.set_seed(seed)

start = time.time()

def load_features(dir, dataset_type):
    if dataset_type == 'kaggle' or dataset_type == 'pandora':
        drop_cols = ['BROWN-FREQ numeric', 'K-F-FREQ numeric', 'K-F-NCATS numeric', 'K-F-NSAMP numeric',
                     'T-L-FREQ numeric', 'Extraversion numeric'
            , '\'Emotional stability\' numeric', 'Agreeableness numeric', 'Conscientiousness numeric',
                     '\'Openness to experience\' numeric']
        mairesse = utils.read_and_process(dir + dataset_type + '_mairesse_labeled.arff')
        mairesse = mairesse.drop(drop_cols, axis=1)
    elif dataset_type == 'essays':
        idx = '#AUTHID'
        mairesse = pd.read_csv(dir + dataset_type + '_mairesse_labeled.csv')
    mairesse = mairesse.set_index(mairesse.columns[0])
    nrc = pd.read_csv(dir + dataset_type + '_nrc.csv').set_index([idx])
    nrc_vad = pd.read_csv(dir + dataset_type + '_nrc_vad.csv').set_index([idx])
    hourglass = pd.read_csv(dir + dataset_type + '_hourglass.csv').set_index([idx])
    readability = pd.read_csv(dir + dataset_type + '_readability.csv').set_index([idx])

    return [nrc, nrc_vad, readability, mairesse, hourglass]

def get_psycholinguist_data(dump_data, dataset_type):
    features = load_features('../data/' + dataset_type + '/psycholinguist_features/', dataset_type)
    first = 1
    for feature in features:
        if first:
            df = feature
            first = 0
        else:
            df = pd.merge(df, feature, left_index=True, right_index=True)
    df = pd.merge(dump_data['text'], df, left_index=True, right_index=True)
    df.reset_index(inplace=True)
    df = df.drop(['text'], axis=1)
    return df


def merge_features(embedding, other_features, full_targets):
    df = pd.merge(embedding, other_features, left_index=True, right_index=True)
    df = pd.merge(df, full_targets, left_index=True, right_index=True)
    df = df.drop(['index'], axis=1)
    data_arr = df[df.columns[:-len(trait_labels)]].values
    targets_arr = df[df.columns[-len(trait_labels):]].values
    return data_arr, targets_arr

if (re.search(r'base', embed)):
    n_hl = 12
    hidden_dim = 768

elif (re.search(r'large', embed)):
    n_hl = 24
    hidden_dim = 1024

file = open(inp_dir + dataset_type + '-' + embed + '-' + embed_mode + '-' + mode + '.pkl', 'rb')

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

n_batches = len(data_y)

for ii in range(n_batches):
    inputs.extend(np.einsum('k,kij->ij', alphaW, data_x[ii]))
    targets.extend(data_y[ii])


inputs = pd.DataFrame(np.array(inputs))
inputs['order'] = orders[0]
inputs = inputs.set_index(['order'])
full_targets = pd.DataFrame(np.array(targets))
full_targets['order'] = orders[0]
full_targets = full_targets.set_index(['order'])

if dataset_type == 'essays':
    dump_data = pd.read_csv('../data/essays/essays.csv', index_col='#AUTHID')
    trait_labels = ['EXT','NEU','AGR','CON','OPN']
elif dataset_type == 'kaggle':
    dump_data = pd.read_csv('../data/kaggle/kaggle.csv', index_col='id')
    trait_labels = ['E', 'N', 'F', 'J']

other_featers_df = get_psycholinguist_data(dump_data, dataset_type)
inputs, full_targets = merge_features(inputs, other_featers_df, full_targets)

fold_acc = {}
for trait_idx in range(full_targets.shape[1]):
    # convert targets to one-hot encoding
    targets = tf.keras.utils.to_categorical(full_targets[:, trait_idx], num_classes=n_classes)
    n_data = targets.shape[0]
    fold_acc[trait_labels[trait_idx]] = []
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    k = -1
    for train_index, test_index in kf.split(inputs):
        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        k+=1
        model = tf.keras.models.Sequential()

        # define the neural network architecture
        model.add(tf.keras.layers.Dense(50, input_dim=hidden_dim, activation='relu'))
        # model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(n_classes))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['mse', 'accuracy'])
        if(k==0):
            print(model.summary())
        
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=0)
        
        print('fold : {} \ntrait : {}\n'.format(k+1, trait_labels[trait_idx]))
        
        # print('\nacc: ', history.history['accuracy'])
        # print('val acc: ', history.history['val_accuracy'])
        # print('MAX', max(history.history['val_accuracy']),'\n')
        fold_acc[trait_labels[trait_idx]].append(max(history.history['val_accuracy']))
        # print('loss: ', history.history['loss'])
        # print('val loss: ', history.history['val_loss'])

        print(timedelta(seconds=int(time.time() - start)), end=' ')

        if (write_file):
            results_file = "MLP_t" + str(trait_idx) + '_results.csv'
            meta_info = (lr, epochs, seed, embed, layer)
            utils.file_writer(results_file, meta_info, history.history['val_accuracy'], history.history['val_loss'], str(k))

print(fold_acc)
for trait in fold_acc.keys():
    fold_acc[trait] = np.mean(fold_acc[trait])

print(fold_acc)
