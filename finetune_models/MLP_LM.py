import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import numpy as np
import re
import pickle
import time
import pandas as pd
from pathlib import Path

# add parent directory to the path as well, if running from the finetune folder
parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

sys.path.insert(0, os.getcwd())

import utils.gen_utils as utils 

def get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer):
    """ Read data from pkl file and prepare for training. """
    file = open(inp_dir + dataset + '-' + embed + '-' + embed_mode + '-' + mode + '.pkl', 'rb')
    data = pickle.load(file)
    author_ids, data_x, data_y = list(zip(*data))
    file.close()

    # alphaW is responsible for which BERT layer embedding we will be using
    if layer == 'all':
        alphaW = np.full([n_hl], 1 / n_hl)

    else:
        alphaW = np.zeros([n_hl])
        alphaW[int(layer) - 1] = 1

    # just changing the way data is stored (tuples of minibatches) and
    # getting the output for the required layer of BERT using alphaW
    inputs = []
    targets = []
    n_batches = len(data_y)
    for ii in range(n_batches):
        inputs.extend(np.einsum('k,kij->ij', alphaW, data_x[ii]))
        targets.extend(data_y[ii])

    inputs = np.array(inputs)
    full_targets = np.array(targets)

    return inputs, full_targets

def training(dataset,inputs, full_targets):
    """ Train MLP model for each trait on 10-fold corss-validtion. """
    if (dataset == 'kaggle'):
        trait_labels = ['E', 'N', 'F', 'J']
    else:
        trait_labels = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']

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
            model.add(tf.keras.layers.Dense(50, input_dim=hidden_dim, activation='relu'))
            model.add(tf.keras.layers.Dense(n_classes))

            k += 1
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          metrics=['mse', 'accuracy'])

            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(x_test, y_test), verbose=0)

            expdata['acc'].append(100 * max(history.history['val_accuracy']))
    print(expdata)
    df = pd.DataFrame.from_dict(expdata)
    return df

def logging(df, log_expdata=True):
    """ Save results and each models config and hyper parameters."""
    df['network'], df['dataset'], df['lr'], df['batch_size'], df['epochs'], df['model_input'], df['embed'], df['layer'], df[
        'mode'], df['embed_mode'], df['jobid'] = network, \
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

if __name__ == "__main__":
    inp_dir, dataset, lr, batch_size, epochs, log_expdata, embed, layer, mode, embed_mode, jobid = utils.parse_args()
    # embed_mode {mean, cls}
    # mode {512_head, 512_tail, 256_head_tail}

    network = 'MLP'
    MODEL_INPUT = 'LM_features'
    print('{} : {} : {} : {} : {}'.format(dataset, embed, layer, mode, embed_mode))
    n_classes = 2
    seed = jobid
    np.random.seed(seed)
    tf.random.set_seed(seed)

    start = time.time()
    path = 'explogs/'

    if (re.search(r'base', embed)):
        n_hl = 12
        hidden_dim = 768

    elif (re.search(r'large', embed)):
        n_hl = 24
        hidden_dim = 1024

    inputs, full_targets = get_inputs(inp_dir, dataset, embed, embed_mode, mode, layer)
    df = training(dataset, inputs, full_targets)
    logging(df, log_expdata)


