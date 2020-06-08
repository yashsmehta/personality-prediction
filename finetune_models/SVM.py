import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import csv
import pickle
import time
import joblib
from datetime import timedelta
from sklearn import svm
from sklearn.model_selection import KFold

import utils


def classification(X_train, X_test, y_train, y_test, file_name):
    model_name = file_name + '.joblib'
    if os.path.isfile(model_name):
        classifier = joblib.load(model_name)
    else:
        classifier = svm.SVC(gamma="scale")
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, model_name)
    acc = classifier.score(X_test, y_test)
    return acc


inp_dir, dataset_type, network, lr, batch_size, epochs, seed, write_file, embed, layer, mode, embed_mode = utils.parse_args()
n_classes = 2
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

start = time.time()

if (embed == 'bert-base'):
    pretrained_weights = 'bert-base-uncased'
    n_hl = 12
    hidden_dim = 768

elif (embed == 'bert-large'):
    pretrained_weights = 'bert-large-uncased'
    n_hl = 24
    hidden_dim = 1024

file = open(inp_dir + dataset_type + '-' + embed + '.pkl', 'rb')

data = pickle.load(file)
data_x, data_y = list(zip(*data))
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
        acc = classification(X_train, X_test, y_train, y_test,
                             'SVM-' + dataset_type + '-' + embed + '-' + str(k) + "_t" + str(trait_idx))
        print(acc)
        acc_list.append(acc)
        k += 1
    total_acc = np.mean(acc_list)
    print('trait: ', trait_idx)
    print('total_acc: ', total_acc)

    if (write_file):
        results_file = 'SVM_' + dataset_type + '_' + embed + "_t" + str(trait_idx) + '_results.txt'
        file = open(results_file, 'w')
        file.write('10 fold accs: ' + str(acc_list) + '\n')
        file.write('total acc: ' + str(total_acc))
        file.close()
