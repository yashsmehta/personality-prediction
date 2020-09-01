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
from sklearn.model_selection import StratifiedKFold
from scipy.io import arff

import sys
sys.path.insert(0,'/nfs/ghome/live/yashm/Desktop/research/personality/utils')

import gen_utils as utils

log_expdata = False
epochs = 3
batch_size = 32
dataset = 'essays'
path = 'explogs/'
n_classes = 2
seed = 0
network = 'MLP'
print(network)

mairesse, nrc, nrc_vad, affectivespace, hourglass, readability = utils.parse_args_SHAP()

np.random.seed(seed)
tf.random.set_seed(seed)

start = time.time()

def read_and_process(path):
    arff = open(path, 'r')
    attributes = []
    values = []
    is_attr = True
    arff.readline()
    arff.readline()
    while is_attr:
        line = arff.readline()
        if len(line.split()) == 0:
            is_attr = False
            continue
        type = line.split()[0]
        attr = ' '.join(line.split()[1:])
        if type == "@attribute":
            attributes.append(attr)
        else:
            is_attr = False
    for line in arff.readlines():
        if len(line.split(",")) < 10:
            continue
        else:
            components = line.split(",")
            values.append(components)
            name = components[0].replace("\'", "").split("\\\\")[-1]
            values[-1][0] = name
    df = pd.DataFrame(columns=attributes, data=values)
    df['idx'] = [int(re.sub('id_', '', i)) for i in df[df.columns[0]]]
    df = df.drop(df.columns[0], axis=1)
    df = df.set_index(['idx'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.sort_index()
    return df

def sentence_preprocess(sentence):
    sentence = p.clean(sentence)
    # Remove hyperlinks
    sentence = re.sub(r'http\S+', ' ', sentence)
    # Remove punctuations and numbers
    # sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub('[^a-zA-Z.?!,]', ' ', sentence)
    # Single character removal (except I)
    sentence = re.sub(r"\s+[a-zA-HJ-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def load_features(dir, dataset):
    idx = 'id'
    if dataset == 'kaggle':
        drop_cols = ['BROWN-FREQ numeric', 'K-F-FREQ numeric', 'K-F-NCATS numeric', 'K-F-NSAMP numeric', 'T-L-FREQ numeric', 'Extraversion numeric'
                  , '\'Emotional stability\' numeric', 'Agreeableness numeric', 'Conscientiousness numeric', '\'Openness to experience\' numeric']
        mairesse = read_and_process(dir + dataset + '_mairesse_labeled.arff')
        mairesse = mairesse.drop(drop_cols, axis=1)
    elif dataset == 'essays':
        idx = '#AUTHID'
        mairesse = pd.read_csv(dir + dataset + '_mairesse_labeled.csv')
    # mairesse = mairesse.set_index(mairesse.columns[0])
    nrc = pd.read_csv(dir + dataset + '_nrc.csv').set_index([idx])
    # nrc = nrc.sort_values(by=['id'])
    # nrc = nrc.drop(['id'], axis=1)
    nrc_vad = pd.read_csv(dir + dataset + '_nrc_vad.csv').set_index([idx])
    # nrc_vad = nrc_vad.sort_values(by=['id'])
    # nrc_vad = nrc_vad.drop(['id'], axis=1)
    # affectivespace = pd.read_csv(dir + 'essays_affectivespace.csv').set_index(['#AUTHID'])
    # hourglass = pd.read_csv(dir + dataset + '_hourglass.csv').set_index([idx])
    readability = pd.read_csv(dir + dataset + '_readability.csv').set_index([idx])

    return [nrc, nrc_vad, readability, mairesse]


def load_essays_df(datafile):
    with open(datafile, "rt") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        df = pd.DataFrame(columns=["user", "text", "EXT", "NEU", "AGR", "CON", "OPN"])
        for line in csvreader:
            if first_line:
                first_line = False
                continue
            text = line[1]
            text = sentence_preprocess(text)

            df = df.append({"user": line[0],
                            "text": text,
                            "EXT": 1 if line[2].lower() == 'y' else 0,
                            "NEU": 1 if line[3].lower() == 'y' else 0,
                            "AGR": 1 if line[4].lower() == 'y' else 0,
                            "CON": 1 if line[5].lower() == 'y' else 0,
                            "OPN": 1 if line[6].lower() == 'y' else 0}, ignore_index=True)
    return df


def load_Kaggle_df(datafile):
    with open(datafile, "rt") as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        first_line = True
        df = pd.DataFrame(columns=["user", "text", "E", "N", "F", "J"])
        for line in csvreader:
            if first_line:
                first_line = False
                continue

            text = line[1]

            df = df.append({"user": line[3],
                            "text": text,
                            "E": 1 if line[0][0] == 'E' else 0,
                            "N": 1 if line[0][1] == 'N' else 0,
                            "F": 1 if line[0][2] == 'F' else 0,
                            "J": 1 if line[0][3] == 'J' else 0, }, ignore_index=True)

    print('E : ', df['E'].value_counts())
    print('N : ', df['N'].value_counts())
    print('F : ', df['F'].value_counts())
    print('J : ', df['J'].value_counts())

    return df

def get_psycholinguist_data(dump_data, dataset):
    features = load_features('data/'+dataset+'/psycholinguist_features/', dataset)
    feature_flags = [nrc, nrc_vad, readability, mairesse]
    first = 1
    for feature, feature_flag in zip(features, feature_flags):
        if feature_flag:
            if first:
                df = feature
                first = 0
            else:
                df = pd.merge(df, feature, left_index=True, right_index=True)
    if dataset == 'essays':
        labels = dump_data[['user', 'EXT', 'NEU', 'AGR', 'CON', 'OPN']]
    if dataset == 'kaggle':
        labels = dump_data[['user', 'E', 'N', 'F', 'J']]
    labels = labels.set_index('user')
    labels.index = pd.to_numeric(labels.index, errors='coerce')
    df.index = pd.to_numeric(df.index, errors='coerce')
    merged = pd.merge(df, labels, left_index=True, right_index=True).fillna(0)
    data = merged[merged.columns[:-4]].values
    full_targets = merged[merged.columns[-4:]].values
    feature_names = merged.columns
    return data, full_targets, feature_names

if __name__ == "__main__":
    if dataset == 'essays':
        dump_data = load_essays_df('data/essays/essays.csv')
        labels_list = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']
    elif dataset == 'kaggle':
        dump_data = load_Kaggle_df('../data/kaggle/kaggle.csv')
        labels_list = ['E', 'N', 'F', 'J']
    print('dataset loaded! Getting psycholinguistic features...')
    inputs, full_targets, feature_names = get_psycholinguist_data(dump_data, dataset)
    inputs = np.array(inputs)
    full_targets = np.array(full_targets)

    print(inputs.shape)
    print(full_targets.shape)
    print(feature_names)
    print('starting k-fold cross validation...')
    trait_labels = ['EXT','NEU','AGR','CON','OPN']
    n_splits = 10
    fold_acc = {}
    expdata = {}
    expdata['acc'], expdata['trait'], expdata['fold'] = [],[],[]

    for trait_idx in range(full_targets.shape[1]):
        # convert targets to one-hot encoding
        targets = full_targets[:, trait_idx]
        n_data = targets.shape[0]
        
        expdata['trait'].extend([trait_labels[trait_idx]] * n_splits)
        expdata['fold'].extend(np.arange(1,n_splits+1))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        k = -1

        for train_index, test_index in skf.split(inputs, targets):
            x_train, x_test = inputs[train_index], inputs[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            #converting to one-hot embedding
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
            model = tf.keras.models.Sequential()

            # define the neural network architecture
            model.add(tf.keras.layers.Dense(50, input_dim=hidden_dim, activation='relu'))
            # model.add(tf.keras.layers.Dense(50, activation='relu'))
            model.add(tf.keras.layers.Dense(n_classes))

            # model.add(tf.keras.layers.Dense(n_classes, input_dim=hidden_dim))

            k+=1
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['mse', 'accuracy'])
            
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(x_test, y_test), verbose=0)
            
            # if(k==0):
            #     print(model.summary())
            
            # print('fold : {} \ntrait : {}\n'.format(k+1, trait_labels[trait_idx]))
            
            # print('\nacc: ', history.history['accuracy'])
            print('val acc: ', history.history['val_accuracy'])
            # print('loss: ', history.history['loss'])
            # print('val loss: ', history.history['val_loss'])
            expdata['acc'].append(max(history.history['val_accuracy']))

    print (expdata)

    df = pd.DataFrame.from_dict(expdata)

    df['network'], df['dataset'], df['lr'], df['batch_size'], df['epochs'], df['embed'], df['layer'], df['mode'], df['embed_mode'], df['jobid'] = network,  \
                                                                        dataset, lr, batch_size, epochs, embed, layer, mode, embed_mode, jobid

    pd.set_option('display.max_columns', None)
    print(df.head(5))

    # save the results of our experiment
    if(log_expdata):
        Path(path).mkdir(parents=True, exist_ok=True)
        if(not os.path.exists(path + 'expdata.csv')):
            df.to_csv(path + 'expdata.csv', mode='a', header=True)
        else:
            df.to_csv(path + 'expdata.csv', mode='a', header=False)
    
    # file = open('MLP_other_features_results_'+dataset+'.txt', 'a')
    
    # for trait_idx in range(full_targets.shape[1]):
    #     targets = full_targets[:, trait_idx]
    #     targets = tf.keras.utils.to_categorical(targets, num_classes=n_classes)

    #     kf = KFold(n_splits=10, shuffle=True, random_state=0)
    #     k = -1
    #     sum_res = 0
    #     for train_index, test_index in kf.split(data):
    #         X_train, X_test = data[train_index], data[test_index]
    #         y_train, y_test = targets[train_index], targets[test_index]
    #         model = tf.keras.models.Sequential()
    #         model.add(tf.keras.layers.Dense(128, input_shape=(data.shape[-1],), activation='relu'))
    #         model.add(tf.keras.layers.Dense(50, activation='relu'))
    #         model.add(tf.keras.layers.Dense(2, activation='softmax'))
    #         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #         saver = tf.keras.callbacks.ModelCheckpoint('model'+str(trait_idx)+".hdf5",save_best_only=True)
    #         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[saver])
    #         results = model.evaluate(X_test, y_test, batch_size=batch_size)
    #         print('Eval loss/accuracy:{}'.format(results))
    #         sum_res += results[1]
    #     file.write(str(trait_idx) +' : '+ str(sum_res/10)+'\n')
    # file.write('\n')
    # file.close()

