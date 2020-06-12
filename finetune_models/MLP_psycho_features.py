import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import csv
import pickle
import time
import pandas as pd
import utils.gen_utils as utils
import tensorflow as tf
import shap
import re
import preprocessor as p
from sklearn.model_selection import KFold
from scipy.io import arff

inp_dir, dataset_type, network, lr, batch_size, epochs, seed, write_file, embed, layer, mode, embed_mode = utils.parse_args()
mairesse, nrc, nrc_vad, affectivespace, hourglass, readability = utils.parse_args_SHAP()
n_classes = 2
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
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


def load_features(dir, dataset_type):
    idx = 'id'
    if dataset_type == 'kaggle':
        drop_cols = ['BROWN-FREQ numeric', 'K-F-FREQ numeric', 'K-F-NCATS numeric', 'K-F-NSAMP numeric', 'T-L-FREQ numeric', 'Extraversion numeric'
                  , '\'Emotional stability\' numeric', 'Agreeableness numeric', 'Conscientiousness numeric', '\'Openness to experience\' numeric']
        mairesse = read_and_process(dir + dataset_type + '_mairesse_labeled.arff')
        mairesse = mairesse.drop(drop_cols, axis=1)
    elif dataset_type == 'essays':
        mairesse = pd.read_csv(dir + dataset_type + '_mairesse_labeled.csv')
    # mairesse = mairesse.set_index(mairesse.columns[0])
    nrc = pd.read_csv(dir + dataset_type + '_nrc.csv').set_index([idx])
    # nrc = nrc.sort_values(by=['id'])
    # nrc = nrc.drop(['id'], axis=1)
    nrc_vad = pd.read_csv(dir + dataset_type + '_nrc_vad.csv').set_index([idx])
    # nrc_vad = nrc_vad.sort_values(by=['id'])
    # nrc_vad = nrc_vad.drop(['id'], axis=1)
    # affectivespace = pd.read_csv(dir + 'essays_affectivespace.csv').set_index(['#AUTHID'])
    # hourglass = pd.read_csv(dir + dataset_type + '_hourglass.csv').set_index([idx])
    readability = pd.read_csv(dir + dataset_type + '_readability.csv').set_index([idx])

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

def get_psycholinguist_data(dump_data, dataset_type):
    features = load_features('../data/'+dataset_type+'/psycholinguist_features/', dataset_type)
    feature_flags = [nrc, nrc_vad, readability, mairesse]
    first = 1
    for feature, feature_flag in zip(features, feature_flags):
        if feature_flag:
            if first:
                df = feature
                first = 0
            else:
                df = pd.merge(df, feature, left_index=True, right_index=True)
    if dataset_type == 'essays':
        labels = dump_data[['user', 'EXT', 'NEU', 'AGR', 'CON', 'OPN']]
    if dataset_type == 'kaggle':
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
    if dataset_type == 'essays':
        dump_data = load_essays_df('../data/essays/essays.csv')
        labels_list = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']
    elif dataset_type == 'kaggle':
        dump_data = load_Kaggle_df('../data/kaggle/kaggle.csv')
        labels_list = ['E', 'N', 'F', 'J']
    data, full_targets, feature_names = get_psycholinguist_data(dump_data, dataset_type)
    file = open('MLP_other_features_results_'+dataset_type+'.txt', 'a')
    for trait_idx in range(full_targets.shape[1]):
        targets = full_targets[:, trait_idx]
        targets = tf.keras.utils.to_categorical(targets, num_classes=n_classes)

        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        k = -1
        sum_res = 0
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = targets[train_index], targets[test_index]
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(128, input_shape=(data.shape[-1],), activation='relu'))
            model.add(tf.keras.layers.Dense(50, activation='relu'))
            model.add(tf.keras.layers.Dense(2, activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            saver = tf.keras.callbacks.ModelCheckpoint('model'+str(trait_idx)+".hdf5",save_best_only=True)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[saver])
            results = model.evaluate(X_test, y_test, batch_size=batch_size)
            print('Eval loss/accuracy:{}'.format(results))
            sum_res += results[1]
        file.write(str(trait_idx) +' : '+ str(sum_res/10)+'\n')
    file.write('\n')
    file.close()

