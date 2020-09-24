import numpy as np
import pandas as pd
import re
import preprocessor as p
from scipy.io import arff

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
        drop_cols = ['BROWN-FREQ numeric', 'K-F-FREQ numeric', 'K-F-NCATS numeric', 'K-F-NSAMP numeric',
                     'T-L-FREQ numeric', 'Extraversion numeric'
            , '\'Emotional stability\' numeric', 'Agreeableness numeric', 'Conscientiousness numeric',
                     '\'Openness to experience\' numeric']
        mairesse = read_and_process(dir + dataset + '_mairesse_labeled.arff')
        mairesse = mairesse.drop(drop_cols, axis=1)
    elif dataset == 'essays':
        idx = '#AUTHID'
        mairesse = pd.read_csv(dir + dataset + '_mairesse_labeled.csv')
        mairesse = mairesse.set_index(mairesse.columns[0])
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


def get_psycholinguist_data(dump_data, dataset, feature_flags):
    features = load_features('data/' + dataset + '/psycholinguist_features/', dataset)

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
    if dataset == 'kaggle':
        labels.index = pd.to_numeric(labels.index, errors='coerce')
        df.index = pd.to_numeric(df.index, errors='coerce')
    merged = pd.merge(df, labels, left_index=True, right_index=True).fillna(0)
    label_size = labels.shape[1]
    data = merged[merged.columns[:(-1*label_size)]].values
    full_targets = merged[merged.columns[(-1*label_size):]].values
    feature_names = merged.columns
    return data, full_targets, feature_names, merged
