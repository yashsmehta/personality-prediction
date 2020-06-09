import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import csv
import pickle
import time
from datetime import timedelta
import shap
import pandas as pd

import utils

inp_dir, dataset_type, network, lr, batch_size, epochs, seed, write_file, embed, layer, mode, embed_mode = utils.parse_args()
n_classes = 2
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

start = time.time()
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.preprocessing import text
import shap
import re
import preprocessor as p

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

            df = df.append({#"user": line[0],
                            "text": text,
                            "EXT": 1 if line[2].lower() == 'y' else 0,
                            "NEU": 1 if line[3].lower() == 'y' else 0,
                            "AGR": 1 if line[4].lower() == 'y' else 0,
                            "CON": 1 if line[5].lower() == 'y' else 0,
                            "OPN": 1 if line[6].lower() == 'y' else 0}, ignore_index=True)


    return df

class TextPreprocessor(object):
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self._tokenizer = None
    def create_tokenizer(self, text_list):
        tokenizer = text.Tokenizer(num_words = self._vocab_size)
        tokenizer.fit_on_texts(text_list)
        self._tokenizer = tokenizer

if __name__ == "__main__":
    dump_data = load_essays_df('data/essays/essays.csv')
    labels_list = ['EXT', 'NEU', 'AGR', 'CON', 'OPN']
    if (embed == 'bert-base'):
        pretrained_weights = 'bert-base-uncased'
        n_hl = 12
        hidden_dim = 768

    elif (embed == 'bert-large'):
        pretrained_weights = 'bert-large-uncased'
        n_hl = 24
        hidden_dim = 1024

    file = open('../'+inp_dir + dataset_type + '-' + embed + '-' + embed_mode + '-' + mode+ '.pkl', 'rb')
    data = pickle.load(file)
    data_x, data_y = list(zip(*data))

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

    data = np.array(inputs)
    full_targets = np.array(targets)
    VOCAB_SIZE = 2000
    train_size = int(len(data) * .8)
    X_train = data[: train_size]
    X_test = data[train_size: ]
    train_post = dump_data['text'].values[: train_size]
    test_post = dump_data['text'].values[train_size: ]
    processor = TextPreprocessor(VOCAB_SIZE)
    processor.create_tokenizer(train_post)
    for trait_idx in range(full_targets.shape[1]):
        targets = full_targets[:, trait_idx]
        y_train = targets[: train_size]
        y_test = targets[train_size: ]
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(50, input_shape = (hidden_dim,), activation='relu'))
        model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, validation_split=0.1)
        print('Eval loss/accuracy:{}'.format(model.evaluate(X_test, y_test, batch_size = batch_size)))

        attrib_data = X_train
        explainer = shap.DeepExplainer(model, attrib_data)
        num_explanations = 20
        shap_vals = explainer.shap_values(X_test[:num_explanations])

        words = processor._tokenizer.word_index
        word_lookup = list()
        for i in words.keys():
          word_lookup.append(i)

        word_lookup = [''] + word_lookup
        shap.summary_plot(shap_vals, show=False, class_names=[labels_list[trait_idx]+' 0', labels_list[trait_idx]+' 1'])
        import matplotlib.pyplot as plt
        plt.savefig(labels_list[trait_idx]+'-'+embed_mode+'-'+mode+".png")
        plt.clf()
