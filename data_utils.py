import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import *
import math

# note: might not be the best preprocessor since it completely removes all punctuations. Since most of the essays are exceeding 512 words, we are currently using this.
# need to experiment with other ones
from data.pandora.author_100recent import get_100_recent_posts


def preprocess_text(sentence):
    # remove hyperlinks, hashtags, smileys, emojies
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

            df = df.append({"user": line[0],
                            "text": text,
                            "EXT": 1 if line[2].lower() == 'y' else 0,
                            "NEU": 1 if line[3].lower() == 'y' else 0,
                            "AGR": 1 if line[4].lower() == 'y' else 0,
                            "CON": 1 if line[5].lower() == 'y' else 0,
                            "OPN": 1 if line[6].lower() == 'y' else 0}, ignore_index=True)

    print('EXT : ', df['EXT'].value_counts())
    print('NEU : ', df['NEU'].value_counts())
    print('AGR : ', df['AGR'].value_counts())
    print('CON : ', df['CON'].value_counts())
    print('OPN : ', df['OPN'].value_counts())

    return df


def essays_embeddings(datafile, tokenizer, token_length, mode):
    hidden_features = []
    targets = []
    token_len = []
    input_ids = []

    df = load_essays_df(datafile)
    cnt = 0
    num_subdocs = []
    for ind in df.index:
        text = preprocess_text(df['text'][ind])
        tokens = tokenizer.tokenize(text)
        token_len.append(len(tokens))
        if mode == 'normal' or mode == '512_head':
            token_ids = tokenizer.encode(tokens, add_special_tokens=True, max_length=token_length, pad_to_max_length=True)
        elif mode == '512_tail':
            token_ids = tokenizer.encode(tokens[-(token_length-2):], add_special_tokens=True, max_length=token_length, pad_to_max_length=True)
        elif mode == '256_head_tail':
            token_ids = tokenizer.encode(tokens[:(token_length-1)]+tokens[-(token_length-1):], add_special_tokens=True, max_length=token_length, pad_to_max_length=True)
        if mode != 'docbert':
            input_ids.append(token_ids)
        elif mode == 'docbert':
            bound = 200
            num_pieces = math.ceil(len(tokens) / bound)
            tokens_list = [tokens[i * bound:(i + 1) * bound] for i in range(num_pieces-1)]
            tokens_list.append(tokens[-(len(tokens) - (num_pieces-1) * bound):])
            token_ids = [tokenizer.encode(x, add_special_tokens=True, max_length=token_length, pad_to_max_length=True)
                         for x in tokens_list]
            num_subdocs.append(num_pieces)
            for subdoc in token_ids:
                input_ids.append(subdoc)
        if (cnt < 10):
            print(tokens)
        targets.append([df['EXT'][ind], df['NEU'][ind], df['AGR'][ind], df['CON'][ind], df['OPN'][ind]])

        cnt += 1
    if mode == 'docbert':
        file = open('num_sobdocuments_'+str(bound)+'.txt', 'w')
        file.write(str(num_subdocs))
        file.close()
    print('token lengths : ', token_len)
    print('average length : ', int(np.mean(token_len)))
    return input_ids, targets


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


def kaggle_embeddings(datafile, tokenizer, token_length):
    hidden_features = []
    targets = []
    token_len = []
    input_ids = []

    df = load_Kaggle_df(datafile)
    cnt = 0
    for ind in df.index:

        text = preprocess_text(df['text'][ind])
        tokens = tokenizer.tokenize(text)
        token_len.append(len(tokens))
        token_ids = tokenizer.encode(tokens, add_special_tokens=True, max_length=token_length, pad_to_max_length=True)
        if (cnt < 10):
            print(tokens)

        input_ids.append(token_ids)
        targets.append([df['E'][ind], df['N'][ind], df['F'][ind], df['J'][ind]])

        cnt += 1
    print('token lengths : ', token_len)
    print('average length : ', int(np.mean(token_len)))
    return input_ids, targets


def load_pandora_df(datafile):
    # load posts_df in a proper [author,100 most recent posts] format
    posts_df = get_100_recent_posts(datafile + "all_comments_since_2015.csv")
    # load profiles
    profiles_df = pd.read_csv(datafile + "author_profiles.csv").set_index('author')
    merged_df = posts_df.join(profiles_df)
    merged_df.extraversion /= 100
    merged_df.neuroticism /= 100
    merged_df.agreeableness /= 100
    merged_df.conscientiousness /= 100
    merged_df.openness /= 100

    print('EXT : ', merged_df.extraversion.describe())
    print('NEU : ', merged_df.neuroticism.describe())
    print('AGR : ', merged_df.agreeableness.describe())
    print('CON : ', merged_df.conscientiousness.describe())
    print('OPN : ', merged_df.openness.describe())

    print('Introverted : ', merged_df.introverted.value_counts())
    print('intuitive : ', merged_df.intuitive.value_counts())
    print('thinking : ', merged_df.thinking.value_counts())
    print('perceiving : ', merged_df.perceiving.value_counts())

    return merged_df


def pandora_embeddings(datafile, tokenizer, token_length):
    hidden_features = []
    targets = []
    token_len = []
    input_ids = []

    df = load_pandora_df(datafile)
    cnt = 0
    for ind in df.index:
        text = preprocess_text(df['text'][ind])
        tokens = tokenizer.tokenize(text)
        token_len.append(len(tokens))
        token_ids = tokenizer.encode(tokens, add_special_tokens=True, max_length=token_length, pad_to_max_length=True)
        if (cnt < 10):
            print(tokens)

        input_ids.append(token_ids)
        # We first add OCEAN traits, then MBTI traits
        targets.append([df.extraversion[ind], df.neuroticism[ind], df.agreeableness[ind], df.conscientiousness[ind],
                        df.openness[ind],
                        df.introverted[ind], df.intuitive[ind], df.thinking[ind],
                        df.perceiving[ind]])

        cnt += 1
    print('token lengths : ', token_len)
    print('average length : ', int(np.mean(token_len)))
    return input_ids, targets


class MyMapDataset(Dataset):
    def __init__(self, dataset_type, datafile, tokenizer, token_length, DEVICE):
        if dataset_type == 'essays':
            input_ids, targets = essays_embeddings(datafile, tokenizer, token_length)
        elif dataset_type == 'kaggle':
            input_ids, targets = kaggle_embeddings(datafile, tokenizer, token_length)
        elif dataset_type == 'pandora':
            input_ids, targets = pandora_embeddings(datafile, tokenizer, token_length)

        input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
        targets = torch.from_numpy(np.array(targets))
        if dataset_type == 'pandora':
            targets = targets.float().to(DEVICE)
        else:
            targets = targets.long().to(DEVICE)

        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.targets[idx])
