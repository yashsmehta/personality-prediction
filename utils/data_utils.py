import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import *
import math

from utils.author_100recent import get_100_recent_posts
import utils.dataset_processors as dataset_processors

def preprocess_text(sentence):
    # remove hyperlinks, hashtags, smileys, emojies
    sentence = p.clean(sentence)
    # Remove hyperlinks
    sentence = re.sub(r'http\S+', ' ', sentence)
    # Remove punctuations and numbers
    # sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # sentence = re.sub('[^a-zA-Z.?!,]', ' ', sentence)
    # # Single character removal (except I)
    # sentence = re.sub(r"\s+[a-zA-HJ-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


class MyMapDataset(Dataset):
    def __init__(self, dataset_type, datafile, tokenizer, token_length, DEVICE, mode):
        if dataset_type == 'essays':
            author_ids, input_ids, targets = dataset_processors.essays_embeddings(datafile, tokenizer, token_length, mode)
        elif dataset_type == 'kaggle':
            author_ids, input_ids, targets = dataset_processors.kaggle_embeddings(datafile, tokenizer, token_length)
        elif dataset_type == 'pandora':
            author_ids, input_ids, targets = dataset_processors.pandora_embeddings(datafile, tokenizer, token_length)

        author_ids = torch.from_numpy(np.array(author_ids))
        input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
        targets = torch.from_numpy(np.array(targets))

        if dataset_type == 'pandora':
            targets = targets.float().to(DEVICE)
        else:
            targets = targets.long().to(DEVICE)

        self.author_ids = author_ids
        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.author_ids[idx], self.input_ids[idx], self.targets[idx])
