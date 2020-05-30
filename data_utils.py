import numpy as np
import pandas as pd
import re
import csv

from torch.utils.data import DataLoader, Dataset
import torch
from transformers import *


def preprocess_text(sentence):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal (except I)
    sentence = re.sub(r"\s+[a-zA-HJ-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def load_essays_df(datafile):
    with open(datafile, "rt") as csvf:
        csvreader=csv.reader(csvf, delimiter=',',quotechar='"')
        first_line=True
        df = pd.DataFrame(columns=["user", "text", "EXT", "NEU", "AGR", "CON", "OPN"])
        for line in csvreader:
            if first_line:
                first_line=False
                continue
            
            text = line[1]

            df = df.append({"user": line[0],
                    "text":text,
                    "EXT":1 if line[2].lower()=='y' else 0,
                    "NEU":1 if line[3].lower()=='y' else 0,
                    "AGR":1 if line[4].lower()=='y' else 0,
                    "CON":1 if line[5].lower()=='y' else 0,
                    "OPN":1 if line[6].lower()=='y' else 0}, ignore_index=True)

    return df

def essays_embeddings(datafile, tokenizer, token_length):
    hidden_features=[]
    targets=[]
    token_len=[]
    input_ids = []

    df = load_essays_df(datafile)
    cnt=0
    for ind in df.index: 

        text = preprocess_text(df['text'][ind])
        tokens = tokenizer.tokenize(text)
        token_len.append(len(tokens))
        token_ids = tokenizer.encode(tokens, add_special_tokens=True, max_length = token_length, pad_to_max_length=True)
        if(cnt<10):
            print(tokens)
        
        input_ids.append(token_ids)
        targets.append(df['OPN'][ind])

        cnt+=1
    print(token_len)
    print(np.mean(token_len))
    return input_ids, targets

class MyMapDataset(Dataset):
  def __init__(self, dataset_type, datafile , tokenizer, token_length, DEVICE):
    if(dataset_type == 'essays'):
        input_ids, targets = essays_embeddings(datafile, tokenizer, token_length)
    
    
    input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
    targets = torch.from_numpy(np.array(targets)).long().to(DEVICE)

    self.input_ids=input_ids
    self.targets=targets
 
  def __len__(self):
      return len(self.targets)
 
  def __getitem__(self,idx):
      return(self.input_ids[idx], self.targets[idx])
