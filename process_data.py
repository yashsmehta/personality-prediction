import numpy as np
import pandas as pd
import csv
import pickle
import re
import time
from datetime import timedelta

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

def load_df(datafile):
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
            break
    return df


datafile = 'data/essays.csv'
start=time.time()

if torch.cuda.is_available():        
    DEVICE = torch.device("cuda")
    print('running on GPU (', torch.cuda.get_device_name(torch.cuda.current_device()),')')
    torch.cuda.set_device(torch.cuda.current_device())
    print('num device avail: ', torch.cuda.device_count())

else:
    DEVICE = torch.device('cpu')
    print('running on cpu')
    

#* Model          | Tokenizer          | Pretrained weights shortcut
# MODEL=(DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
MODEL=(BertModel, BertTokenizer, 'bert-base-uncased')
n_hl=12

MAX_TOKENIZATION_LENGTH=512

model_class, tokenizer_class, pretrained_weights=MODEL

model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
                                # output_attentions=False,
                                # )
tokenizer=tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

model=model.cuda()

#* model.parameters() returns a generator obj
print('model loaded to gpu? ', next(model.parameters()).is_cuda)
print('gpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')


def dataset_embeddings(datafile):
    hidden_features=[]
    targets=[]
    token_len=[]
    input_ids = []

    df = load_df(datafile)

    for ind in df.index: 
        text = preprocess_text(df['text'][ind])
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(tokens, add_special_tokens=True, max_length = MAX_TOKENIZATION_LENGTH, pad_to_max_length=True)
        
        input_ids.append(token_ids)
        targets.append(df['OPN'][ind])
            
    return input_ids, targets

with torch.no_grad():
    input_ids, targets = dataset_embeddings(datafile)
    print(len(input_ids))
    input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
    targets = torch.from_numpy(np.array(targets)).long().to(DEVICE)

print(input_ids.shape)
print(targets.shape)

# bert_output = model(input_ids)

# print(targets)
# print(hidden_features)
# file = open('pkl data/imdb-test-'+pretrained_weights, 'wb')
# pickle.dump(zip(hidden_features, targets), file)
# file.close()

# print('gpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')
print(timedelta(seconds=int(time.time()-start)), end=' ')
# print('extracting embeddings for the test set: DONE!')
