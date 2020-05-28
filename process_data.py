import numpy as np
import pandas as pd
import csv
import pickle
import time
from datetime import timedelta

import torch
from transformers import *


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

hidden_features=[]
all_targets=[]

df = load_df(datafile)

for ind in df.index: 

    with torch.no_grad():
        all_targets.append(df['OPN'][ind])
        bert_output = model(df['text'][ind])
        
        last_features=bert_output[0][:,0,:]
        tmp=[]
        for ii in range(n_hl):
            tmp.append(bert_output[2][ii+1][:,0,:].cpu().numpy())
        
        hidden_features.append(np.array(tmp))

