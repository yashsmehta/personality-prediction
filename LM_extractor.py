import numpy as np
import pandas as pd
import csv
import pickle
import re
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import *

from data_utils import MyMapDataset

start=time.time()

if torch.cuda.is_available():        
    DEVICE = torch.device("cuda")
    print('GPU found (', torch.cuda.get_device_name(torch.cuda.current_device()),')')
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
datafile = 'data/essays.csv'
dataset_type = 'essays'


model_class, tokenizer_class, pretrained_weights=MODEL

model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True) # output_attentions=False
tokenizer=tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

batch_size = 32

map_dataset = MyMapDataset(dataset_type, datafile, tokenizer, MAX_TOKENIZATION_LENGTH, DEVICE)

data_loader = DataLoader(dataset=map_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                        )

model=model.cuda()

#* model.parameters() returns a generator obj
# print('model loaded to gpu? ', next(model.parameters()).is_cuda)
print('\ngpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')
print('starting to extract LM embeddings...')

hidden_features=[]
all_targets=[]
for input_ids, targets in data_loader:
    with torch.no_grad():
        all_targets.append(targets)        
        bert_output = model(input_ids)
        
        tmp=[]
        for ii in range(n_hl):
            tmp.append(bert_output[2][ii+1][:,0,:].cpu().numpy())
        
        hidden_features.append(np.array(tmp))

file = open('pkl_data/essays'+pretrained_weights+'.pkl', 'wb')
pickle.dump(zip(hidden_features, all_targets), file)
file.close()

print('gpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')
print(timedelta(seconds=int(time.time()-start)), end=' ')
print('extracting embeddings for Essays dataset: DONE!')
