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
import utils

start=time.time()
#argument extractor
dataset_type, token_length, datafile, batch_size, embed, op_dir = utils.parse_args_extractor()

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
if (embed=='bert-base'):
    n_hl=12; MODEL=(BertModel, BertTokenizer, 'bert-base-uncased')

elif (embed=='bert-large'):
    n_hl=24; MODEL=(BertModel, BertTokenizer, 'bert-large-uncased')

elif (embed=='albert-base'):
    n_hl=12; MODEL=(AlbertModel, AlbertTokenizer, 'albert-base-v2')

elif (embed=='albert-large'):
    n_hl=24; MODEL=(AlbertModel, AlbertTokenizer, 'albert-large-v2')

model_class, tokenizer_class, pretrained_weights=MODEL

#load the LM model and tokenizer from the HuggingFace Transformeres library
model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True) # output_attentions=False
tokenizer=tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

#create a class which can be passed to the pyTorch dataloader. responsible for returning tokenized and encoded values of the Essays dataset
#this class will have __getitem__(self,idx) function which will return input_ids and target values
#currently it is just returning the targets for the 'OPN' trait - need to generalize this
map_dataset = MyMapDataset(dataset_type, datafile, tokenizer, token_length, DEVICE)

data_loader = DataLoader(dataset=map_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                        )

if(DEVICE == torch.device("cuda")):
    model=model.cuda()
    # model.parameters() returns a generator obj
    # print('model loaded to gpu? ', next(model.parameters()).is_cuda)
    print('\ngpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')

print('starting to extract LM embeddings...')

hidden_features=[]
all_targets=[]
j = 0
for input_ids, targets in data_loader:
    with torch.no_grad():
        all_targets.append(targets.cpu().numpy())        
        #get the LM embeddings
        bert_output = model(input_ids)
        
        # bert_output[2](this id gives all BERT outputs)[ii+1](which BERT layer)[:,0,:](taking the <CLS> output)
        tmp=[]
        for ii in range(n_hl):
            tmp.append(bert_output[2][ii+1][:,0,:].cpu().numpy())
        
        hidden_features.append(np.array(tmp))
        j+=1


#storing the embeddings into a pickle file
# file = open(op_dir+dataset_type+'-'+embed+'.pkl', 'wb')
# pickle.dump(zip(hidden_features, all_targets), file)
# file.close()

print(timedelta(seconds=int(time.time()-start)), end=' ')
print('extracting embeddings for '+dataset_type+' dataset: DONE!')


print(j)
