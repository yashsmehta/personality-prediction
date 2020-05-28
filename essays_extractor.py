import pickle
import time
from datetime import timedelta

import torch
from transformers import *
import numpy as np
from data_utils import IMDBDataset
from torch.utils.data import DataLoader

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=torch.float32).cuda()

    return zeros.scatter(scatter_dim, y_tensor, 1)

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
MODEL=(BertModel, BertTokenizer, 'bert-large-uncased')
n_hl=24

MAX_TOKENIZATION_LENGTH=512
n_classes=2

model_class, tokenizer_class, pretrained_weights=MODEL

model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
                                # output_attentions=False,
                                # )
tokenizer=tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

#* faster to load a pre-saved Bert model

# model = model_class.from_pretrained('saved_models/')  # re-load
# tokenizer = tokenizer_class.from_pretrained('saved_models/')

start=time.time()
data_url='../data/IMDB/'
batch_size=32

train_dataset = IMDBDataset(input_directory=data_url+'train',
                            tokenizer=tokenizer,
                            max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                            apply_cleaning=False,
                            device=DEVICE)

test_dataset = IMDBDataset(input_directory=data_url+'test',
                           tokenizer=tokenizer,
                           max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                           apply_cleaning=False,
                           device=DEVICE)

print('time to load dataset: ')
print(timedelta(seconds=int(time.time()-start)))
start=time.time()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                        )

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         )
start=time.time()

model=model.cuda()
#* model.parameters() returns a generator obj
print('model loaded to gpu? ', next(model.parameters()).is_cuda)
print('gpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')

hidden_features=[]
all_targets=[]
for sentences, targets in train_loader:
    with torch.no_grad():
        targets=_to_one_hot(targets, num_classes=2).cpu().numpy()
        all_targets.append(targets)
        bert_output = model(sentences)
        
        last_features=bert_output[0][:,0,:]
        tmp=[]
        for ii in range(n_hl):
            tmp.append(bert_output[2][ii+1][:,0,:].cpu().numpy())
        
        hidden_features.append(np.array(tmp))

file = open('pkl data/imdb-train-'+pretrained_weights, 'wb')
pickle.dump(zip(hidden_features, all_targets), file)
file.close()

print('gpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')
print(timedelta(seconds=int(time.time()-start)), end=' ')
print('extracting embeddings for the train set: DONE!')

hidden_features=[]
all_targets=[]
for sentences, targets in test_loader:
    with torch.no_grad():
        targets=_to_one_hot(targets, num_classes=2).cpu().numpy()
        all_targets.append(targets)
        bert_output = model(sentences)
        
        last_features=bert_output[0][:,0,:]
        tmp=[]
        for ii in range(n_hl):
            tmp.append(bert_output[2][ii+1][:,0,:].cpu().numpy())
        
        hidden_features.append(np.array(tmp))

file = open('pkl data/imdb-test-'+pretrained_weights, 'wb')
pickle.dump(zip(hidden_features, all_targets), file)
file.close()

print('gpu mem alloc: ', round(torch.cuda.memory_allocated()*1e-9, 2), ' GB')
print(timedelta(seconds=int(time.time()-start)), end=' ')
print('extracting embeddings for the test set: DONE!')
