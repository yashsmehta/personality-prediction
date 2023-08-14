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

import utils.gen_utils as utils
from utils.data_utils import MyMapDataset
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

start = time.time()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU found (", torch.cuda.get_device_name(torch.cuda.current_device()), ")")
    torch.cuda.set_device(torch.cuda.current_device())
    print("num device avail: ", torch.cuda.device_count())

else:
    DEVICE = torch.device("cpu")
    print("running on cpu")


def extract_bert_features(input_ids, mode, n_hl):
    """Extract bert embedding for each input."""
    if mode == "docbert":
        # print(input_ids.shape)
        tmphidden_features = []
        input_ids = input_ids.permute(1, 0, 2)

        for jj in range(input_ids.shape[0]):
            tmp = []
            if input_ids[jj][0][0] == 0:
                break

            bert_output = model(input_ids[jj])
            for ii in range(n_hl):
                if embed_mode == "mean":
                    tmp.append((bert_output[2][ii + 1].cpu().numpy()).mean(axis=1))
                elif embed_mode == "cls":
                    tmp.append(bert_output[2][ii + 1][:, 0, :].cpu().numpy())

            tmphidden_features.append(tmp)

        tmphidden_features = np.array(tmphidden_features)
        hidden_features.append(tmphidden_features.mean(axis=0))

    else:
        tmp = []
        bert_output = model(input_ids)
        # bert_output[2](this id gives all BERT outputs)[ii+1](which BERT layer)[:,0,:](taking the <CLS> output)

        for ii in range(n_hl):
            if embed_mode == "cls":
                tmp.append(bert_output[2][ii + 1][:, 0, :].cpu().numpy())
            elif embed_mode == "mean":
                tmp.append((bert_output[2][ii + 1].cpu().numpy()).mean(axis=1))

        hidden_features.append(np.array(tmp))
        return hidden_features


def get_model(embed):
    # * Model          | Tokenizer          | Pretrained weights shortcut
    # MODEL=(DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
    if embed == "bert-base":
        n_hl = 12
        hidden_dim = 768
        MODEL = (BertModel, BertTokenizer, "bert-base-uncased")

    elif embed == "bert-large":
        n_hl = 24
        hidden_dim = 1024
        MODEL = (BertModel, BertTokenizer, "bert-large-uncased")

    elif embed == "albert-base":
        n_hl = 12
        hidden_dim = 768
        MODEL = (AlbertModel, AlbertTokenizer, "albert-base-v2")

    elif embed == "albert-large":
        n_hl = 24
        hidden_dim = 1024
        MODEL = (AlbertModel, AlbertTokenizer, "albert-large-v2")

    model_class, tokenizer_class, pretrained_weights = MODEL

    # load the LM model and tokenizer from the HuggingFace Transformers library
    model = model_class.from_pretrained(
        pretrained_weights, output_hidden_states=True
    )  # output_attentions=False
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=True)

    return model, tokenizer, n_hl, hidden_dim


if __name__ == "__main__":
    # argument extractor
    (
        dataset,
        token_length,
        batch_size,
        embed,
        op_dir,
        mode,
        embed_mode,
    ) = utils.parse_args_extractor()
    print(
        "\n{} | {} | {} | {} | {}\n".format(
            dataset, embed, token_length, mode, embed_mode
        )
    )
    batch_size = int(32)
    model, tokenizer, n_hl, hidden_dim = get_model(embed)

    # create a class which can be passed to the pyTorch dataloader. responsible for returning tokenized and encoded values of the dataset
    # this class will have __getitem__(self,idx) function which will return input_ids and target values

    map_dataset = MyMapDataset(dataset, tokenizer, token_length, DEVICE, mode)

    data_loader = DataLoader(dataset=map_dataset, batch_size=batch_size, shuffle=False,)

    if DEVICE == torch.device("cuda"):
        model = model.cuda()
        # model.parameters() returns a generator obj
        # print('model loaded to gpu? ', next(model.parameters()).is_cuda)
        print(
            "\ngpu mem alloc: ", round(torch.cuda.memory_allocated() * 1e-9, 2), " GB"
        )

    print("starting to extract LM embeddings...")

    hidden_features = []
    all_targets = []
    all_author_ids = []

    # get bert embedding for each input
    for author_ids, input_ids, targets in data_loader:
        with torch.no_grad():
            all_targets.append(targets.cpu().numpy())
            all_author_ids.append(author_ids.cpu().numpy())
            extract_bert_features(input_ids, mode, n_hl)

    Path(op_dir).mkdir(parents=True, exist_ok=True)
    pkl_file_name = dataset + "-" + embed + "-" + embed_mode + "-" + mode + ".pkl"

    file = open(os.path.join(op_dir, pkl_file_name), "wb")
    pickle.dump(zip(all_author_ids, hidden_features, all_targets), file)
    file.close()

    # print(timedelta(seconds=int(time.time() - start)), end=' ')
    print("extracting embeddings for {} dataset: DONE!".format(dataset))
