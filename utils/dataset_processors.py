import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p
import math


def preprocess_text(sentence):
    # remove hyperlinks, hashtags, smileys, emojies
    sentence = p.clean(sentence)
    # Remove hyperlinks
    sentence = re.sub(r"http\S+", " ", sentence)
    # Remove punctuations and numbers
    # sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # sentence = re.sub('[^a-zA-Z.?!,]', ' ', sentence)
    # Single character removal (except I)
    # sentence = re.sub(r"\s+[a-zA-HJ-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub(r"\|\|\|", " ", sentence)

    return sentence


def load_essays_df(datafile):
    with open(datafile, "rt") as csvf:
        csvreader = csv.reader(csvf, delimiter=",", quotechar='"')
        first_line = True
        df = pd.DataFrame(
            columns=["user", "text", "token_len", "EXT", "NEU", "AGR", "CON", "OPN"]
        )
        for line in csvreader:
            if first_line:
                first_line = False
                continue

            text = line[1]
            new_row = pd.DataFrame(
                {
                    "user": [line[0]],
                    "text": [text],
                    "token_len": [0],
                    "EXT": [1 if line[2].lower() == "y" else 0],
                    "NEU": [1 if line[3].lower() == "y" else 0],
                    "AGR": [1 if line[4].lower() == "y" else 0],
                    "CON": [1 if line[5].lower() == "y" else 0],
                    "OPN": [1 if line[6].lower() == "y" else 0],
                }
            )
            df = pd.concat([df, new_row], ignore_index=True)

    print("EXT : ", df["EXT"].value_counts())
    print("NEU : ", df["NEU"].value_counts())
    print("AGR : ", df["AGR"].value_counts())
    print("CON : ", df["CON"].value_counts())
    print("OPN : ", df["OPN"].value_counts())

    return df


def essays_embeddings(datafile, tokenizer, token_length, mode):
    targets = []
    input_ids = []

    df = load_essays_df(datafile)
    cnt = 0

    # sorting all essays in ascending order of their length
    for ind in df.index:
        tokens = tokenizer.tokenize(df["text"][ind])
        df.at[ind, "token_len"] = len(tokens)

    df.sort_values(by=["token_len", "user"], inplace=True, ascending=True)
    tmp_df = df["user"]
    tmp_df.to_csv("data/essays/author_id_order.csv", index_label="order")
    print("Mean length of essay: ", df["token_len"].mean())

    for ii in range(len(df)):
        text = preprocess_text(df["text"][ii])
        tokens = tokenizer.tokenize(text)

        if mode == "normal" or mode == "512_head":
            input_ids.append(
                tokenizer.encode(
                    tokens,
                    add_special_tokens=True,
                    max_length=token_length,
                    pad_to_max_length=True,
                )
            )
        elif mode == "512_tail":
            input_ids.append(
                tokenizer.encode(
                    tokens[-(token_length - 2) :],
                    add_special_tokens=True,
                    max_length=token_length,
                    pad_to_max_length=True,
                )
            )
        elif mode == "256_head_tail":
            input_ids.append(
                tokenizer.encode(
                    tokens[: (token_length - 1)] + tokens[-(token_length - 1) :],
                    add_special_tokens=True,
                    max_length=token_length,
                    pad_to_max_length=True,
                )
            )

        elif mode == "docbert":
            docmax_len = 2048
            subdoc_len = 512
            max_subdoc_num = docmax_len // subdoc_len
            subdoc_tokens = [
                tokens[i : i + subdoc_len] for i in range(0, len(tokens), subdoc_len)
            ][:max_subdoc_num]
            # print(subdoc_tokens)
            token_ids = [
                tokenizer.encode(
                    x,
                    add_special_tokens=True,
                    max_length=token_length,
                    pad_to_max_length=True,
                )
                for x in subdoc_tokens
            ]
            # print(token_ids)
            token_ids = np.array(token_ids).astype(int)

            buffer_len = docmax_len // subdoc_len - token_ids.shape[0]
            # print(buffer_len)
            tmp = np.full(shape=(buffer_len, token_length), fill_value=0, dtype=int)
            token_ids = np.concatenate((token_ids, tmp), axis=0)

            input_ids.append(token_ids)

        targets.append(
            [df["EXT"][ii], df["NEU"][ii], df["AGR"][ii], df["CON"][ii], df["OPN"][ii]]
        )
        cnt += 1

    author_ids = np.array(df.index)
    print("loaded all input_ids and targets from the data file!")
    return author_ids, input_ids, targets


def load_Kaggle_df(datafile):
    with open(datafile, "rt", encoding="utf-8") as csvf:
        csvreader = csv.reader(csvf, delimiter=",", quotechar='"')
        first_line = True
        df = pd.DataFrame(columns=["user", "text", "E", "N", "F", "J"])
        for line in csvreader:
            if first_line:
                first_line = False
                continue

            text = line[1]

            df = df.append(
                {
                    "user": line[3],
                    "text": text,
                    "E": 1 if line[0][0] == "E" else 0,
                    "N": 1 if line[0][1] == "N" else 0,
                    "F": 1 if line[0][2] == "F" else 0,
                    "J": 1 if line[0][3] == "J" else 0,
                },
                ignore_index=True,
            )

    print("E : ", df["E"].value_counts())
    print("N : ", df["N"].value_counts())
    print("F : ", df["F"].value_counts())
    print("J : ", df["J"].value_counts())

    return df


def kaggle_embeddings(datafile, tokenizer, token_length):
    hidden_features = []
    targets = []
    token_len = []
    input_ids = []
    author_ids = []

    df = load_Kaggle_df(datafile)
    cnt = 0
    for ind in df.index:

        text = preprocess_text(df["text"][ind])
        tokens = tokenizer.tokenize(text)
        token_len.append(len(tokens))
        token_ids = tokenizer.encode(
            tokens,
            add_special_tokens=True,
            max_length=token_length,
            pad_to_max_length=True,
        )
        if cnt < 10:
            print(tokens[:10])

        input_ids.append(token_ids)
        targets.append([df["E"][ind], df["N"][ind], df["F"][ind], df["J"][ind]])
        author_ids.append(int(df["user"][ind]))
        cnt += 1

    print("average length : ", int(np.mean(token_len)))
    author_ids = np.array(author_ids)

    return author_ids, input_ids, targets
