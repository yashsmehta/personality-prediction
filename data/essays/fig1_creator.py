import nltk
import seaborn as sns
import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt


def simple_tokenize(essays):
    essays['token_num'] = essays['TEXT'].apply(
        lambda x: len([word for word in nltk.word_tokenize(x) if word.isalnum()]))
    return essays


def bert_tokenize(essays):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    essays['bert_token_num'] = essays['TEXT'].apply(lambda x: len(tokenizer.tokenize(x)))
    return essays


def plot_data(essays):
    fig, ax = plt.subplots()
    sns.set()
    sns.set_palette(palette="muted")
    for a, b in zip(['token_num', 'bert_token_num'], ['Words', "BERT Tokens"]):
        sns.distplot(essays[a], ax=ax, label=b)
        plt.legend()
    ax.set(xlim=(0, 2000))
    ax.set(xlabel='Count')
    return ax


if __name__ == "__main__":
    # essays = pd.read_csv('essays.csv')
    essays = pd.read_pickle("token_count.pkl")
    fig = plot_data(essays)
    plt.show()
