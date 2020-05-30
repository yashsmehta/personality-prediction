# Language Models and Automated Personality Prediction
This repository is a set of experiments written in tensorflow + pytorch to explore automated personality detection using BERT on the Essays dataset which contains the Big-Five personality labelled traits


## Installation

Pull the this package from GitLab via:

```bash
git clone git@gitlab.com:yashsmehta/personality.git
```

See the requirements.txt for the list of dependent packages

```bash
pip -r requirements.txt
```

## Usage
First run LMextractor.py, which passes the dataset through the language model and then stores the embeddings in a pickle file. Creating this 'new dataset' saves us a lot of compute time and allows effective searching of the hyperparameters for the finetuning network. Before running the code, create a pkl_data folder in the repo folder.

```bash
python LM_extractor.py -dataset_type 'essays' -token_length 512 -datafile 'data/essays.csv' -batch_size 32 -embed 'bert-base' -op_dir 'pkl_data'
```
All are optional arguments. Passing no arguments runs it with default values.


```bash
python finetuneNet.py 
```
