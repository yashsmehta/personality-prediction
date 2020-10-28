# Language Models and Automated Personality Prediction

This repository contains code for the paper [Bottom-Up and Top-Down: 
Predicting Personality with Psycholinguistic and Language Model Features](https://sentic.net/predicting-personality-with-psycholinguistic-and-language-model-features.pdf), published in **IEEE International Conference of Data Mining 2020**.

Here are a set of experiments written in tensorflow + pytorch to explore automated personality detection using Language Models on the Essays dataset (Big-Five personality labelled traits) and the Kaggle MBTI dataset.


## Installation

Pull this repository from GitLab via:

```bash
git clone git@gitlab.com:ml-automated-personality-detection/personality.git
```

See the requirements.txt for the list of dependent packages which can be installed via:

```bash
pip -r requirements.txt
```

## Usage
First run the LM extractor code which passes the dataset through the language model and stores the embeddings (of all layers) in a pickle file. Creating this 'new dataset' saves us a lot of compute time and allows effective searching of the hyperparameters for the finetuning network. Before running the code, create a pkl_data folder in the repo folder. All the arguments are optional and passing no arguments runs the extractor with the default values.

```bash
python LM_extractor.py -dataset_type 'essays' -token_length 512 -batch_size 32 -embed 'bert-base' -op_dir 'pkl_data'
```

Next run the finetuning network which is currently a MLP.

```bash
python finetuneNet.py
```

## Running Time

```bash
LM_extractor.py
```
On a RTX2080 GPU, the -embed 'bert-base' extractor takes about ~2m 30s and 'bert-large' takes about ~5m 30s

On a CPU, 'bert-base' extractor takes about ~25m

```bash
finetuneNet.py
```
On a RTX2080 GPU, running for 15 epochs (with no cross-validation) takes from 5s-60s, depending on the MLP architecture.

## Citation

If you find this repo useful for your research, please cite it using the following BibTex entry:

```
@inproceedings{mehtabottom,
  title={Bottom-Up and Top-Down: Predicting Personality with Psycholinguistic and Language Model Features},
  author={Mehta, Yash and Fatehi, Samin and Kazameini, Amirmohammad and Stachl, Clemens and Cambria, Erik and Eetemadi, Sauleh},
  booktitle={Proceedings of the International Conference of Data Mining},
  Organization = {IEEE},
  year={2020}}
}
```
