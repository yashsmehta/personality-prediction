import csv
import numpy as np
import argparse
from datetime import datetime, timedelta


def file_writer(results_file, meta_info, acc, loss_val):
    lr, epochs, seed, embed, layer = meta_info
    params = ["EMBED ", embed, " LAYER ", str(layer), " LR ", str(lr), " SEED ", str(seed), " EPOCHS ", str(epochs)]

    with open(results_file, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        writer.writerow(loss_val)
        writer.writerow(acc)
        writer.writerow("")

        csvFile.flush()

    csvFile.close()
    return


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-inp_dir", type=str, default='pkl_data/')
    ap.add_argument("-dataset_type", type=str, default='essays')
    ap.add_argument("-network", type=str, default='fc')
    ap.add_argument("-lr", type=float, default=1e-3)
    ap.add_argument("-batch_size", type=int, default=32)
    ap.add_argument("-epochs", type=int, default=4)
    # ap.add_argument("-seed", type=int, default=np.random.randint(0,1000))
    ap.add_argument("-seed", type=int, default=0)
    ap.add_argument('-write_file', type=str_to_bool, nargs='?', const=True, default=True)
    ap.add_argument("-embed", type=str, default='bert-base')
    ap.add_argument("-layer", type=str, default='11')
    args = ap.parse_args()
    return args.inp_dir, args.dataset_type, args.network, args.lr, args.batch_size, args.epochs, args.seed, args.write_file, args.embed, args.layer


def parse_args_extractor():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dataset_type", type=str, default='essays')
    # ap.add_argument("-dataset_type", type=str, default='pandora')  # pandora example
    ap.add_argument("-token_length", type=int, default=512)
    ap.add_argument("-datafile", type=str, default='data/essays/essays.csv')
    # ap.add_argument("-datafile", type=str, default='data/pandora/')  # pandora example
    ap.add_argument('-batch_size', type=str, default=32)
    ap.add_argument("-embed", type=str, default='bert-base')
    ap.add_argument("-op_dir", type=str, default='pkl_data/')
    ap.add_argument("-mode", type=str, default='512_tail')
    args = ap.parse_args()
    return args.dataset_type, args.token_length, args.datafile, args.batch_size, args.embed, args.op_dir, args.mode


def parse_args_metafeatures():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dataset_type", type=str, default='essays')
    ap.add_argument("-datafile", type=str, default='meta_features_data/essays_concept_count_final.p')
    ap.add_argument("-feature_type", type=str, default='hourglass')
    ap.add_argument("-op_dir", type=str, default='data/')
    args = ap.parse_args()
    return args.dataset_type, args.datafile, args.feature_type, args.op_dir

