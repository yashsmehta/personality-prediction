import csv
import numpy as np
import argparse
from datetime import datetime, timedelta


def file_writer(results_file, meta_info, acc, loss_val, cv):
    lr, epochs, seed, embed, layer = meta_info
    params = [
        "EMBED ",
        embed,
        " LAYER ",
        str(layer),
        " LR ",
        str(lr),
        " SEED ",
        str(seed),
        " EPOCHS ",
        str(epochs),
    ]

    with open(results_file, "a") as csvFile:
        writer = csv.writer(csvFile)
        if cv == "0":
            writer.writerow(params)
        writer.writerow(["cv", cv])
        writer.writerow(["loss_val: ", str(loss_val)])
        writer.writerow(["acc_val: ", str(acc)])
        # writer.writerow(['loss_test: ', test_result[0]])
        # writer.writerow(['acc_test: ', test_result[2]])
        writer.writerow("")

        csvFile.flush()

    csvFile.close()
    return


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-inp_dir", type=str, default="pkl_data/")
    ap.add_argument("-dataset", type=str, default="essays")
    ap.add_argument("-lr", type=float, default=5e-4)
    ap.add_argument("-batch_size", type=int, default=32)
    ap.add_argument("-epochs", type=int, default=10)
    # ap.add_argument("-seed", type=int, default=np.random.randint(0,1000))
    ap.add_argument(
        "-log_expdata", type=str_to_bool, nargs="?", const=True, default=True
    )
    ap.add_argument("-embed", type=str, default="bert-base")
    ap.add_argument("-layer", type=str, default="11")
    ap.add_argument("-mode", type=str, default="512_head")
    ap.add_argument("-embed_mode", type=str, default="cls")
    ap.add_argument("-jobid", type=int, default=0)
    ap.add_argument("-save_model", type=str, default="no")
    args = ap.parse_args()
    return (
        args.inp_dir,
        args.dataset,
        args.lr,
        args.batch_size,
        args.epochs,
        args.log_expdata,
        args.embed,
        args.layer,
        args.mode,
        args.embed_mode,
        args.jobid,
        args.save_model,
    )


def parse_args_extractor():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dataset_type", type=str, default="essays")
    # ap.add_argument("-dataset_type", type=str, default='pandora')  # pandora example
    ap.add_argument("-token_length", type=int, default=512)
    # ap.add_argument("-datafile", type=str, default='data/pandora/')  # pandora example
    ap.add_argument("-batch_size", type=str, default=32)
    ap.add_argument("-embed", type=str, default="bert-base")
    ap.add_argument("-op_dir", type=str, default="pkl_data/")
    ap.add_argument("-mode", type=str, default="512_head")
    ap.add_argument("-embed_mode", type=str, default="cls")
    args = ap.parse_args()
    return (
        args.dataset_type,
        args.token_length,
        args.batch_size,
        args.embed,
        args.op_dir,
        args.mode,
        args.embed_mode,
    )


def parse_args_metafeatures():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dataset_type", type=str, default="essays")
    ap.add_argument(
        "-datafile", type=str, default="meta_features_data/essays_concept_count_final.p"
    )
    # ap.add_argument("-feature_type", type=str, default='readability')
    ap.add_argument(
        "-op_dir", type=str, default="../data/essays/psycholinguist_features/"
    )
    args = ap.parse_args()
    return args.dataset_type, args.datafile, args.op_dir


def parse_args_SHAP():
    ap = argparse.ArgumentParser()
    ap.add_argument("-mairesse", type=bool, default=1)
    ap.add_argument("-nrc", type=bool, default=1)
    ap.add_argument("-nrc_vad", type=bool, default=1)
    ap.add_argument("-affectivespace", type=bool, default=1)
    ap.add_argument("-hourglass", type=bool, default=1)
    ap.add_argument("-readability", type=bool, default=1)
    args = ap.parse_args()
    return (
        args.mairesse,
        args.nrc,
        args.nrc_vad,
        args.affectivespace,
        args.hourglass,
        args.readability,
    )


def parse_args_predictor():
    ap = argparse.ArgumentParser()
    ap.add_argument("-dataset_type", type=str, default="essays")
    ap.add_argument("-token_length", type=int, default=512)
    ap.add_argument("-batch_size", type=str, default=32)
    ap.add_argument("-embed", type=str, default="bert-base")
    ap.add_argument("-op_dir", type=str, default="pkl_data/")
    ap.add_argument("-mode", type=str, default="512_head")
    ap.add_argument("-embed_mode", type=str, default="cls")
    ap.add_argument("-finetune_model", type=str, default="mlp_lm")
    args = ap.parse_args()
    return (
        args.dataset_type,
        args.token_length,
        args.batch_size,
        args.embed,
        args.op_dir,
        args.mode,
        args.embed_mode,
        args.finetune_model,
    )
