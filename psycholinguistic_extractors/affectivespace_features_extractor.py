import pandas as pd
import utils.gen_utils as utils

dataset_type, datafile, op_dir = utils.parse_args_metafeatures()


def extract_sentic_features(x, sentic_df):
    df = pd.DataFrame(x, columns=["concept", "count"]).set_index("concept")
    merged_df = pd.merge(df, sentic_df, left_index=True, right_index=True)
    for col in merged_df.columns[1:]:
        merged_df[col] *= merged_df["count"]

    result = merged_df.sum()
    result /= result["count"]
    result = result.iloc[1:]
    return result


if __name__ == "__main__":
    count_df = pd.read_pickle(datafile)
    sentic_path = "meta_features_data/affectivespace.csv"
    sentic_df = pd.read_csv(sentic_path, header=None)
    sentic_df = sentic_df.set_index(sentic_df.columns[0])

    tmp = count_df["concept_count"].apply(
        lambda x: extract_sentic_features(x, sentic_df)
    )

    result = pd.concat([count_df["#AUTHID"], tmp], axis=1)
    output_file = op_dir + dataset_type + "_affectivespace.csv"
    result.to_csv(output_file, index=False)
