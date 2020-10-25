import pandas as pd
import utils.gen_utils as utils

dataset_type, datafile, op_dir = utils.parse_args_metafeatures()
from collections import Counter


def extract_NRC_features(x, sentic_df):
    # tokens = re.sub('[^a-zA-Z]', ' ', x).split()
    tokens = x.split()
    tokens = Counter(tokens)
    df = pd.DataFrame.from_dict(tokens, orient='index', columns=['count'])
    merged_df = pd.merge(df, sentic_df, left_index=True, right_index=True)
    for col in merged_df.columns[1:]:
        merged_df[col] *= merged_df["count"]

    result = merged_df.sum()
    result /= result["count"]
    result = result.iloc[1:]
    return result


if __name__ == "__main__":
    count_df = pd.read_pickle(datafile)
    NRC_path = "meta_features_data/NRC-Emotion-Lexicon.xlsx"
    NRC_df = pd.read_excel(NRC_path, index_col=0)

    tmp = count_df["TEXT"].apply(lambda x: extract_NRC_features(x, NRC_df))

    result = pd.concat([count_df['#AUTHID'], tmp], axis=1)
    output_file = op_dir + dataset_type + '_nrc.csv'
    result.to_csv(output_file, index=False)
