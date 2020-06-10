import pandas as pd
import utils

dataset_type, datafile, op_dir = utils.parse_args_metafeatures()
from collections import Counter
import readability
import re



def extract_readability_features(text):
    text = re.sub(r'\.', '.\n', text)
    text = re.sub(r'\?', '?\n', text)
    text = re.sub(r'!', '!\n', text)
    features = dict(readability.getmeasures(text, lang='en'))
    result = {}
    for d in features:
        result.update(features[d])
    result = pd.Series(result)
    return result


if __name__ == "__main__":
    count_df = pd.read_pickle(datafile)
    tmp = count_df["TEXT"].apply(lambda x: extract_readability_features(x))

    result = pd.concat([count_df['#AUTHID'], tmp], axis=1)
    output_file = op_dir + dataset_type + '_readability.csv'
    result.to_csv(output_file, index=False)
