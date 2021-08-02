from functools import reduce
import os
import pandas as pd
import numpy as np
import multiprocessing as mp


class MapReducer:
    def __init__(self, df):
        self.df = df
        self.counter = 0

    def mapper(self, group):
        gp_name, lst = group
        gp_df = (
            pd.DataFrame([self.df.loc[x] for x in lst], columns=self.df.columns)
            .sort_values("created_utc")
            .iloc[-100:]
        )
        res = " ||| ".join(gp_df.body.to_list())
        self.counter += 1
        print("author {} done".format(self.counter))
        return gp_name, res


# tmp.groupby('author')["proper_tokenized"].agg(sum)
def get_100_recent_posts(data_file):
    recent_df_path = os.path.dirname(data_file) + "/recent100.pkl"
    if os.path.isfile(recent_df_path):
        return pd.read_pickle(recent_df_path)
    else:
        print("extracting 100 most recent posts per author")
        multi_core = False
        reddits = pd.read_csv(data_file)
        map_reducer = MapReducer(reddits)
        groups = reddits.groupby("author").groups.items()
        results = []
        if multi_core:
            p = mp.Pool(mp.cpu_count())  # Data parallelism Object
            results = p.map(map_reducer.mapper, groups)
        else:
            for group in groups:
                results.append(map_reducer.mapper(group))
        x = pd.DataFrame(results, columns=["author", "text"]).set_index("author")
        print("#done#")
        pd.to_pickle(x, recent_df_path)
        return x


if __name__ == "__main__":
    get_100_recent_posts("filtered_reddit_ocean_full.csv")
