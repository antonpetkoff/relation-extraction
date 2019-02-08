from pandas.io.json import json_normalize
import json
import pandas as pd
import numpy as np


def json_to_df(path):
    with open(path) as json_file:
        json_data = json.load(json_file)

    df = json_normalize(json_data)
    return df


def load_data_set(path):
    return pd.read_csv(
        path,
        index_col=0,
        dtype=np.str,
        keep_default_na=False
    )
