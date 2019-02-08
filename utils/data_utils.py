from pandas.io.json import json_normalize
import json


def json_to_df(path):
    with open(path) as json_file:
        json_data = json.load(json_file)

    df = json_normalize(json_data)
    return df
