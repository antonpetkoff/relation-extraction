from pprint import pprint
from pandas.io.json import json_normalize
import json
import numpy as np
import matplotlib.pyplot as plt


def json_to_df(path):
    with open(path) as json_file:
        json_data = json.load(json_file)

    df = json_normalize(json_data)
    return df


def analyze_relation_distribution(df):
    unique_relation_labels = np.unique(df['relation'])
    relation_distribution = {
        relation: df[df.relation == relation]['relation'].count()
        for relation in unique_relation_labels
    }
    data_size = len(df)
    relation_shares = [
        relation_size / data_size
        for relation_size in relation_distribution.values()
    ]
    sorted_relation_distribution = dict(sorted(relation_distribution.items(), key=lambda kv: kv[1]))

    print('Relations in alphabetical order:\n')
    pprint(relation_distribution)

    chart = plt.pie(
        sorted_relation_distribution.values(),
        labels=sorted_relation_distribution.keys(),
        autopct='%1.1f%%',
        radius=4,
        startangle=0
    )


def analyze_data_set(df):
    return analyze_relation_distribution(df)
