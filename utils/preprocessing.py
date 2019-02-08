import json
import numpy as np
from scipy import stats


def add_sentence_length(df):
    df['sentence.length'] = df['sentence'].apply(len)
    return df


def remove_outlier_sentence_lengths(df):
    total_count = len(df)
    cleaned = df[np.abs(stats.zscore(df['sentence.length'])) < 10]
    cleaned_count = len(cleaned)
    print('Removed {} outliers\n'.format(total_count - cleaned_count))
    return cleaned

# TODO: draw boxplot

# TODO: find out which are the outlier sentences
