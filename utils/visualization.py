from pprint import pprint
from pandas.io.json import json_normalize
import json
import numpy as np
import matplotlib.pyplot as plt
import preprocessing


def get_relation_distribution(df):
    unique_relation_labels = np.unique(df['relation'])
    relation_distribution = {
        relation: df[df.relation == relation]['relation'].count()
        for relation in unique_relation_labels
    }
    return dict(sorted(relation_distribution.items(), key=lambda kv: kv[1]))


def draw_relation_distribution(relation_distribution):
    plt.pie(
        relation_distribution.values(),
        labels=relation_distribution.keys(),
        autopct='%1.1f%%',
        # radius=3,
        startangle=0
    )
    plt.show()


def show_examples(df, relation_distribution):
    attributes = ['sentence', 'head.word', 'relation', 'tail.word']

    for relation in relation_distribution.keys():
        example = df[df.relation == relation].iloc[0]
        for attribute in attributes:
            print('{}: {}'.format(attribute, example[attribute]))
        print()


def draw_sentence_length_distribution(df):
    lengths = df['sentence'].apply(len)
    # max_len = lengths.max()
    return lengths.hist(bins=1200)


def analyze_data_set(df):
    plt.rcParams['figure.figsize'] = [16,9]

    relation_distribution = get_relation_distribution(df)

    print('Relations in alphabetical order:\n')
    pprint(relation_distribution)

    print('Relation distribution:\n')
    draw_relation_distribution(relation_distribution)

    print('Relation distribution without NA:\n')
    draw_relation_distribution({
        relation: count
        for relation, count in relation_distribution.items()
        if relation != 'NA'
    })

    print('Examples:\n')
    show_examples(df, relation_distribution)

    print('Sentence length distribution:\n')
    draw_sentence_length_distribution(df)

    print('Boxplot of sentence length:\n')
    df = preprocessing.add_sentence_length(df)
    df['sentence.length'].plot.box()
