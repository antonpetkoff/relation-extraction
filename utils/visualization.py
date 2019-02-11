from pprint import pprint
from pandas.io.json import json_normalize
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
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


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    np.set_printoptions(precision=1)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = plt.text(j, i, format(cm[i, j], fmt),
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")
        if i == j:
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='yellow')])


    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    return plt


def plot_precision_recall_curve(precision, recall):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.show()
    return plt
