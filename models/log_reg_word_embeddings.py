import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from functools import partial

from base_model import BaseModel


class LogRegWordEmbeddings(BaseModel):
    def __init__(self):
        self.params = {
            'description': 'Logistic Regression with Word Embeddings (Baseline)',
            'logreg_clf_params': {
                'penalty': 'l2',
                'class_weight': 'balanced',
                'solver': 'lbfgs',
                'max_iter': 3000,
                'n_jobs': 1, # get_n_jobs(),
            },
            'word_embeddings_path': '../data/raw/word_vec.json'
        }

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
        self.word_embeddings = LogRegWordEmbeddings.load_word_embeddings(
            self.params['word_embeddings_path']
        )
        self.average = partial(
            LogRegWordEmbeddings.average_embeddings,
            self.word_embeddings,
            len(self.word_embeddings['the'])
        )


    # train_x is a dataframe with columns head.word, tail.word, sentence
    def fit(self, train_x, train_y):
        features = self.transform(train_x)

        # self.model = LogisticRegression(**self.params['logreg_clf_params'])
        # self.model.fit(features, train_y)


    def predict(self, test_x):
        features = self.transform(test_x)

        return self.model.predict(features)


    # transforms the input train_x or test_x examples into features for the model
    # df is a dataframe with columns head.word, tail.word, sentence
    def transform(self, df):
        print('Tokenizing sentences...')
        df_tokenized = df['sentence'].apply(self.preprocess)
        print(df_tokenized.head())

        print('Averaging word embeddings...')
        df_vectors = df_tokenized.apply(self.average)
        print(df_vectors.head())

        return df_vectors


    def preprocess(self, text):
        # tokenize
        word_tokens = self.tokenizer.tokenize(text)

        # clean stop words and lower cases
        return [
            word.lower()
            for word in word_tokens
            if not word in self.stop_words
        ]


    @staticmethod
    def load_word_embeddings(path):
        with open(path) as f:
            word_vec = json.load(f)

        word_embeddings = {
            obj['word']: np.asarray(obj['vec'])
            for obj in word_vec
        }
        return word_embeddings


    @staticmethod
    def average_embeddings(word_embeddings, word_embeddings_dim, words):
        embeddings = [
            word_embeddings[word]
            for word in words
            if word in word_embeddings
        ]

        if len(embeddings) > 0:
            return np.average(embeddings, axis=0)
        else:
            return np.zeros(word_embeddings_dim)

    def get_grid_params(self):
        return {
            'max_iter': (5, 100),
            'solver': ('liblinear', 'sag'),
        }
