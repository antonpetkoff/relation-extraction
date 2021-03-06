import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from functools import partial

from base_model import BaseModel


class LogRegWordEmbeddingsAll(BaseModel):
    def __init__(self):
        self.params = {
            'description': 'Logistic Regression with Word Embeddings (Baseline)',
            'logreg_clf_params': {
                'multi_class': 'multinomial',
                'penalty': 'l2',
                # 'class_weight': 'balanced',
                'solver': 'lbfgs',
                # 'max_iter': 3000,
                'n_jobs': 6, # get_n_jobs(),
            },
            'word_embeddings_path': '../data/raw/word_vec.json'
        }

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))
        self.word_embeddings = LogRegWordEmbeddingsAll.load_word_embeddings(
            self.params['word_embeddings_path']
        )
        self.word_embeddings_dim = len(self.word_embeddings['the'])
        self.average = partial(
            LogRegWordEmbeddingsAll.average_embeddings,
            self.word_embeddings,
            self.word_embeddings_dim
        )
        self.label_encoder = LabelEncoder()


    # train_x is a dataframe with columns head.word, tail.word, sentence
    def fit(self, train_x, train_y):
        self.train_features = self.transform(train_x)
        self.train_labels = self.transform_labels(train_y)

        self.model = LogisticRegression(**self.params['logreg_clf_params'])

        print('Fitting model...')
        self.model.fit(self.train_features, self.train_labels)


    def predict(self, test_x):
        features = self.transform(test_x)

        predictions = self.model.predict(features)

        return self.label_encoder.inverse_transform(predictions)


    def predict_proba(self, test_x):
        features = self.transform(test_x)

        return self.model.predict_proba(features)


    # transforms the input train_x or test_x examples into features for the model
    # df is a dataframe with columns head.word, tail.word, sentence
    def transform(self, df):
        print('Tokenizing head.words, tail.words and sentences...')
        df_tokenized = df.applymap(self.preprocess)
        print(df_tokenized.head())

        print('Averaging word embeddings...')
        df_vectors = df_tokenized.applymap(self.average)
        print(df_vectors.head())

        vectors = np.hstack([
            np.hstack(df_vectors['head.word'].values).reshape(-1, self.word_embeddings_dim),   # flatten
            np.hstack(df_vectors['tail.word'].values).reshape(-1, self.word_embeddings_dim),   # flatten
            np.hstack(df_vectors['sentence'].values).reshape(-1, self.word_embeddings_dim)     # flatten
        ])
        print('Shape of transformed input: {}'.format(vectors.shape))

        return vectors


    def transform_labels(self, df):
        print('Fitting label encoder...')
        self.label_encoder.fit(df)
        print(self.label_encoder.classes_)

        print('Transforming labels...')
        labels = self.label_encoder.transform(df)
        print('Shape of transformed labels: {}'.format(labels.shape))

        return labels


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
