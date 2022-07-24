from abc import ABC
import sys
sys.path.insert(0, '../lib')  # noqa
import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression

from prepare_movie_data import MovieData


class RankingModel(ABC):
    def __init__(self, pairwise=True):
        self.model = None

        movie_data = MovieData(pairwise=pairwise)
        self.X_train, self.X_val, self.y_train, self.y_val = movie_data.get_train_and_validation_data()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

        y_train_pred = np.argmax(self.model.predict_proba(self.X_train), axis=1).astype(np.float32)

        print('train precision: ' + str(precision_score(self.y_train, y_train_pred)))
        print('train recall: ' + str(recall_score(self.y_train, y_train_pred)))
        print('train accuracy: ' + str(accuracy_score(self.y_train, y_train_pred)))

    def predict_probabilities(self):
        return self.model.predict_proba(self.X_val)

    # def prediction_function(self):
    #     ...


class LogisticRanking(RankingModel):
    def __init__(self, pairwise=True):
        super().__init__(pairwise=pairwise)
        self.model = LogisticRegression()
        self.train_model()
    #
    # def prediction_function(self):
    #     return np.argmax(self.model.predict_proba(self.X_val), axis=1).astype(np.float32)
    #
    # def get_predicted_rank(self, data):
    #     return self.model.predict_proba(data)[:, 1]


