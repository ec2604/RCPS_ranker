from abc import ABC
import sys

sys.path.insert(0, '../lib')  # noqa
import numpy as np
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from prepare_movie_data import MovieData
from prepare_year_msd_data import YearMSDData
from prepare_university_ranking_data import UniversityRankingData


class RankingModel(ABC):
    def __init__(self, pairwise=True, data_type='yearmsd'):
        self.model = None
        self.data_type = data_type
        if data_type == 'yearmsd':
            self.data = YearMSDData()
        elif data_type == 'moviedata':
            self.data = MovieData(pairwise=pairwise)
        elif data_type == 'university_rankings':
            self.data = UniversityRankingData()
        else:
            pass

    def train_model(self):
        self.X_train, self.X_val, self.y_train, self.y_val, self.X_test, self.y_test = self.data.get_train_and_validation_data()
        self.model.fit(self.X_train, self.y_train)

        # y_train_pred = np.argmax(self.model.predict_proba(self.X_train), axis=1).astype(np.float32)
        #
        # print('train precision: ' + str(precision_score(self.y_train, y_train_pred)))
        # print('train recall: ' + str(recall_score(self.y_train, y_train_pred)))
        # print('train accuracy: ' + str(accuracy_score(self.y_train, y_train_pred)))

    def predict_probabilities(self, validation=True):
        if validation:
            return self.model.predict_proba(self.X_val)
        else:
            return self.model.predict_proba(self.X_test)

    def lambda_wrapper(self, lmbd, validation=True):
        def predict():
            a1 = self.predict_probabilities(validation=validation)[:, 1].reshape(-1, 1) * 2 - 1 - lmbd
            a2 = self.predict_probabilities(validation=validation)[:, 1].reshape(-1, 1) * 2 - 1 + lmbd
            pred = np.hstack([a1, a2])
            # print(f'a1={a1.shape}')
            # print(f'a2={a2.shape}')
            # print(f'pred={pred.shape}')
            return pred

        return predict

    # def prediction_function(self):
    #     return np.argmax(self.model.predict_proba(self.X_val), axis=1).astype(np.float32)
    #
    # def get_predicted_rank(self, data):
    #     return self.model.predict_proba(data)[:, 1]
    def evaluator_val(self, lmbd):
        # print('in evaluator')
        # print(f'data_valid.shape={data_valid.shape}')
        predict = self.lambda_wrapper(lmbd)
        prediction_set = predict()
        return self.rank_loss(self.y_val, prediction_set)

    def rank_loss(self, label, prediction_set):
        r_loss = ((((label > 0) * (np.max(prediction_set, axis=1) < 0))).sum() + (
            ((label < 0) * (np.min(prediction_set, axis=1) > 0))).sum()) / len(label)
        # print(f'r_loss={r_loss}')
        return r_loss

    def evaluator_test(self, lmbd):
        # print('in evaluator')
        # print(f'data_valid.shape={data_valid.shape}')
        predict = self.lambda_wrapper(lmbd, validation=False)
        prediction_set = predict()
        frac_abstained = (np.sign(prediction_set)[:, 0] != np.sign(prediction_set)[:, 1]).sum() / len(prediction_set)
        interval_sizes_clipped = np.clip(prediction_set, -1, 1)[:, 1] - np.clip(prediction_set, -1, 1)[:, 0]
        return self.rank_loss(self.y_test, prediction_set), frac_abstained, interval_sizes_clipped


class LogisticRanking(RankingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression()
        self.train_model()


class DecisionTreeRanking(RankingModel):
    def __init__(self, pairwise=True):
        super().__init__(pairwise=pairwise)
        self.model = tree.DecisionTreeClassifier()
        self.train_model()
