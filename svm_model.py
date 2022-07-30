import numpy as np
from sklearn import svm


class SvmRanking:
    def __init__(self):
        self.model = svm.SVC(kernel='linear', probability=True)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train.ravel())

    def get_val_data_and_labels(self):
        return self.X_val, self.y_val

    def predict_probabilities(self, validation=True):
        if validation:
            return self.model.predict_proba(self.X_val)
        else:
            return self.model.predict_proba(self.X_train)
