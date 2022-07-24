import numpy as np
from sklearn import svm


def generate_svm_data():
    data_a = np.random.normal(0, 2, (100, 50))
    data_b = np.random.normal(-3, 2, (100, 50))
    pos_data = data_a[:50] - data_b[:50]
    neg_data = data_b[50:] - data_a[50:]
    data = np.vstack([pos_data, neg_data])
    label = np.vstack([np.ones((50, 1)), -1 * np.ones((50, 1))])
    return data, label


class SvmRanking:
    def __init__(self):
        self.model = svm.SVC(kernel='linear', probability=True)
        self.train_data, self.train_label = generate_svm_data()
        self.val_data, self.val_label = generate_svm_data()

    def get_val_data_and_labels(self):
        return self.val_data, self.val_label

    def fit_model_on_training(self):
        self.model.fit(self.train_data, self.train_label.ravel())

    def predict_probabilities(self, x):
        return self.model.predict_proba(x)
