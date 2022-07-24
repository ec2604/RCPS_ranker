import numpy as np
from sklearn import svm
from concentration_inequalities import find_tighest_lambda

data_a = np.random.normal(0, 2, (100, 50))
data_b = np.random.normal(-3, 2, (100, 50))
pos_data = data_a[:50] - data_b[:50]
neg_data = data_b[50:] - data_a[50:]
data = np.vstack([pos_data, neg_data])
label = np.vstack([np.ones((50, 1)), -1 * np.ones((50, 1))])
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(data, label.ravel())
# print(clf.coef_ @ (np.random.normal(0,2, (50,1)) - np.random.normal(-3,2,(50,1))))
# for i in range(50):
#     print(clf.predict_proba(np.random.normal(-3,2,(1,50)) - np.random.normal(0,2,(1,50)))[0][1]*2-1)
data_a_valid = np.random.normal(0, 2, (100, 50))
data_b_valid = np.random.normal(-3, 2, (100, 50))
pos_data_valid = data_a_valid[:50] - data_b_valid[:50]
neg_data = data_b_valid[50:] - data_a_valid[50:]
data_valid = np.vstack([pos_data_valid, neg_data])
label_valid = np.vstack([np.ones((50, 1)), -1 * np.ones((50, 1))])

def lambda_wrapper(lmbd):
    def predict(x):
        return np.hstack([clf.predict_proba(x)[:,1].reshape(-1,1)*2-1 - lmbd, clf.predict_proba(x)[:,1].reshape(-1,1)*2-1 + lmbd])

    return predict


def rank_loss(label, prediction_set):
    return (((label[:,0] > 0) * (np.min(prediction_set, axis=1) < 0))).sum() + (((label[:,0] < 0) * (np.max(prediction_set,axis=1) > 0))).sum()


def evaluator(lmbd):
    predict = lambda_wrapper(lmbd)
    prediction_set = predict(data_valid)
    return rank_loss(label_valid, prediction_set)

print(find_tighest_lambda(100, 0.01, 5, evaluator))