import numpy as np
from concentration_inequalities import find_tightest_lambda
from svm_model import SvmRanking
from ranking_model import LogisticRanking, DecisionTreeRanking

def lambda_wrapper(lmbd):
    def predict():
        a1 = model.predict_probabilities()[:, 1].reshape(-1, 1) * 2 - 1 - lmbd
        a2 = model.predict_probabilities()[:, 1].reshape(-1, 1) * 2 - 1 + lmbd
        pred = np.hstack([a1, a2])
        # print(f'a1={a1.shape}')
        # print(f'a2={a2.shape}')
        # print(f'pred={pred.shape}')
        return pred

    return predict


def rank_loss(label, prediction_set):
    r_loss = ((((label > 0) * (np.max(prediction_set, axis=1) < 0))).sum() + (
                ((label < 0) * (np.min(prediction_set, axis=1) > 0))).sum()) / len(label)
    # print(f'r_loss={r_loss}')
    return r_loss


def evaluator(lmbd):
    # print('in evaluator')
    # print(f'data_valid.shape={data_valid.shape}')
    predict = lambda_wrapper(lmbd)
    prediction_set = predict()
    return rank_loss(model.y_val, prediction_set)

if __name__ == '__main__':
    model = LogisticRanking()
    # model = SvmRanking()
    # model = DecisionTreeRanking()
    model.train_model()
    min_lambda = find_tightest_lambda(26048, 0.1, 0.05, evaluator)
    print('#############################################################################')
    print(f'min_lambda={min_lambda}')
    print('#############################################################################')

