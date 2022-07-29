import numpy as np
import os, shutil
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm
from time import time
from svm_model import SvmRanking
from concentration_inequalities import find_tightest_lambda
from ranking_model import LogisticRanking, DecisionTreeRanking

RCPS_params = namedtuple('RCPS_parameters', ['delta', 'alpha'])

def overwrite_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)  # Removes all the subdirectories!
        os.makedirs(path)

def run_experiment(model, params):
    model.train_model()
    min_lambda = find_tightest_lambda(len(model.X_val), params[data_type].delta, params[data_type].alpha,
                                      model.evaluator_val)
    risk, frac_abstained, interval_size_capped = model.evaluator_test(min_lambda)
    return risk, frac_abstained, interval_size_capped[:, np.newaxis], 2 * min_lambda

def plot_results(params, risk_list, interval_size_list, interval_size_capped_list, frac_abstained_list, result_dir):
    fig, ax = plt.subplots(2, 1)
    bins = np.linspace(0.0, params[data_type].alpha, 20).tolist() + [params[data_type].alpha + 0.01]
    counts, _ = np.histogram(risk_list, bins=bins)
    counts_weighter = counts.sum()
    ax[0].hist(bins[:-1], bins=bins, weights=counts / counts_weighter)
    ax[0].set_title(
        f'Risk at $ \\delta={params[data_type].delta}, \\alpha={params[data_type].alpha} $ Error={counts[-1] / counts_weighter}')
    ax[0].axvline(params[data_type].alpha)
    ax[0].grid()
    ax[1].hist(frac_abstained_list)
    ax[1].set_title('Fraction of samples abstained')
    ax[1].grid()
    plt.tight_layout()
    plt.savefig(result_dir + '/risk_abstained.png')
    # plt.show()

    fig, ax = plt.subplots(2, 1)
    ax[0].hist(interval_size_list)
    ax[0].set_title('Interval size')
    ax[0].grid()
    ax[1].hist(interval_size_capped_list.ravel())
    ax[1].set_title('Capped interval sizes')
    ax[1].grid()
    plt.tight_layout()
    plt.savefig(result_dir + '/interval_sizes.png')

if __name__ == '__main__':
    params = {'university_rankings': RCPS_params(0.05, 0.01),
              'yearmsd': RCPS_params(0.05, 0.1),
              'moviedata': RCPS_params(0.05, 0.1)}
    for data_type in ['moviedata', 'yearmsd', 'university_rankings']:
        risk_list = []
        frac_abstained_list = []
        interval_size_list = []
        interval_size_capped_list = []
        model = LogisticRanking(data_type=data_type)
        result_dir = f'./{data_type}_results'
        overwrite_dir(result_dir)
        for i in tqdm(range(50)):
            risk, frac_abstained, interval_size_capped, interval_size = run_experiment(model, params)
            risk_list.append(risk)
            frac_abstained_list.append(frac_abstained)
            interval_size_capped_list.append(interval_size_capped)
            interval_size_list.append(interval_size)

        interval_size_capped_list = np.concatenate(interval_size_capped_list)
        plot_results(params, risk_list, interval_size_list, interval_size_capped_list, frac_abstained_list, result_dir)
