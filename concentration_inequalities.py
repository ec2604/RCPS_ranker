import numpy as np
from scipy.stats import binom
from scipy.optimize import fminbound, brentq


def h_1(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def naive_hoeffding(n, r_lambda, alpha):
    return -n * h_1(np.min([r_lambda.astype(np.float), alpha]), alpha)


def bentkus(n, alpha, r_lambda):
    rv = binom(n, alpha)
    return 1 + np.log(np.max([rv.cdf(np.ceil(n * r_lambda)),1e-10]))


def g(nu):
    return (np.exp(nu) - nu - 1) / nu


def maurer(nu, alpha, r_lambda, n):
    return (-n * nu / 2) * (alpha / (1 + 2 * g(nu)) - np.min([r_lambda, alpha]))


def hbm(n, r_lambda, alpha, delta):
    m = np.floor(n / 2)
    nh_res = naive_hoeffding(m, np.max([r_lambda,1e-10]), alpha)
    bentkus_res = bentkus(m, alpha, r_lambda)
    curr_maurer = lambda nu: maurer(nu, alpha, r_lambda, n)
    maurer_res = fminbound(curr_maurer, 0, 1, full_output=True)[1]
    return np.min([maurer_res, bentkus_res, nh_res]) - np.log(delta)


def rcps_lambda(n, alpha, lmbd, risk_evaluator, delta):
    r_lambda = risk_evaluator(lmbd)
    return hbm(n, r_lambda, alpha, delta)


def find_lambda_ucb(n, lmbd, risk_evaluator, delta):
    optimized_func = lambda alpha: rcps_lambda(n, alpha, lmbd, risk_evaluator, delta)
    res = brentq(optimized_func, 1e-10, 1-1e-12)
    return res


def find_tightest_lambda(n, delta, req_alpha, risk_evaluator):
    lmbd = brentq(lambda lmbd: find_lambda_ucb(n, lmbd, risk_evaluator, delta) - req_alpha, 1e-10, 1.2)
    return lmbd
