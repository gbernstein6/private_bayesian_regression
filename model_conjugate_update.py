import numpy as np
import scipy

from util import isPD, nearestPD, symmetrize
from generate_data import privatize_suff_stats


def run_non_private(model_prior_params, S, N):

    theta, sigma_squared = model_conjugate_update(model_prior_params, S, N, project=False, size=2000)

    return theta, sigma_squared


def run_naive(model_prior_params, S, N, sensitivity_x, sensitivity_y, epsilon):

    Z, sensitivity = privatize_suff_stats(S, sensitivity_x, sensitivity_y, epsilon)

    theta, sigma_squared = model_conjugate_update(model_prior_params, Z, N, project=True, size=2000)

    return theta, sigma_squared


def model_conjugate_update(model_prior_params, S, N, project=False, size=1):

    mu_0, lambda_0, a_0, b_0 = model_prior_params

    lambda_n = S['XX'] + lambda_0
    inv_lambda_n = np.linalg.inv(lambda_n)

    if not isPD(lambda_n):
        lambda_n = nearestPD(lambda_n)

    if not isPD(inv_lambda_n):
        inv_lambda_n = nearestPD(inv_lambda_n)

    mu_n = inv_lambda_n.dot(S['Xy'] + lambda_0.dot(mu_0))
    a_n = a_0 + .5 * N
    b_n = b_0 + .5 * (S['yy'] + mu_0.T.dot(lambda_0).dot(mu_0) - mu_n.T.dot(lambda_n).dot(mu_n))[0, 0]

    if project:
        b_n = max(b_n, .1)

    sigma_squared = scipy.stats.invgamma.rvs(a=a_n, scale=b_n, size=size)

    theta = np.array([scipy.stats.multivariate_normal.rvs(mu_n.flatten(), symmetrize(ss * inv_lambda_n)) for ss in sigma_squared])

    return theta, sigma_squared
