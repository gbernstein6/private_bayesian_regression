import numpy as np
import scipy
import scipy.stats
from numpy.linalg import inv

from util import isPD, nearestPD, symmetrize


def NIG_rvs(mu, lam, a, b, size=1):

    if size == 1:
        return NIG_rvs_single_variance(mu, lam, a, b)

    sigma_squared = scipy.stats.invgamma.rvs(a=a, scale=b, size=size)

    inv_lam = inv(lam)

    theta = np.array([scipy.stats.multivariate_normal.rvs(mu.flatten(), symmetrize(ss * inv_lam)) for ss in sigma_squared])

    return theta, sigma_squared


def NIG_rvs_single_variance(mu, lam, a, b, size=1):

    sigma_squared = scipy.stats.invgamma.rvs(a=a, scale=b)

    cov = symmetrize(sigma_squared * inv(lam))

    if not isPD(cov):
        cov = nearestPD(cov)

    theta = scipy.stats.multivariate_normal.rvs(mu.flatten(), cov, size=size)

    if isinstance(theta, float):
        theta = np.array([theta])

    if size == 1:
        theta = theta[:, None]

    return theta, sigma_squared


def NIG_conjugate_update(S, data_prior_params, N):
    # prior_params[0] = mu_0
    # prior_params[1] = sigma^2_0 = lambda_0
    # prior_params[2] = alpha_0
    # prior_params[3] = beta_0

    x_bar = S['X'] / N  # sample mean
    # s_bar = suff_stats[1] / N - x_bar ** 2  # sample variance
    s_bar = (S['XX'] - S['X'] ** 2 / N) / (N - 1)  # sample variance

    mu0, lambda0, alpha0, beta0 = data_prior_params

    mu = (lambda0 * mu0 + N * x_bar) / (lambda0 + N)
    lam = lambda0 + N
    alpha = alpha0 + .5 * N
    bet = beta0 + .5 * N * (s_bar + lambda0 * (x_bar - mu0) ** 2 / (lambda0 + N))

    updated_posterior_params = np.array((mu, lam, alpha, bet))

    # constrain scale params to positive values
    for i in range(1, 4):
        updated_posterior_params[i] = np.max((updated_posterior_params[i], .1))

    draw = NIG_rvs(*updated_posterior_params)

    return draw