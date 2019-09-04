import numpy as np
import scipy
import scipy.stats

from NIG import NIG_rvs, NIG_rvs_single_variance
from NIW import NIW_rvs, NIW_conjugate_update
from Gibbs_suff_stat_update import update_sufficient_statistics, calc_Cov_xx_xx
from util import isPD, nearestPD, project_suff_stats, symmetrize, calc_posterior_params, calc_hier_norm_X_fourth_moment


def Gibbs(data_prior_params,
          model_prior_params,
          N,
          epsilon_S,
          Z,
          sensitivity,
          num_burnin,
          num_iterations,
          gibbs_flavor,
          X_fourth_moment=None):

    model_params, data_params, noise_covariance = initialize_values(data_prior_params, model_prior_params, epsilon_S, sensitivity, Z, gibbs_flavor)

    if gibbs_flavor == 'gibbs-update':
        X_second_moment, X_fourth_moment, Cov_xx_xx = calc_hier_norm_moments(data_params)
    else:
        X_second_moment = X_fourth_moment[:, :, -1, -1]
        Cov_xx_xx = calc_Cov_xx_xx(X_second_moment, X_fourth_moment)
        if not isPD(X_second_moment):
            X_second_moment = nearestPD(X_second_moment)

    theta = np.zeros((num_iterations, Z['X'].shape[0]))
    sigma_squared = np.zeros(num_iterations)
    for iteration in range(num_iterations + num_burnin):

        S = update_sufficient_statistics(X_second_moment, Cov_xx_xx, Z, model_prior_params, model_params, noise_covariance, N)

        model_params = update_model_params(S, model_prior_params, N)

        X_second_moment, X_fourth_moment, Cov_xx_xx, data_params = update_moments(data_prior_params, X_second_moment, X_fourth_moment, Cov_xx_xx, S, N, gibbs_flavor)

        noise_covariance = update_noise_covariance(S, Z, epsilon_S, sensitivity)

        if iteration >= num_burnin:
            theta[iteration - num_burnin, :] = model_params[0].flatten()
            sigma_squared[iteration - num_burnin] = model_params[1]

    return theta, sigma_squared


def initialize_values(data_prior_params, model_prior_params, epsilon_S, sensitivity, Z, gibbs_flavor):

    model_params = NIG_rvs(*model_prior_params)

    if gibbs_flavor == 'gibbs-update':
        data_params = NIW_rvs(*data_prior_params)
    else:
        data_params = None

    dim = Z['X'].shape[0] ** 2 + Z['X'].shape[0] + 1  # = dim(XX^T) + dim(xy) + dim(y)
    exp_scale = 2 * (sensitivity / epsilon_S) ** 2
    noise_covariance = np.diag(np.random.exponential(scale=exp_scale, size=dim))

    return model_params, data_params, noise_covariance


def update_moments(data_prior_params, Ex2, Ex4, Cov_xx_xx, S, N, gibbs_flavor):

    if gibbs_flavor == 'gibbs-update':
        data_params = NIW_conjugate_update(project_suff_stats(S), data_prior_params, N)
        Ex2, Ex4, Cov_xx_xx = calc_hier_norm_moments(data_params)
    elif gibbs_flavor in ['gibbs-noisy', 'gibbs-exact', 'gibbs-prior']:
        data_params = None
    else:
        raise ValueError(f'Unrecognized moments source! ({gibbs_flavor})')

    return Ex2, Ex4, Cov_xx_xx, data_params


def calc_hier_norm_moments(data_params):

    mu_x, Tau = data_params

    if isinstance(mu_x, float):
        mu_x = np.array([[mu_x]])
        Tau = np.array([[Tau]])

    Ex4 = calc_hier_norm_X_fourth_moment(mu_x, Tau)
    Ex2 = Ex4[:, :, -1, -1]

    Cov_xx_xx = calc_Cov_xx_xx(Ex2, Ex4)

    if not isPD(Ex2):
        Ex2 = nearestPD(Ex2)

    return Ex2, Ex4, Cov_xx_xx


def update_noise_covariance(S, Z, epsilon_S, sensitivity):

    abs_noise = np.abs(np.hstack(([Z['XX'].flatten() - S['XX'].flatten(),
                                   Z['Xy'].flatten() - S['Xy'].flatten(),
                                   Z['yy'] - S['yy']])))

    laplace_lambda = sensitivity / epsilon_S

    inverse_variance = np.random.wald(1 / (laplace_lambda * abs_noise), 1 / laplace_lambda ** 2)

    variance = 1 / inverse_variance

    covariance = np.diag(np.array(variance).flatten())

    return covariance


def update_model_params(S, model_prior_params, N):

    mu_n, lambda_n, a_n, b_n = calc_posterior_params(S, N, model_prior_params)

    sigma_squared = scipy.stats.invgamma.rvs(a=a_n, scale=b_n)

    cov = symmetrize(sigma_squared * np.linalg.inv(lambda_n))

    if not isPD(cov):
        cov = nearestPD(cov)

    theta = scipy.stats.multivariate_normal.rvs(mu_n.flatten(), cov)

    if isinstance(theta, float):
        theta = np.array([theta])

    theta = theta[:, None]

    return theta, sigma_squared

