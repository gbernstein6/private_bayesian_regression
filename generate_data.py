import numpy as np
import scipy

from NIG import NIG_rvs_single_variance
from NIW import NIW_rvs
from util import symmetrize


def setup_data(data_dim, N):

    data_prior_params, model_prior_params = setup_prior_params(data_dim)
    true_params = generate_true_params(data_prior_params, model_prior_params)

    X, y, sensitivity_x, sensitivity_y = generate_data(true_params, N, data_dim)

    return data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, true_params


def setup_prior_params(data_dim):

    # NIW = [mu_0, lambda_0, psi_0, nu_0]
    data_prior_params = [np.array([0] * data_dim)[:, None],
                         1,
                         np.diag([1] * data_dim),
                         50
                        ]

    # NIG = [mu_0, lambda_0, a_0, b_0]
    a_0, b_0 = 20, .5
    c_0 = 1
    lambda_0 = b_0/(a_0 - 1) / c_0
    model_prior_params = [np.array([0] * (data_dim + 1))[:, None],
                          np.diag([lambda_0] * (data_dim + 1)),
                          a_0,
                          b_0
                          ]

    return data_prior_params, model_prior_params


def generate_true_params(data_prior_params, model_prior_params):

    theta, sigma_squared = NIG_rvs_single_variance(*model_prior_params)

    mu_x, Tau = NIW_rvs(*data_prior_params)

    if isinstance(mu_x, float):
        mu_x = np.array([mu_x])[:, None]

    true_params = {'theta': theta,
                   'sigma_squared': sigma_squared,
                   'mu_x': mu_x,
                   'Tau': Tau}

    return true_params


def generate_data(true_params, N, data_dim):

    X = scipy.stats.multivariate_normal.rvs(true_params['mu_x'].flatten(), true_params['Tau'], size=N)

    if N == 1:
        X = np.array([X])

    if data_dim == 1:
        X = X[:, None]

    # append constant bias term
    X = np.hstack((X, np.ones((N, 1))))

    y = scipy.stats.norm.rvs(X.dot(true_params['theta']), np.sqrt(true_params['sigma_squared']))

    # NOTE: these sensitivity calculations are not private
    sensitivity_x = X.max(axis=0) - X.min(axis=0)
    sensitivity_y = y.max() - y.min()

    return X, y, sensitivity_x, sensitivity_y


def privatize_suff_stats(S, sensitivity_x, sensitivity_y, epsilon):

    data_dim = S['XX'].shape[0]

    XX_comps = data_dim * (data_dim + 1) / 2  # upper triangular, not counting last column which is X
    X_comps = data_dim  # last column
    Xy_comps = data_dim
    yy_comps = 1
    sensitivity = XX_comps * sum(sensitivity_x[:-1]) ** 2 \
                  + X_comps * sum(sensitivity_x[:-1]) \
                  + Xy_comps * sum(sensitivity_x[:-1]) * sensitivity_y \
                  + yy_comps * sensitivity_y ** 2

    Z = {key: np.random.laplace(loc=val, scale=sensitivity / epsilon) for key, val in S.items()}

    # symmetrize Z_XX since we only want to add noise to upper triangle
    Z['XX'] = symmetrize(Z['XX'])

    Z['X'] = Z['XX'][:, 0][:, None]

    return Z, sensitivity