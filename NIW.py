import numpy as np
import scipy
import scipy.stats

from util import isPD, nearestPD


def NIW_rvs(mu, lamb, psi, nu, size=1):

    cov = scipy.stats.invwishart.rvs(nu, psi, size=size)
    if isinstance(cov, float):
        cov = np.array([[cov]])

    if size == 1:

        cov_inv_lamb = cov / lamb
        if not isPD(cov_inv_lamb):
            cov_inv_lamb = nearestPD(cov_inv_lamb)

        mean = np.random.multivariate_normal(mu.flatten(), cov_inv_lamb, check_valid='warn')[:, None]

    else:
        if len(mu) == 1:
            mean = [np.array([scipy.stats.multivariate_normal.rvs(mu.flatten(), cov[i] / lamb)])[:, None] for i in range(size)]
        else:
            mean = [scipy.stats.multivariate_normal.rvs(mu.flatten(), cov[i, :, :] / lamb)[:, None] for i in range(size)]

    return mean, cov


def NIW_conjugate_update(S, data_prior_params, N, size=1):
    # prior_params[0] = mu_0
    # prior_params[1] = lambda_0
    # prior_params[2] = psi_0
    # prior_params[3] = nu_0

    d = len(S['X']) - 1
    xx_bar = S['XX'][:d, :d] / N
    x_bar = S['X'][:d] / N
    s_bar = N * (xx_bar - x_bar.T.dot(x_bar))

    mu0, lambda0, psi0, nu0 = data_prior_params

    mu = (lambda0 * mu0 + N * x_bar) / (lambda0 + N)
    lam = lambda0 + N
    nu = nu0 + N
    psi = psi0 + s_bar + lambda0 * N / (lambda0 + N) * (x_bar - mu0).dot((x_bar - mu0).T)

    for i in range(psi.shape[0]):
        psi[i, i] = max(psi[i, i], .1)

    mu_x, Tau = NIW_rvs(mu, lam, psi, nu, size=size)

    if isinstance(mu_x, float):
        mu_x = np.array([[mu_x]])
        Tau = np.array([[Tau]])

    return mu_x, Tau


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    data_prior_params = [np.array([[0]]), np.array([[1]]), 1, 50]
    N = 10000
    num_trials = 100

    Us = np.empty((num_trials, 2))
    for t in range(num_trials):
        mu_x, tau_squared = NIW_rvs(*data_prior_params)
        X = np.random.multivariate_normal(mu_x.flatten(), tau_squared, size=N)
        X = np.hstack((X, np.ones(X.shape)))
        S = {'X': sum(X)[:, None],
             'XX': X.T.dot(X)}

        posterior = NIW_conjugate_update(S, data_prior_params, N, size=2000)

        Us[t, 0] = np.sum(np.array(posterior[0]).squeeze() < mu_x.flatten()) / 2000
        Us[t, 1] = np.sum(np.array(posterior[1]).squeeze() < tau_squared) / 2000

    fig, axes = plt.subplots(ncols=2)
    for p in range(2):

        ax = axes[p]

        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.plot(np.array(range(num_trials)) / float(num_trials), sorted(Us[:, p]))

        ax.set_xlabel('rank sort position')
        if p == 0:
            ax.set_ylabel('cdf of true parameter')
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        ax.set(aspect='equal')  # , adjustable='box-forced'

    plt.savefig('/Users/gbernstein/Desktop/QQ.png', bbox_inches='tight')
    plt.close(fig)

