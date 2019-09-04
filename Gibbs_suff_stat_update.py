import numpy as np

from util import isPD, nearestPD, product_of_two_multivariate_normals, project_suff_stats, calc_posterior_params


# https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
# https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
def update_sufficient_statistics(Ex2, Cov_xx_xx, Z, model_prior_params, model_params, noise_covariance, N):

    # XX, Xy, yy
    mu_S = calc_mu_S(model_params, Ex2, N)
    Sigma_S = calc_Sigma_S(model_params, Ex2, Cov_xx_xx, N)

    Z_vec = np.hstack((Z['XX'].flatten(), Z['Xy'].flatten(), Z['yy']))[:, None]

    # mu_1 = Z, var_1 = noise_covariance
    # mu_2 = mu_S, var_2 = Sigma_S
    mu_combined, Sigma_combined = product_of_two_multivariate_normals(Z_vec, noise_covariance,
                                                                      mu_S, Sigma_S)

    tries = 0
    while not isPD(Sigma_combined) and tries < 3:
        Sigma_combined = nearestPD(Sigma_combined)
        tries += 1

    if False and tries > 1: print(tries)

    tries = 0
    while True and tries < 10:
        # S_vec = draw_S_vec(mu_combined, Sigma_combined)
        S_vec = np.random.multivariate_normal(mean=mu_combined.flatten(), cov=Sigma_combined, check_valid='warn')

        d = Z['X'].shape[0]
        S = {'XX': S_vec[:d ** 2].reshape((d, d)),
             'Xy': S_vec[d ** 2:-1][:, None],
             'yy': S_vec[-1]}

        # S = project_suff_stats(S)

        tries += 1

        if S['XX'][0, 0] > 0 and S['yy'] > 0 and calc_posterior_params(S, N, model_prior_params)[-1] > .1:
            break

    if False and tries > 1: print(tries)

    return S


def draw_S_vec(mu_combined, Sigma_combined):

    # https://stats.stackexchange.com/questions/159313/generating-samples-from-singular-gaussian-distribution
    # https://stats.stackexchange.com/questions/61719/cholesky-versus-eigendecomposition-for-drawing-samples-from-a-multivariate-norma
    # d = Sigma_combined.shape[0]
    # standard_draw = np.random.multivariate_normal(mean=np.zeros(d), cov=np.diag(np.ones(d)))[:, None]
    #
    # try:
    #     L = np.linalg.cholesky(Sigma_combined)
    #     S_vec = (mu_combined + L.dot(standard_draw)).flatten()
    # except:
    #     u, s, vh = np.linalg.svd(Sigma_combined)
    #     S_vec = (mu_combined + u.dot(np.diag(np.sqrt(s))).dot(standard_draw)).flatten()

    S_vec = np.random.multivariate_normal(mean=mu_combined.flatten(), cov=Sigma_combined, check_valid='warn')

    return S_vec


def calc_Cov_xx_xx(Ex2, Ex4):

    d = Ex2.shape[0]

    Cov_xx_xx = np.empty((d, d, d, d))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    Cov_xx_xx[i, j, k, l] = Ex4[i, j, k, l] - Ex2[i, j] * Ex2[k, l]

    if not isPD(Cov_xx_xx):
        Cov_xx_xx = nearestPD(Cov_xx_xx)

    return Cov_xx_xx


def calc_mu_S(model_params, Ex2, N):

    Ey2 = calc_Ey2(Ex2, model_params)

    d = Ex2.shape[0]

    Exy = np.zeros(d)
    for i in range(d):
        for j in range(d):
            Exy[i] += model_params[0][j] * Ex2[i, j]

    # XX, Xy, yy
    mu_S = N * np.hstack((Ex2.flatten(),
                          Exy.flatten(),
                          Ey2)
                         )[:, None]

    return mu_S


def calc_Ey2(Ex2, model_params):

    theta, sigma_squared = model_params
    d = len(theta)

    Ey2 = sigma_squared
    for i in range(d):
        for j in range(d):
            Ey2 += theta[i] * theta[j] * Ex2[i, j]

    return Ey2


def calc_Sigma_S(model_params, Ex2, Cov_xx_xx, N):

    theta, sigma_squared = model_params
    d = len(theta)

    Cov_xx_xy = np.zeros((d, d, d))
    Cov_xx_yy = np.zeros((d, d))
    Cov_xy_xy = np.zeros((d, d))
    Cov_xy_yy = np.zeros(d)
    Cov_yy_yy = 2 * sigma_squared ** 2
    for i in range(d):
        for j in range(d):
            Cov_xy_xy[i, j] += sigma_squared * Ex2[i, j]
            Cov_xy_yy[i] += 2 * sigma_squared * theta[j] * Ex2[i, j]
            Cov_yy_yy += 4 * sigma_squared * theta[i] * theta[j] * Ex2[i, j]
            for k in range(d):
                for l in range(d):
                    Cov_xx_xy[i, j, k]   += theta[l] * Cov_xx_xx[i, j, k, l]
                    Cov_xx_yy[i, j]      += theta[k] * theta[l] * Cov_xx_xx[i, j, k, l]
                    Cov_xy_xy[i, j]      += theta[k] * theta[l] * Cov_xx_xx[i, k, j, l]
                    Cov_xy_yy[i]         += theta[j] * theta[k] * theta[l] * Cov_xx_xx[i, j, k, l]
                    Cov_yy_yy            += theta[i] * theta[j] * theta[k] * theta[l] * Cov_xx_xx[i, j, k, l]

    if not isPD(Cov_xy_xy):
        Cov_xy_xy = nearestPD(Cov_xy_xy)

    Sigma_S = np.hstack((Cov_xx_xx.reshape(d ** 2, d ** 2),
                         Cov_xx_xy.reshape(d ** 2, d),
                         Cov_xx_yy.reshape(d ** 2, 1)
                         ))
    Sigma_S = np.vstack((Sigma_S,
                         np.hstack((Cov_xx_xy.reshape(d ** 2, d).T,
                                    Cov_xy_xy.reshape(d, d),
                                    Cov_xy_yy.reshape(d, 1)
                                    ))))
    Sigma_S = np.vstack((Sigma_S,
                         np.hstack((Cov_xx_yy.reshape(d ** 2, 1).T,
                                    Cov_xy_yy.reshape(d, 1).T,
                                    Cov_yy_yy.reshape(1, 1))
                                   )))

    Sigma_S *= N

    return Sigma_S