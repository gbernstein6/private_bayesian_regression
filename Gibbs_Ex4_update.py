import numpy as np
import scipy

from util.Methods.Gibbs_util import isPD, nearestPD, product_of_two_multivariate_normals


def update_Ex4(Ex2):

    d = Ex2.shape[0]
    Ex4 = np.empty((d, d, d, d))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    Ex4[i, j, k, l] = Ex2[i, j] * Ex2[k, l] \
                                      + Ex2[i, k] * Ex2[j, l] \
                                      + Ex2[i, l] * Ex2[j, k]

    if not isPD(Ex4):
        Ex4 = nearestPD(Ex4)

    return Ex4

# def update_Ex4(moments_of_x, Cov_xx_xx, Zx4, model_params, noise_covariance, N, settings):
#
#     # XX, Xy, yy
#     mu_S = calc_mu_S(model_params, moments_of_x, N)
#     Sigma_S = calc_Sigma_S(moments_of_x, Cov_xx_xx, model_params, N)
#
#     Z_vec = Zx4.flatten()[:, None]
#
#     # mu_1 = Z, var_1 = noise_covariance
#     # mu_2 = mu_S, var_2 = Sigma_S
#     mu_combined, Sigma_combined = product_of_two_multivariate_normals(Z_vec, noise_covariance,
#                                                                       mu_S, Sigma_S)
#
#     tries = 0
#     while not isPD(Sigma_combined) and tries < 3:
#         Sigma_combined = nearestPD(Sigma_combined)
#         tries += 1
#
#     # if not isPD(Sigma_combined):
#     #     Sigma_combined = nearestPD(Sigma_combined)
#
#     # S_vec = np.random.multivariate_normal(mean=mu_combined.flatten(), cov=Sigma_combined)#, check_valid='raise')
#     Ex4_vec = scipy.stats.multivariate_normal.rvs(mean=mu_combined.flatten(), cov=Sigma_combined)
#
#     d = settings['data_dim'] + 1
#     Ex4 = Ex4_vec.reshape((d, d, d, d))
#
#     return Ex4