import numpy as np
from numpy.linalg import inv
import autograd.numpy as npa
from autograd import jacobian


# https://mathematica.stackexchange.com/questions/34779/how-to-efficiently-find-moments-of-a-multinormal-distribution
# https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A8513D9EAC25318B5E83B695271527BE/S000497270000976Xa.pdf/moments_of_the_multivariate_normal.pdf
def calc_hier_norm_X_fourth_moment(mu_x, Tau):

    M = lambda t: npa.exp(npa.dot(t.T, mu_x) + .5 * npa.dot(npa.dot(t.T, Tau), t))

    M_1 = jacobian(M)
    M_2 = jacobian(M_1)
    M_3 = jacobian(M_2)
    M_4 = jacobian(M_3)

    t = npa.array([0])

    d = len(mu_x) + 1
    if d > 2:
        raise ValueError('Extend this to work for multivariate covariate data') # TODO
    E_vals = [M(t), M_1(t).squeeze(), M_2(t).squeeze(), M_3(t).squeeze(), M_4(t).squeeze()]
    Ex4 = np.empty((d, d, d, d))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                        Ex4[i, j, k, l] = E_vals[[i, j, k, l].count(0)]

    return Ex4


def calc_posterior_params(S, N, model_prior_params):

    mu_0, lambda_0, a_0, b_0 = model_prior_params

    S = project_suff_stats(S)

    lambda_n = symmetrize(S['XX'] + lambda_0)
    inv_lambda_n = symmetrize(np.linalg.inv(lambda_n))

    # if not isPD(lambda_n):
    #     lambda_n = nearestPD(lambda_n)
    #
    # if not isPD(inv_lambda_n):
    #     inv_lambda_n = nearestPD(inv_lambda_n)

    mu_n = inv_lambda_n.dot(S['Xy'] + lambda_0.dot(mu_0))
    a_n = a_0 + .5 * N
    b_n = b_0 + .5 * (S['yy'] + mu_0.T.dot(lambda_0).dot(mu_0) - mu_n.T.dot(lambda_n).dot(mu_n))[0, 0]

    # b_n = max(b_n, .1) # TODO put back in?

    return mu_n, lambda_n, a_n, b_n


def product_of_two_multivariate_normals(mean1, cov1, mean2, cov2):

    # https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
    # cholesky
    try:
        L_inv = inv(np.linalg.cholesky(cov1 + cov2))
        cov1_tilde = L_inv.dot(cov1)
        cov2_tilde = L_inv.dot(cov2)
        combined_covariance = cov1_tilde.T.dot(cov2_tilde)
        combined_mean = cov2_tilde.T.dot(L_inv.dot(mean1)) + cov1_tilde.T.dot(L_inv.dot(mean2))

    # cov1+cov2 isn't PSD
    except np.linalg.LinAlgError:
        temp = inv(cov1 + cov2)
        combined_covariance = cov1.dot(temp).dot(cov2)
        combined_mean = cov2.dot(temp).dot(mean1) \
                        + cov1.dot(temp).dot(mean2)
    #
    # # alternative method with direct inverses of individual cov matrices, not as accurate
    # if False:
    #     combined_covariance = inv(inv(cov1) + inv(cov2))
    #     combined_mean = combined_covariance.dot(inv(cov1).dot(mean1)
    #                                                   + inv(cov2).dot(mean2))

    return combined_mean, combined_covariance


def project_suff_stats(S):

    AA = S['XX'].copy()
    AA = np.hstack((AA, S['Xy']))
    AA = np.vstack((AA, np.hstack((S['Xy'].T, [[S['yy']]]))))

    if not isPD(AA):
        AA = nearestPD(AA)

    d = AA.shape[0]
    S['XX'] = AA[:d-1, :d-1]
    S['X'] = S['XX'][:, -1][:, None]
    S['Xy'] = AA[-1, :-1][:, None]
    S['yy'] = AA[-1, -1]

    return S


def nearestPD_iterative(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


# https://math.stackexchange.com/questions/2776803/matrix-projection-onto-positive-semi-definite-cone-with-respect-to-the-spectral
def nearestPD(B):

    B = B.copy()

    flip = False
    if len(B.shape) == 4:
        d = B.shape[0]
        B = B.reshape((d ** 2, d ** 2))
        flip = True

    if True:
        w, v = np.linalg.eig(B)
        w = np.real(w)
        v = np.real(v)
        # w[w < 1e-3] = 1e-3 # np.finfo(float).eps
        w[w < 0] = 0  # np.finfo(float).eps
        B = v.dot(np.diag(w)).dot(v.T)

        B = symmetrize(B)

        min_eig = np.min(np.real(np.linalg.eigvals(B)))
        if min_eig < 0:
            B -= 10 * min_eig * np.eye(*B.shape)

    else:
        u, s, vh = np.linalg.svd(B)

        # u_new = (u + vh) / 2
        u_new = u
        B = u_new.dot(np.diag(s)).dot(u_new.T)

    B = symmetrize(B)

    if flip:
        B = B.reshape((d, d, d, d))

    return B


def symmetrize(B):

    B = B.copy()

    flip = False
    if len(B.shape) == 4:
        d = B.shape[0]
        B = B.reshape((d ** 2, d ** 2))
        flip = True

    B = np.triu(B) + np.triu(B, k=1).T

    if flip:
        B = B.reshape((d, d, d, d))

    return B


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""

    B = B.copy()

    if len(B.shape) == 4:
        d = B.shape[0]
        B = B.reshape((d ** 2, d ** 2))

    try:
        np.linalg.cholesky(B)
        np.linalg.svd(B)
    except np.linalg.LinAlgError:
        return False

    try:
        np.random.multivariate_normal(mean=np.zeros(B.shape[0]), cov=B, check_valid='raise')
    except ValueError:
        return False

    return True
