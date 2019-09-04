import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np


# non-centered hierarchical models
# https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
def mcmc(model_prior_params, data_prior_params, N, epsilon, Z, sensitivity, num_samples):

    data_dim = Z['XX'].shape[0] - 1

    if data_dim > 1:
        raise ValueError(f'MCMC only works for data dim 1! ({data_dim})')

    Z = Z.copy()
    Z['X'] = Z['XX'][:, -1][:, None]

    import pymc3 as pm
    from pymc3.distributions.continuous import InverseGamma
    from pymc3.distributions.continuous import Normal
    from pymc3.distributions.multivariate import MvNormal
    from pymc3.distributions.continuous import Laplace
    from pymc3 import Deterministic
    import theano.tensor as T

    num_tune_samples = 500
    max_treedepth = 12
    target_accept = .95

    with pm.Model():
        # data prior
        tau_squared = InverseGamma('ts', alpha=data_prior_params[2], beta=data_prior_params[3][0, 0])

        mu_x_offset = Normal('mu_x_offset', mu=0, sd=1)
        mu_x = Deterministic('mu', data_prior_params[0][0, 0] + mu_x_offset * pm.math.sqrt(tau_squared / data_prior_params[1][0, 0]))

        x_offset = Normal('x_offset', mu=0, sd=1, shape=N)
        x_temp = Deterministic('X', mu_x + x_offset * pm.math.sqrt(tau_squared))

        ones = T.shape_padright(pm.math.ones_like(x_temp))
        x = pm.math.concatenate((T.shape_padright(x_temp), ones), axis=1)

        # regression model
        sigma_squared = InverseGamma('ss', alpha=model_prior_params[2], beta=model_prior_params[3])

        L = pm.math.sqrt(sigma_squared) * np.linalg.cholesky(np.linalg.inv(model_prior_params[1]))
        theta_offset = MvNormal('theta_offset',
                                mu=[0] * (data_dim + 1),
                                cov=np.diag([1] * (data_dim + 1)),
                                shape=data_dim + 1)
        thetas = Deterministic('t', model_prior_params[0].flatten() + pm.math.dot(L, theta_offset))

        # response data
        y_offset = Normal('y_offset', mu=0, sd=1, shape=N)
        y = Deterministic('y', pm.math.flatten(pm.math.dot(thetas, x.T)) + y_offset * pm.math.sqrt(sigma_squared))

        # noisy sufficient statistics
        noise_scale = sensitivity / epsilon
        Laplace('z-X', mu=pm.math.sum(x),               b=noise_scale, observed=Z['X'])
        Laplace('z-XX', mu=pm.math.sum(pm.math.sqr(x)), b=noise_scale, observed=Z['XX'].flatten())
        Laplace('z-Xy', mu=pm.math.sum(x.T * y),        b=noise_scale, observed=Z['Xy'])
        Laplace('z-yy', mu=pm.math.sum(pm.math.sqr(y)), b=noise_scale, observed=Z['yy'])

        trace = pm.sampling.sample(draws=num_samples, tune=num_tune_samples,
                                   nuts_kwargs={'max_treedepth': max_treedepth,
                                                'target_accept': target_accept,
                                                }
                                   )

    theta = trace.get_values('t')
    sigma_squared = trace.get_values('ss')

    return theta.squeeze(), sigma_squared
