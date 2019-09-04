from generate_data import setup_data
from run_methods import run_methods
from evaluation import plot_posteriors


def main():

    # number of individuals
    N = int(1e3)

    # privacy setting
    epsilon = .01

    # dimension of covariate data
    data_dim = 1

    methods = [
               'non-private',
               'naive',
               'mcmc',
               'gibbs-noisy',
               'gibbs-update'
              ]

    data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, true_params = setup_data(data_dim, N)

    posteriors = run_methods(data_prior_params, model_prior_params, X, y, sensitivity_x, sensitivity_y, epsilon, N, methods)

    plot_posteriors(posteriors, true_params, N, epsilon)


if __name__ == '__main__':
    main()