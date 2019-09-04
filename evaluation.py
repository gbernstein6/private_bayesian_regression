import matplotlib.pyplot as plt
import numpy as np


def plot_posteriors(posteriors, true_params, N, epsilon):

    data_dim = true_params['mu_x'].shape[0]

    true_params = np.hstack((true_params['theta'].flatten(), true_params['sigma_squared']))

    param_labels = [r'$\theta_%d$' % i for i in range(data_dim)]
    param_labels.extend([r'$\theta_{bias}$', r'$\sigma^2$'])

    fig, axes = plt.subplots(ncols=data_dim + 2) # 2 = bias term + sigma_squared
    for p, param in enumerate(param_labels):

        ax = axes[p]

        alpha = 0.4

        for method in posteriors:

            to_plot = np.hstack((posteriors[method][0], posteriors[method][1][:, None]))

            np_hist, _, _ = ax.hist(to_plot[:, p], bins=50, alpha=alpha, label=method,
                                    linewidth=1.5,
                                    histtype='step', stacked=True, fill=False)

        ax.axvline(true_params[p], color='k', linestyle='--', label='true parameter')

        ax.set_xlabel(param)
        ax.set_yticks(())

        if p == len(param_labels) - 1:
            ax.legend()

    plt.suptitle(r'data dim $= %d; N = %d; \epsilon = %.2f$' % (data_dim, N, epsilon))
    plt.show()