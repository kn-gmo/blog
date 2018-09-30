# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

class GaussianProcessRegression(object):
    """
        Gaussian Process Regression
    """

    def __init__(self, x_train, y_train, kernel_func, noise_var= 1e-3):
        """

        :param x_train: observed input [N_obs, x_dim]
        :param y_train: observed output [N_obs]
        :param kernel_func: kernel function
        :param noise_var: The noise of the prediction.
                          This value is added to the diagonal element of
                          the covariance matrix.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.kernel_func = kernel_func
        self.noise_var = noise_var
        covar = self.__calc_covar_mat(self.x_train, self.x_train)
        self.covar_inv = covar.I

    def __calc_covar_mat(self, x_row_vec, x_col_vec):
        covar_mat = np.matrix(np.zeros((len(x_row_vec), len(x_col_vec))))
        for ir, xr in enumerate(x_row_vec):
            for ic, xc in enumerate(x_col_vec):
                c = self.kernel_func(xr, xc)
                if ir == ic:  c += self.noise_var
                covar_mat[ir, ic] = c
        return covar_mat

    def calc_mean_and_covariance_from_prior(self, x_vec):
        covar = self.__calc_covar_mat(x_vec, x_vec)
        mean = np.zeros(len(x_vec))
        return (mean, covar)

    def calc_mean_and_covariance(self, x_vec):
        assert len(self.x_train) != 0
        var_vec = self.__calc_covar_mat(x_vec, self.x_train)
        y_train = np.matrix(self.y_train).T
        mean = (var_vec * self.covar_inv * y_train)
        mean = np.squeeze(np.asarray(mean))
        c = self.__calc_covar_mat(x_vec, x_vec)
        var = c - var_vec * self.covar_inv * var_vec.T
        return (mean, var)

    def sample(self, x_vec):
        if len(self.x_train) == 0:
            mean, covar = self.calc_mean_and_covariance_from_prior(x_vec)
        else:
            mean, covar = self.calc_mean_and_covariance(x_vec)
        return np.random.multivariate_normal(mean, covar)


def plot_fig():
    param = (1.0, 1.0)
    kernel_func = lambda x, y: param[0] * math.exp(-np.sum((x - y) ** 2)/param[1])
    #kernel_func = lambda x, y : param[0] * math.exp(-math.sqrt(np.sum((x - y) ** 2))/param[1])

    n_obs_list = [0, 2, 5, 10]
    for i, n in enumerate(n_obs_list):
        plt.subplot(2, 2, i+1)
        x_o = np.random.uniform(low=-4.0, high=4.0, size=(n, 1))
        y_o = np.sin(x_o).squeeze(-1)

        obj = GaussianProcessRegression(x_o, y_o, kernel_func, 0.01)
        x_t = np.linspace(-6, 6, 100)
        y_t = np.sin(x_t)

        plt.plot(x_o,y_o, 'o', color="black")
        plt.plot(x_t, y_t, linewidth=1.0, color="black")
        for _ in range(20):
            y_t_pred = obj.sample(x_t)
            plt.plot(x_t, y_t_pred, "-", alpha=0.3)
        plt.title("n_obs = {}".format(n))
    plt.show()

if __name__ == "__main__":
    np.random.seed(100)
    plot_fig()