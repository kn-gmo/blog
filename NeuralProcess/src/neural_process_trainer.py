import torch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt

from neural_process_model import get_npmodel
#from NeuralProcess.src.neural_process_model import get_npmodel

class NeuralProcessTrainer(object):

    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        params = list(self.model.Encoder.parameters()) \
               + list(self.model.ZMeanStdFunc.parameters()) \
               + list(self.model.Decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def _calc_zparam(self, x, y):
        output = self.model.Encoder(x, y)
        output = torch.mean(output, dim=-2)  # aggregate for r_N
        output = self.model.ZMeanStdFunc(output)
        return output

    def _sample_z(self, mu, std, z_draw):
        mu = mu.unsqueeze(dim=-2).expand(z_draw, -1)
        eps = torch.autograd.Variable(std.data.new(mu.shape).normal_())
        return mu + std * eps

    def _loss_function(self, x_context, y_context, x_target, y_target, z_draw):

        x_all = torch.cat([x_context, x_target], dim=-2)
        y_all = torch.cat([y_context, y_target], dim=-2)

        (z_all_mu, z_all_sigma) = self._calc_zparam(x_all, y_all)
        (z_c_mu, z_c_sigma) = self._calc_zparam(x_context, y_context)
        z_all_sample = self._sample_z(z_all_mu, z_all_sigma, z_draw)
        (y_mu, y_sigma) = self.model.Decoder(z_all_sample, x_target)
        y_normal = Normal(loc=y_mu, scale=y_sigma)

        y_target_exp = y_target.unsqueeze(dim=-2).expand(-1, z_draw, -1)  # dim is expanded to [target_N,  z_draw, y_dim]
        loglik = y_normal.log_prob(y_target_exp).mean(dim=-2).sum(dim=(-1,-2)) # sum for y_dim, y_target, mean for z_draw
        kldiv = NeuralProcessTrainer.calc_kldiv_gaussian(z_all_mu, z_all_sigma, z_c_mu, z_c_sigma)
        loss = -loglik + kldiv
        return loss

    def train(self, x, y, z_draw=10):
        """

        :param x: [N, x_dim]
        :param y: [N, y_dim]
        :param z_draw: the number of sampling of z
        :return:
        """

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)

        ctx_N = np.random.randint(1, len(x_tensor))

        index_random = torch.randperm(len(x_tensor))
        x_ctx = x_tensor[index_random[:ctx_N]]
        y_ctx = y_tensor[index_random[:ctx_N]]
        x_tgt = x_tensor[index_random[ctx_N:]]
        y_tgt = y_tensor[index_random[ctx_N:]]

        self.optimizer.zero_grad()
        loss = self._loss_function(x_ctx, y_ctx, x_tgt, y_tgt, z_draw)
        loss.backward()
        train_loss = loss.item()
        self.optimizer.step()
        return train_loss

    def posterior_predict(self, x_obs, y_obs, x_new, z_draw=10):
        """

        :param x_obs: [N_obs, x_dim]
        :param y_obs: [N_obs, y_dim]
        :param x_new: [N_new, x_dim]
        :param z_draw:
        :return:
        """

        x_obs_tensor = torch.from_numpy(x_obs)
        y_obs_tensor = torch.from_numpy(y_obs)
        x_new_tensor = torch.from_numpy(x_new)

        if len(x_obs_tensor) == 0:
            # prior dist
            z_dim = self.model.z_dim
            z_normal = Normal(loc=torch.zeros(z_dim), scale=1.0)
            z_all_sample = z_normal.sample(sample_shape=(z_draw,))
        else:
            z_mu, z_sigma = self._calc_zparam(x_obs_tensor, y_obs_tensor)
            z_all_sample = self._sample_z(z_mu, z_sigma, z_draw)

        (y_mu, y_sigma) = self.model.Decoder(z_all_sample, x_new_tensor)
        return y_mu.data.numpy()

    @staticmethod
    def calc_kldiv_gaussian(q_mu, q_std, p_mu, p_std):
        q_var = q_std ** 2
        p_var = p_std ** 2
        p_var_dev = 1. / p_var
        ret = (q_var + (q_mu - p_mu) ** 2) * p_var_dev - torch.log(q_var * p_var_dev) - 1.
        ret = 0.5 * ret.sum()
        return ret

def sample_constant(n):
    #x = np.linspace(-4, 4, n).reshape(n,1).astype(np.float32)
    x = np.random.uniform(low=-4.0, high=4.0, size=(n,1)).astype(np.float32)
    y = np.sin(x).astype(np.float32)
    return x, y

def sample_random(n):
    x = np.random.uniform(low=-4.0, high=4.0, size=(n,1)).astype(np.float32)
    #x = np.linspace(-4, 4, n).reshape(n,1).astype(np.float32)
    a = np.random.uniform(low=-2., high=2.)
    y = (a * np.sin(x)).astype(np.float32)
    return x, y

def train():
    model = get_npmodel(1, 1)
    obj = NeuralProcessTrainer(model)
    train_iter = int(1e5)
    log_interval = int(5e3)

    loss = 0.
    for ite in range(train_iter):
        x, y = sample_constant(10)
        x, y = sample_random(10)
        if ite % log_interval == 0:
            print("{} times Loss: {}".format(ite, loss))
            x_t = np.linspace(-6, 6, 100).reshape(100,1).astype(np.float32)
            y_t_pred = obj.posterior_predict(x, y, x_t, 20)
            plt.plot(x, y, '.')
            for i in range(y_t_pred.shape[1]):
                plt.plot(x_t, y_t_pred[:, i, 0], color="gray", alpha=0.5)
                plt.title("Trained after {} times".format(ite))
            plt.pause(.1)
            plt.clf()

        loss = obj.train(x, y)

    return obj

def show_fig(obj, n_obs_list):
    plt.figure()
    for i, n in enumerate(n_obs_list):
        plt.subplot(2, 2, i+1)

        #x_o, y_o = sample_constant(n)
        x_o, y_o = sample_random(n)

        plt.plot(x_o, y_o, 'o', color="black")

        x_t = np.linspace(-6, 6, 100).reshape(100, 1).astype(np.float32)
        y_t_pred = obj.posterior_predict(x_o, y_o, x_t, 20)
        y_t = np.sin(x_t)
        #plt.plot(x_t, y_t, '-', color="black")
        for i in range(y_t_pred.shape[1]):
            plt.plot(x_t, y_t_pred[:, i, 0], alpha=0.5)
        plt.title("n_obs = {}".format(n))
    plt.show()

if __name__ == "__main__":
    obj = train()
    show_fig(obj, [0, 2, 5, 10])
    show_fig(obj, [0, 1, 1, 1])
