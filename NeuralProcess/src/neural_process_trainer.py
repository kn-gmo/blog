import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import os


class NeuralProcessTrainer(object):

    def __init__(self, model, ngpu, z_draw=20, learning_rate=1e-3, viz_dir=None, load_model=True):

        self.ngpu = ngpu
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0)
                                   else "cpu")
        print("device: {}, gpu: {}".format(self.device, self.ngpu))

        self.viz_dir = viz_dir
        self.model = model

        if load_model:
            self.load_model()
        else:
            self.model.init(self.device, self.ngpu)


        params = list(self.model.Encoder.parameters()) \
               + list(self.model.ZMeanStdFunc.parameters()) \
               + list(self.model.Decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        self.N_z = z_draw
        self.train_losses = []
        self.count_epochs = 1
        print(self.model)
        if self.viz_dir is not None:
            os.makedirs(self.viz_dir, exist_ok=True)

    def _calc_zparam(self, x, y):
        """

        :param x: [batch, N_sample, x_dim]
        :param y: [batch, N_sample, y_dim]
        :return: mean and std of the noraml dist of z_param [batch, z_dim]
        """
        r = self.model.Encoder(x, y)
        r_mean = torch.mean(r, dim=-2)  # aggregate for N_sample
        mu, std = self.model.ZMeanStdFunc(r_mean)
        return mu, std

    def _sample_z(self, mu, std, N_z):
        mu = mu.unsqueeze(dim=-2).expand(-1, N_z, -1)
        std = std.unsqueeze(dim=-2).expand(-1, N_z, -1)
        eps = torch.randn_like(mu, device=self.device)
        return mu + std * eps

    def _loss_function(self, x_context, y_context, x_target, y_target):
        """

        :param x_context:  [batch, N_con, x_dim]
        :param y_context:  [batch, N_con, y_dim]
        :param x_target:   [batch, N_tar, x_dim]
        :param y_target:  [batch, N_tar, x_dim]
        :return:
        """
        x_all = torch.cat([x_context, x_target], dim=-2)
        y_all = torch.cat([y_context, y_target], dim=-2)

        z_all_mu, z_all_sigma = self._calc_zparam(x_all, y_all) # z_all [batch, z_dim]
        z_c_mu, z_c_sigma = self._calc_zparam(x_context, y_context) # z_c [batch, z_dim]
        z_all_sample = self._sample_z(z_all_mu, z_all_sigma, self.N_z) # z_all_sample [batch, N_z, z_dim]
        y_t_mu, y_t_sigma = self.model.Decoder(z_all_sample, x_target) # y_t [batch, N_t, N_z, y_dim]
        y_normal = Normal(loc=y_t_mu, scale=y_t_sigma)
        y_target_exd = y_target.unsqueeze(dim=-2).expand(-1, -1, self.N_z, -1)  # to [batch, N_t,  N_z, y_dim]
        loglik = -y_normal.log_prob(y_target_exd).sum(dim=[-1,-3]).mean() # sum for N_t, y_dim, mean for N_z, batch
        kldiv = NeuralProcessTrainer.calc_kldiv_gaussian(z_all_mu, z_all_sigma, z_c_mu, z_c_sigma) # sum for z_dim, mean for batch
        return loglik, kldiv

    def calc_batch_loss(self, batch):
        """
        :param z_draw: the number of sampling of z
        :return:
        """
        x, y = batch
        x = x.to(self.device) #[batch, N, x_dim]
        y = y.to(self.device) #[batch, N, y_dim]
        ctx_N = np.random.randint(3, x.size(1))
        index_random = torch.randperm(x.size(1))
        x_ctx = x[:, index_random[:ctx_N], :]
        y_ctx = y[:, index_random[:ctx_N], :]
        x_tgt = x[:, index_random[ctx_N:], :]
        y_tgt = y[:, index_random[ctx_N:], :]
        l_loglik, l_kldiv = self._loss_function(x_ctx, y_ctx, x_tgt, y_tgt)
        return l_loglik, l_kldiv

    def train(self, num_epochs, train_iter):
        self.model.train()
        min_loss = 1e9
        for epoch in range(num_epochs):
            train_losses_eopch = []
            for i, batch in enumerate(train_iter):
                #print("Epch[%d] %d/%d" %(epoch, i, len(train_iter)))
                self.optimizer.zero_grad()
                loss_loglik, loss_kldiv = self.calc_batch_loss(batch)
                loss = loss_loglik + loss_kldiv
                loss.backward()
                self.optimizer.step()
                train_losses_eopch.append((loss.item(), loss_loglik.item(), loss_kldiv.item()))
            tot_mloss, lglk_mloss, kld_mloss = np.mean(train_losses_eopch, axis=0)
            print("Epoch[%d]: Total loss: %.4f, Loglike part: %.4f, KLDiv part: %.4f" %(self.count_epochs, tot_mloss,
                  lglk_mloss, kld_mloss))

            if tot_mloss < min_loss:
                self.save_model()
                min_loss = tot_mloss
                print("model is saved!""")

            self.count_epochs += 1
            self.train_losses.extend(train_losses_eopch)

    def posterior_predict(self, x_obs, y_obs, x_new, z_draw=20):
        """

        :param x_obs: [batch, N_obs, x_dim]
        :param y_obs: [batch, N_obs, y_dim]
        :param x_new: [batch, N_new, x_dim]
        :return: return mean[batch, N_new, N_z, y_dim] of posterior dist of y
        """
        self.model.eval()
        if len(x_obs) == 0:
            # prior dist
            z_dim = self.model.z_dim
            z_normal = Normal(loc=torch.zeros(z_dim), scale=1.0)
            z_all_sample = z_normal.sample(sample_shape=(z_draw,))
        else:
            z_mu, z_sigma = self._calc_zparam(x_obs, y_obs)
            z_all_sample = self._sample_z(z_mu, z_sigma, z_draw)

        y_mu, y_sigma = self.model.Decoder(z_all_sample, x_new)
        return y_mu.detach(), y_sigma.detach()

    def viz_loss(self):
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8,6)
        labels=["Total", "LogLike part", "KLDiv part"]
        train_losses = np.array(self.train_losses)
        for i, label in enumerate(labels):
            plt.plot(np.linspace(1, self.count_epochs, len(train_losses)),
                 train_losses[:,i], "-", label=label)
        plt.legend()
        fig.savefig(self.viz_dir + '/loss.png')

    @staticmethod
    def calc_kldiv_gaussian(q_mu, q_std, p_mu, p_std):
        """
            D_kl(q|p)
        """
        q_var = q_std ** 2
        p_var = p_std ** 2
        p_var_dev = 1. / p_var
        ret = (q_var + (q_mu - p_mu) ** 2) * p_var_dev - torch.log(q_var * p_var_dev) - 1.
        ret = 0.5 * ret.sum(dim=-1).mean() #sum for z_dim, mean for batch
        return ret

    def save_model(self):
        savepath = self.viz_dir + '/model.pt'
        self.model.uninit()
        torch.save(self.model.state_dict(), savepath)
        self.model.init(self.device, self.ngpu)

    def load_model(self):
        load_path = self.viz_dir + '/model.pt'
        if os.path.isfile(load_path):
            state = torch.load(load_path)
            self.model.load_state_dict(state)
        self.model.init(self.device, self.ngpu)
