import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
        encoder: r = h(x,y)
    """
    def __init__(self, x_dim, y_dim, r_dim, hidden_layer_dims):
        super(Encoder, self).__init__()
        layer_dim_list = [x_dim+y_dim] + hidden_layer_dims + [r_dim]
        self.module_list = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1])
                                          for i in range(len(layer_dim_list)-1)])

    def forward(self, x, y):
        output = torch.cat((x, y), dim=-1)
        for f in self.module_list[:-1]:
            output = F.relu(f(output))
        output = self.module_list[-1](output)
        return output


class ZMeanStdFunc(nn.Module):
    """
        z ~ N(mu(r), sigma(r))
    """
    def __init__(self, r_dim, z_dim):
        super(ZMeanStdFunc, self).__init__()
        self.fc_mean = nn.Linear(r_dim, z_dim)
        self.fc_std = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        mu = self.fc_mean(r)
        std = nn.functional.softplus(self.fc_std(r))
        return mu, std


class Decoder(nn.Module):
    """
        y = g(z, x)
    """

    def __init__(self, x_dim, y_dim, z_dim, hidden_layer_dims, sigma=0.05):
        super(Decoder, self).__init__()
        layer_dim_list = [x_dim + z_dim] + hidden_layer_dims
        self.module_list = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1])
                                                    for i in range(len(layer_dim_list)-1)])
        self.m_layer = nn.Linear(hidden_layer_dims[-1], y_dim)
        self.sigma = sigma

    def forward(self, z, x_target):
        # z : [batch, N_z, z_dim] -> [batch, N_x, N_z, z_dim]
        z_exd = z.unsqueeze(dim=-3).expand(-1, x_target.size(1), -1, -1)
        # x_target : [batch, N_x, x_dim] -> [batch, N_x, N_z, x_dim]
        x_target_exd = x_target.unsqueeze(dim=-2).expand(-1, -1, z.size(1), -1)

        # [x_target_N, z_N, y_dim]
        output = torch.cat((x_target_exd, z_exd), dim=-1)

        for f in self.module_list:
            output = F.relu(f(output))
        mu = torch.sigmoid(self.m_layer(output))
        #mu = self.m_layer(output)
        std = torch.ones_like(mu)*self.sigma
        return mu, std


class NPModel(nn.Module):
    """ Super class to contain all models for NeuralProcess
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, enc_hidden_dims, dec_hidden_dims):
        super().__init__()

        self.__dict__.update(locals())
        self.Encoder = Encoder(x_dim, y_dim, r_dim, enc_hidden_dims)
        self.ZMeanStdFunc = ZMeanStdFunc(r_dim, z_dim)
        self.Decoder = Decoder(x_dim, y_dim, z_dim, dec_hidden_dims)
        self.z_dim = z_dim

    def init(self, device, ngpu):
        self.Encoder = self.Encoder.to(device)
        self.ZMeanStdFunc = self.ZMeanStdFunc.to(device)
        self.Decoder = self.Decoder.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            self.Encoder = nn.DataParallel(self.Encoder, list(range(ngpu)))
            self.ZMeanStdFunc = nn.DataParallel(self.ZMeanStdFunc, list(range(ngpu)))
            self.Decoder = nn.DataParallel(self.Decoder, list(range(ngpu)))

    def uninit(self):
        if hasattr(self.Encoder, 'module'):
            self.Encoder = self.Encoder.module.cpu()
            self.ZMeanStdFunc = self.ZMeanStdFunc.module.cpu()
            self.Decoder = self.Decoder.module.cpu()


def get_npmodel(x_dim, y_dim):
    return NPModel(x_dim, y_dim, 128, 128, [128, 128, 128], [128, 128])
    #return NPModel(x_dim, y_dim, 4, 4, [8], [8])
