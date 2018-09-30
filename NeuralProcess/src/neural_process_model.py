import torch
from torch import nn


class Encoder(nn.Module):
    """
        encoder: r = h(x,y)
    """
    def __init__(self, layer_dim_list, ):
        super(Encoder, self).__init__()
        self.a = nn.ReLU()
        self.module_list = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1])
                                          for i in range(len(layer_dim_list)-1)])

    def forward(self, x, y):
        output = torch.cat((x, y), dim=-1)
        for f in self.module_list[:-1]:
            output = self.a(f(output))
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

    def __init__(self, layer_dim_list):
        super(Decoder, self).__init__()
        self.a = nn.Sigmoid()
        self.module_list = nn.ModuleList([nn.Linear(layer_dim_list[i], layer_dim_list[i+1])
                                                    for i in range(len(layer_dim_list)-1)])

    def forward(self, z, x_target):
        # x_target : [x_target_N, x_dim]
        # z : [z_N, z_dim]
        # x_target_N = x_target.size(0)
        # z_N = z.size(0)
        z_exp = z.unsqueeze(dim=0).expand(x_target.shape[0], -1, -1)
        x_target_exp = x_target.unsqueeze(dim=1).expand(-1, z.shape[0], -1)

        # [x_target_N, z_N, y_dim]
        output = torch.cat((x_target_exp, z_exp), dim=-1)

        for f in self.module_list[:-1]:
            output = self.a(f(output))
        mu = self.module_list[-1](output)

        return mu, 0.05


class NPModel(nn.Module):
    """ Super class to contain all models for NeuralProcess
    """
    def __init__(self, x_dim, y_dim, r_dim, z_dim, enc_hidden_dims, dec_hidden_dims):
        super().__init__()

        self.__dict__.update(locals())
        encoder_layer_dims = [x_dim+y_dim] + enc_hidden_dims + [r_dim]
        self.Encoder = Encoder(encoder_layer_dims)
        self.ZMeanStdFunc = ZMeanStdFunc(r_dim, z_dim)
        decoder_layer_dims = [x_dim+z_dim] + dec_hidden_dims + [y_dim]
        self.Decoder = Decoder(decoder_layer_dims)
        self.z_dim = z_dim


def get_npmodel(x_dim, y_dim):
    return NPModel(x_dim, y_dim, 4, 4, [8], [8])
