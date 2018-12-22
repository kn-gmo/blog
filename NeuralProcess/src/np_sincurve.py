import torch
import numpy as np
from neural_process_model import get_npmodel
from neural_process_trainer import NeuralProcessTrainer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import matplotlib as mpl
#mpl.use('Agg')

def get_sampler(n):
    def sin_sampler():
        x = torch.linspace(0, 1, n).unsqueeze(-1) * 8.0 - 4.0 # x in uniform[-4, 4]
        a = torch.rand(1) * 4.0 - 2.0 # a from uniform[-2,2]
        noise = torch.randn_like(x) * 0.05
        y = a * np.sin(x) + noise
        return x, y
    return sin_sampler

class SimpleCurveDataSet(Dataset):
    def __init__(self, sampler, size, store_sample=True):
        super().__init__()
        self.size = size
        self.sampler = sampler
        self.store_sample = store_sample
        if self.store_sample:
            self.sample = [self.sampler() for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if self.store_sample:
            return self.sample[item]
        else:
            return self.sampler()


def main():
    sampler = get_sampler(200)
    batch_size = 32
    train_dset = SimpleCurveDataSet(sampler, batch_size*200, store_sample=False)
    train_iter = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=2)
    result_dir = './viz/sincurve'

    model = get_npmodel(1, 1)
    trainer = NeuralProcessTrainer(model, learning_rate=1e-3, ngpu=2, viz_dir=result_dir, load_model=False)

    n_context = 50
    epoch = 100
    trainer.train(epoch, train_iter=train_iter)

    fig = plt.figure()
    for i in range(2):
        x_all, y_all = iter(train_iter).next()
        ind = torch.randperm(x_all.size(1))[:n_context]
        x_o = x_all[:,ind, :]
        y_o = y_all[:,ind, :]
        y_mu, y_std = trainer.posterior_predict(x_o, y_o, x_all)
        plt.plot(x_o[0,:,0].numpy(), y_o[0,:,0].numpy(), '.',  markersize=5, color='b')
        for iz in range(1):
            plt.plot(x_all[0, :, 0].numpy(), y_all[0, :, 0].numpy(), "-" , linewidth=0.5)
            plt.plot(x_all[0, :, 0].numpy(), y_mu[0, :, 0, 0].numpy(), "-" , linewidth=0.5)
        plt.title("n_obs = {}".format(n_context))
        plt.show()

if __name__ == "__main__":
    main()
