#for server side
import matplotlib as mpl
#mpl.use('Agg')

import torch
import numpy as np
import argparse
import random
from neural_process_model import get_npmodel
from neural_process_trainer import NeuralProcessTrainer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ImageDataSetWrapper(Dataset):

    def __init__(self, image_dset, size):
        super().__init__()
        image, _ = image_dset[0]
        self.c, self.h, self.w = image.shape
        self.x = self.get_x()
        self.size = size
        self.image_dset = image_dset

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        image = self.get_original(item)
        y = self.trans_fig(image)
        return self.x, y

    def get_x(self):
        h = self.h
        w = self.w
        x = torch.zeros(h*w, 2)
        for ih in range(h):
            for jw in range(w):
                n = ih * w + jw
                x[n, 0] =  float(ih) / h
                x[n, 1] =  float(jw) / w
        return x

    def trans_fig(self, image):
        """

        :param image: [c, h, w] -
        :return: y[h*w, c]
        """
        y = image.reshape(self.c, self.h*self.w).transpose(0, 1)
        return y

    def restore_fig(self, y):
        """
        :param y: [n, c]
        :param h:
        :return: [3, h, w] to rgb
        """
        image = y.transpose(0,1).reshape(self.c, self.h, self.w).expand(3,-1,-1)
        return image

    def restore_partial_fig(self, x, y):
        image = torch.zeros(3, self.h, self.w)
        image[2,:,:] = 1.0
        for n, ix in enumerate(x):
            ih = int(ix[0] * self.h)
            iw = int(ix[1] * self.w)
            image[:, ih, iw] = y[n, :]
        return image

    def get_original(self, item):
        image, _ = self.image_dset[item]
        return image

    def get_nc(self):
        return self.c



def get_dset(type, datadir, size):
    if type == 'mnist':
        image_dset = datasets.MNIST(root=datadir, download=True,
                            transform=transforms.Compose([
                                  #transforms.Resize(1,20,10),
                                  transforms.ToTensor(),
                                ]))
    elif type == 'celeba':
        image_dset = datasets.ImageFolder(root=datadir + '/celeba',
                transform=transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    ]))
    else:
        raise NotImplementedError
    dset_wrapper = ImageDataSetWrapper(image_dset, size)
    return dset_wrapper


def check_image(dset_wrapper):
    n_context = 64 * 10
    item = np.random.randint(0, len(dset_wrapper))
    true_image = dset_wrapper.get_original(item)
    print("size:",len(image_dset), "shape: ",true_image.shape)
    x_all, y_all = dset_wrapper[item]
    index_random = torch.randperm(x_all.size(0))[:n_context]
    x_c = x_all[index_random]
    y_c = y_all[index_random]
    res_image  = dset_wrapper.restore_partial_fig(x_c, y_c)
    #res_image = dset_wrapper.restore_fig(y_all)
    to_img = transforms.ToPILImage()
    plt.imshow(to_img(res_image))
    plt.title("restore")
    plt.show()
    plt.imshow(to_img(true_image))
    plt.title("true")
    plt.show()


def show_fig(dset_wrapper, n_context, n_sample, trainer):
    fig = plt.figure()
    train_iter = DataLoader(dset_wrapper, batch_size=n_sample, shuffle=True, num_workers=2)
    x_all, y_all = iter(train_iter).next()
    index_random = torch.randperm(x_all.size(1))[:n_context]
    x_c = x_all[:, index_random]
    y_c = y_all[:, index_random]
    z_draw = 200
    y_mu, y_std = trainer.posterior_predict(x_c, y_c, x_all, z_draw=z_draw)
    to_img = transforms.ToPILImage()

    images = []
    for i_sample in range(n_sample):
        images.append(dset_wrapper.restore_fig(y_all[i_sample]))
        images.append(dset_wrapper.restore_partial_fig(x_c[i_sample], y_c[i_sample]))
        for iz in range(3):
            images.append(dset_wrapper.restore_fig(y_mu[i_sample, :,iz, :].cpu()))
    plt.imshow(to_img(utils.make_grid(images,nrow=5, padding=1, normalize=True)))
    plt.title('n_context={}'.format(n_context))
    fig.savefig(result_dir + '/sample_{}.png'.format(n_context))
    #plt.show()

def show_movie(dset_wrapper, n_context, n_sample, trainer, result_dir):
    fig = plt.figure()

    train_iter = DataLoader(dset_wrapper, batch_size=n_sample, shuffle=True, num_workers=2)
    x_all, y_all = iter(train_iter).next()
    index_random = torch.randperm(x_all.size(1))[:n_context]
    x_c = x_all[:, index_random]
    y_c = y_all[:, index_random]
    z_draw = 200
    y_mu, y_std = trainer.posterior_predict(x_c, y_c, x_all, z_draw=z_draw)

    to_img = transforms.ToPILImage()

    images = []
    for i_sample in range(n_sample):
        images.append(dset_wrapper.restore_fig(y_all[i_sample]))
        images.append(dset_wrapper.restore_partial_fig(x_c[i_sample], y_c[i_sample]))
        images.append(None) # for prediction fig

    def update(iz):
        for i_sample in range(n_sample):
            images[2+i_sample*3] = dset_wrapper.restore_fig(y_mu[i_sample,:,iz, :].cpu())
        plt.clf()
        plt.imshow(to_img(utils.make_grid(images, nrow=3, padding=1, normalize=True)))
        plt.title('n_context={}'.format(n_context))
    ani = animation.FuncAnimation(fig, update, range(z_draw), interval=100)
    ani.save(result_dir + '/sample_{}.gif'.format(n_context))
    #plt.show()


if __name__ == "__main__":
    n_dataset = 60000

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', required=True, help='train|evaluate')
    parser.add_argument('-dt','--data_type', required=True, help='mnist|celeba')
    parser.add_argument('-dr','--data_root', default="../../data", help='data directory')
    parser.add_argument('-ng','--ngpu', default=0, type=int, help='number of GPUs to use')
    parser.add_argument('-lr','--learning_rate', default=2e-4, type=float, help='learning rate, default=0.0002')
    parser.add_argument('-b','--batch', default=64, type=int, help='batch size, default=64')
    parser.add_argument('-e','--epoch', default=100, type=int, help='the number of epoch, default=100')
    parser.add_argument('-nc','--n_context', default=100, type=int, help='the number of context only for evaluation, default=100')
    parser.add_argument('-ms','--manual_seed', type=int, help='manual seed')


    args = parser.parse_args()
    data_type = args.data_type.lower()
    result_dir = './viz/{}'.format(data_type)

    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    torch.manual_seed(args.manual_seed)

    image_dset = get_dset(data_type, args.data_root, size=n_dataset)
    nc = image_dset.get_nc()
    model = get_npmodel(2, nc)
    trainer = NeuralProcessTrainer(model, learning_rate=args.learning_rate, ngpu=args.ngpu, viz_dir=result_dir, load_model=True)

    if args.mode == 'train':
        train_iter = DataLoader(image_dset, batch_size=args.batch, shuffle=True, num_workers=2)
        trainer.train(args.epoch, train_iter=train_iter)
    elif args.mode == 'evaluate':
        #show_movie(image_dset, args.n_context, 5, trainer, result_dir)
        show_fig(image_dset, args.n_context, 5, trainer)
    else:
        pass
