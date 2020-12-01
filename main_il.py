import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from a2c_ppo_acktr.model import DMPNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, DiagGaussianDist
from a2c_ppo_acktr.utils import init
from dmp.utils.dmp_layer import DMPIntegrator, DMPParameters
from a2c_ppo_acktr import pytorch_util as ptu
import cv2
from dmp.utils.smnist_loader import MatLoader
from os.path import dirname, realpath
import os
import sys
import argparse
from datetime import datetime
from dmp.utils.mnist_cnn import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='ndp-il')
args = parser.parse_args()


class NdpCnn(nn.Module):
    def __init__(
            self,
            init_w=3e-3,
            layer_sizes=[784, 200, 100],
            hidden_activation=F.relu,
            pt='./dmp/data/pretrained_weights.pt',
            output_activation=None,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            state_index=np.arange(1),
            N = 5,
            T = 10,
            l = 10,
            *args,
            **kwargs
    ):
        super().__init__()
        self.N = N
        self.l = l
        self.output_size = N*len(state_index) + 2*len(state_index)
        output_size = self.output_size
        self.T = T
        self.output_activation=torch.tanh
        self.state_index = state_index
        self.output_dim = output_size
        tau = 1
        dt = 1.0 / (T*self.l)
        self.output_activation=torch.tanh
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None)
        self.func = DMPIntegrator()
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)
        layer_sizes=[784, 200, 100, 200, 2*output_size, output_size]
        self.input_size = input_size
        self.hidden_activation = hidden_activation
        in_size = input_size
        self.pt = CNN()
        self.pt.load_state_dict(torch.load(pt))
        self.convSize = 4*4*50
        self.imageSize = 28
        self.N = N
        self.middle_layers = []
        for i in range(1, len(layer_sizes)-1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            hidden_init(layer.weight)
            layer.bias.data.fill_(b_init_value)
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(layer_sizes[-1], output_size))



    def forward(self, input, y0, return_preactivations=False):
        x = input
        x = x.view(-1, 1, self.imageSize, self.imageSize)
        x = F.relu(self.pt.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu((self.pt.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        activation_fn = self.hidden_activation
        x = activation_fn(self.pt.fc1(x))
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        output = self.last_fc(x)*1000
        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0)
        y = y.view(input.shape[0], len(self.state_index), -1)
        return y.transpose(2, 1)





data_path = './dmp/data/40x40-smnist.mat'
images, outputs, scale, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
images = np.array([cv2.resize(img, (28, 28)) for img in images])/255.0
input_size = images.shape[1] * images.shape[2]

inds = np.arange(12000)
np.random.shuffle(inds)
test_inds = inds[10000:]
train_inds = inds[:10000]
X = torch.Tensor(images[:12000]).float()
Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:12000]


time = str(datetime.now())
time = time.replace(' ', '_')
time = time.replace(':', '_')
time = time.replace('-', '_')
time = time.replace('.', '_')
model_save_path = './dmp/data/' + args.name + '_' + time
os.mkdir(model_save_path)
k = 1
T = 300/k
N = 30
learning_rate = 1e-3
Y = Y[:, ::k, :]


X_train = X[train_inds]
Y_train = Y[train_inds]
X_test = X[test_inds]
Y_test = Y[test_inds]



num_epochs = 71
batch_size = 100
ndpn = NdpCnn(T=T,l=1, N=N, state_index=np.arange(2))
optimizer = torch.optim.Adam(ndpn.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    inds = np.arange(X_train.shape[0])
    np.random.shuffle(inds)
    for ind in np.split(inds, len(inds)//batch_size):
        optimizer.zero_grad()
        y_h = ndpn(X_train[ind], Y_train[ind, 0, :])
        loss = torch.mean((y_h - Y_train[ind])**2)
        loss.backward()
        optimizer.step()
    torch.save(ndpn.state_dict(), model_save_path + '/model.pt')
    if epoch % 20 == 0:
        x_test = X_test[np.arange(100)]
        y_test = Y_test[np.arange(100)]
        y_htest = ndpn(x_test, y_test[:, 0, :])
        for j in range(18):
            plt.figure(figsize=(8, 8))
            plt.plot(0.667*y_h[j, :, 0].detach().cpu().numpy(), -0.667*y_h[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
            plt.axis('off')
            plt.savefig(model_save_path + '/train_img_' + str(j) + '.png')

            plt.figure(figsize=(8, 8))
            img = X_train[ind][j].cpu().numpy()*255
            img = np.asarray(img*255, dtype=np.uint8)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(model_save_path + '/ground_train_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

            plt.figure(figsize=(8, 8))
            plt.plot(0.667*y_htest[j, :, 0].detach().cpu().numpy(), -0.667*y_htest[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
            plt.axis('off')
            plt.savefig(model_save_path + '/test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

            plt.figure(figsize=(8, 8))
            img = X_test[j].cpu().numpy()*255
            img = np.asarray(img*255, dtype=np.uint8)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(model_save_path + '/ground_test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
        test = ((y_htest - y_test)**2).mean(1).mean(1)
        print('Epoch: ' + str(epoch) + ', Test Error: ' + str(test.mean().item()))
