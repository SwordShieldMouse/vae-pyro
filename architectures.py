import torch
import pyro
import os
import numpy as np
import pyro.distributions as dist
import torch.nn as nn
import torchvision
import pyro.optim

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

class Decoder(nn.Module):
    # the posterior distribution of x given z
    # takes in a code z and outputs parameter of p(x)
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 784) # 784 is the MNIST image size

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        hidden = self.leaky_relu(self.fc1(z))
        output = self.sigmoid(self.fc2(hidden)) # sigmoid since pixel values are Bernouilli distributed?
        return output

class Encoder(nn.Module):
    # the posterior dist of z given x
    # takes in an input x and outputs parameters for a code z
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = x.reshape(-1, 784)
        hidden = self.leaky_relu(self.fc1(x))

        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden)) # use exp since scale is positive

        return z_loc, z_scale


class VAE(nn.Module):
    def __init__(self, z_dim = 50, hidden_dim = 400):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

    # the stochastic model p(x|z)p(z)
    # with data, generate codes, then generate images
    def model(self, x):
        pyro.module("decoder", self.decoder) # so that pyro knows about parameters in decoder
        with pyro.plate("data", x.shape[0]):
            # hyperparameters for prior p(z)
            z_loc = x.new_zeros([x.shape[0], self.z_dim])
            z_scale = x.new_ones([x.shape[0], self.z_dim])
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1)) # dependent within rows

            # decode the latent z
            p = self.decoder(z) # why call forward explicitly instead of just __call__?

            # score against real images
            pyro.sample("obs", dist.Bernoulli(p).to_event(1), obs = x.reshape(-1, 784)) # dependent within rows

    # the variational distribution q(z|x)
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
