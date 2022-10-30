#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class Multi_head_ExternalAttention(nn.Module):
    """
    # Input : x , an array with shape [B, N, C_in]
    # (batchsize, pixels, channels)
    # Parameter: M_K, a linearlayer
    # Parameter: M_V, a linearlayer
    # Parameter: heads number of heads
    # Output : out , an array with shape [B, N, C_in]
    """

    def __init__(self, channels, external_memory, point_num, heads):
        super(Multi_head_ExternalAttention, self).__init__()
        self.channels = channels
        self.external_memory = external_memory
        self.point_num = point_num
        self.heads = heads
        self.Q_linear = nn.Linear(self.channels, self.channels)
        self.K_linear = nn.Linear(self.channels // self.heads, self.external_memory)
        self.softmax = nn.Softmax(dim=2)
        self.LNorm = nn.LayerNorm([self.heads, self.point_num, self.external_memory])
        self.V_Linear = nn.Linear(self.external_memory, self.channels // self.heads)
        self.O_Linear = nn.Linear(self.channels, self.channels)

    def forward(self, x):
        x = x.transpose(2, 1)
        B, N, C = x.shape
        x = self.Q_linear(x).contiguous().view(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attention = self.LNorm(self.softmax(self.K_linear(x)))
        output = self.O_Linear(self.V_Linear(attention).permute(0, 2, 1, 3).contiguous().view(B, N, C))
        return output.transpose(2, 1)


class Generator(nn.Module):

    """
        latent_dim: 2560
    """

    def __init__(self, latent_dim=2560):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, self.latent_dim)

    def forward(self, latent_feature):
        x = F.leaky_relu(self.fc1(latent_feature), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x_256 = F.leaky_relu(self.fc4(x), negative_slope=0.2)
        x = torch.cat([x, x_256], dim=1)
        x_512 = F.leaky_relu(self.fc5(x), negative_slope=0.2)
        x = torch.cat([x, x_512], dim=1)
        x = F.leaky_relu(self.fc6(x), negative_slope=0.2)
        generate_feature = F.leaky_relu(self.fc7(x), negative_slope=0.2)
        return generate_feature


class Discriminator(nn.Module):
    def __init__(self, latent_dim=2560, regular_dim=512):
        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.regular_dim = 512
        self.scales = self.latent_dim // self.regular_dim

        self.conv1 = nn.Conv1d(512, 512, 1)
        self.conv2 = nn.Conv1d(512, 128, 1)
        self.conv3 = nn.Conv1d(128, 64, 1)
        self.EA = Multi_head_ExternalAttention(64, 64, self.scales, 4)
        self.conv4 = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        B, _ = x.shape
        # print(x.shape)
        # print(self.regular_dim)
        # print(self.scales)
        input = x.reshape(B, self.regular_dim, self.scales)
        x = F.leaky_relu(self.conv1(input), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = self.EA(x)
        x = self.conv4(x)
        output = torch.mean(x, dim=2)
        return output


if __name__ == '__main__':
    sim_data = Variable(torch.rand(4, 2560))
    sim_data = sim_data.cuda()
    extract = Discriminator()
    extract = extract.cuda()
    output = extract(sim_data)
    print('input', sim_data.size())
    print('Output:', output.size())
