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
        self.Q_linear = nn.Linear(self.channels + 3, self.channels)
        self.K_linear = nn.Linear(self.channels // self.heads, self.external_memory)
        self.softmax = nn.Softmax(dim=2)
        self.LNorm = nn.LayerNorm([self.heads, self.point_num, self.external_memory])
        self.V_Linear = nn.Linear(self.external_memory, self.channels // self.heads)
        self.O_Linear = nn.Linear(self.channels, self.channels)

    def forward(self, x, xyz):
        B, N, C = x.transpose(2, 1).shape
        x = torch.cat([x.transpose(2, 1), xyz], dim=2)
        x = self.Q_linear(x).contiguous().view(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attention = self.LNorm(self.softmax(self.K_linear(x)))
        output = self.O_Linear(self.V_Linear(attention).permute(0, 2, 1, 3).contiguous().view(B, N, C))
        return output.transpose(2, 1)


class Fold(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class Generator(nn.Module):

    """
        num_dense:  16384
        latent_dim: 3840
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=2560, grid_size=2):
        super(Generator, self).__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)
        '''
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        '''
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(4096, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            # nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            # nn.BatchNorm1d(512),
            nn.Conv1d(self.latent_dim + 3 + 2, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 3, 1)
        )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)


        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, latent_feature):
        B, _ = latent_feature.shape

        # decoder
        coarse = self.mlp(latent_feature).reshape(-1, self.num_coarse, 3) # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)  # (B, 3, num_fine)

        rebuild_feature = torch.cat([
            latent_feature.unsqueeze(-2).expand(-1, self.num_coarse, -1),
            coarse], dim=-1)  # B num_coarse latent_dim + 3
        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1)) # BM C

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)

        feature_global = latent_feature.unsqueeze(2).expand(-1, -1, self.num_dense) # (B, 3840, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 3840+2+3, num_fine)

        fine = self.final_conv(feat) + point_feat  # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), point_feat.transpose(1, 2).contiguous()


class Discriminator(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=2560, grid_size=4, point_scales=2048):
        super(Discriminator, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.point_scales = point_scales

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.EA = Multi_head_ExternalAttention(64, 64, self.point_scales, 4)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.CEA1 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.CEA2 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.conv4 = nn.Conv1d(256, 256, 1)
        '''
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        '''

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)
        '''
        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(16)
        '''

    def forward(self, x):
        B, N, _ = x.shape
        self.xyz = x
        # x = F.relu(self.bn1(self.conv1(x.transpose(2, 1))))
        x = F.relu(self.conv1(x.transpose(2, 1)))
        x = self.EA(x, self.xyz)
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.bn3(self.conv3(x)))
        temp00_x_128 = self.CEA1(x, self.xyz)
        temp01_x_128 = self.CEA2(temp00_x_128, self.xyz)
        x = torch.cat((temp00_x_128, temp01_x_128), dim=1)
        # x_256 = F.relu(self.bn4(self.conv4(x)))
        x_256 = F.relu(self.conv4(x))
        low_feature_global = torch.max(x_256, dim=2, keepdim=False)[0]  # (B, 256)
        '''
        output = F.relu(self.bn_1(self.fc1(low_feature_global)))
        output = F.relu(self.bn_2(self.fc2(output)))
        output = F.relu(self.bn_3(self.fc3(output)))
        '''
        output = F.relu(self.fc1(low_feature_global))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output


if __name__ == '__main__':
    sim_data = Variable(torch.rand(4, 3, 2048))
    sim_data = sim_data.cuda()
    sim_data = torch.transpose(sim_data, 1, 2)
    extract = Discriminator(point_scales=2048)
    extract = extract.cuda()
    output = extract(sim_data)
    print('input', sim_data.size())
    print('Output:', output.size())
