#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super(ConvLayer, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, self.latent_dim, 1),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape

        feature = self.first_conv(xyz.transpose(2, 1))  # (B,  256, N)
        low_feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1)
        feature = torch.cat([low_feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)
        multi_level_global_feature = torch.cat([torch.squeeze(low_feature_global, dim=2), feature_global], dim=1)
        return multi_level_global_feature


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


class ConvLayer_Transformer(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, point_scales=2048):
        super(ConvLayer_Transformer, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.point_scales = point_scales

        assert self.num_dense % self.grid_size ** 2 == 0

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.EA = Multi_head_ExternalAttention(64, 64, self.point_scales, 4)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.CEA1 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.CEA2 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.conv4 = nn.Conv1d(256, 256, 1)
        self.conv5 = nn.Conv1d(512, 512, 1)
        self.CEA3 = Multi_head_ExternalAttention(512, 512, self.point_scales, 4)
        self.CEA4 = Multi_head_ExternalAttention(512, 512, self.point_scales, 4)
        self.conv6 = nn.Conv1d(1024, 1024, 1)
        self.conv7 = nn.Conv1d(1024, self.latent_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(self.latent_dim)

    def forward(self, x):
        B, N, _ = x.shape
        self.xyz = x
        x = F.leaky_relu(self.bn1(self.conv1(x.transpose(2, 1))), negative_slope=0.2)
        x = self.EA(x, self.xyz)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        temp00_x_128 = self.CEA1(x, self.xyz)
        temp01_x_128 = self.CEA2(temp00_x_128, self.xyz)
        x = torch.cat((temp00_x_128, temp01_x_128), dim=1)
        x_256 = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        low_feature_global = torch.max(x_256, dim=2, keepdim=True)[0]  # (B,  256, 1)
        x = torch.cat([low_feature_global.expand(-1, -1, N), x_256], dim=1)  # (B,  512, N)
        x_512 = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        temp00_x_256 = self.CEA3(x_512, self.xyz)
        temp01_x_256 = self.CEA4(temp00_x_256, self.xyz)
        x = torch.cat((temp00_x_256, temp01_x_256), dim=1)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)
        feature_global = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)
        multi_level_global_feature = torch.cat([torch.squeeze(low_feature_global, dim=2), feature_global], dim=1)
        return multi_level_global_feature


class SuperDecoder(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=2 * (1024 + 256), grid_size=2):
        super(SuperDecoder, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(4096, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + self.grid_size, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, feature_global):
        B, _ = feature_global.shape

        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)  # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, int((self.grid_size ** 2)), -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)  # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 1024+2+3, num_fine)

        fine = self.final_conv(feat) + point_feat  # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()


class AutoEncoder(nn.Module):
    def __init__(self, multi_global_size=2 * (1024 + 256)):
        super(AutoEncoder, self).__init__()
        self.multi_global_size = multi_global_size
        self.Encoder = ConvLayer()
        self.Encoder1 = ConvLayer_Transformer()
        self.FC = nn.Sequential(
            nn.Linear(self.multi_global_size, self.multi_global_size, bias=True),
            nn.BatchNorm1d(self.multi_global_size),
            nn.ReLU(inplace=True)
        )
        # self.fc = nn.Linear(self.multi_global_size, self.multi_global_size, bias=True)
        # self.bn = nn.BatchNorm1d(self.multi_global_size)

    def forward(self, x):
        output = self.Encoder(x)
        output1 = self.Encoder1(x)
        output = torch.cat([output, output1], dim=1)
        # output = F.relu(self.bn(self.fc(output)))
        output = self.FC(output)
        return output


class PreModel(nn.Module):
    def __init__(self):
        super(PreModel, self).__init__()
        self.Encoder = AutoEncoder()
        self.Decoder = SuperDecoder()

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x


if __name__ == "__main__":

    sim_data = Variable(torch.rand(8, 3, 2048))
    sim_data = sim_data.cuda()
    sim_data = torch.transpose(sim_data, 1, 2)
    extract = PreModel()
    extract = extract.cuda()
    co, out = extract(sim_data)
    print('input', sim_data.size())
    print('Output', co.size())
