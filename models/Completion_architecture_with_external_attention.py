#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class Convlayer(nn.Module):
    def __init__(self, point_scales):
        super(Convlayer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool1d(self.point_scales, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(1024)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        x_256 = F.relu(self.bn4(self.conv4(x_128)))
        x_512 = F.relu(self.bn5(self.conv5(x_256)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        x_512 = torch.squeeze(self.maxpool(x_512))
        x_1024 = torch.squeeze(self.maxpool(x_1024))
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


class ConvLayer_Transformer(nn.Module):
    def __init__(self, point_scales):
        super(ConvLayer_Transformer, self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        # self.OA = offset_attention(64)
        self.EA = Multi_head_ExternalAttention(64, 64, self.point_scales, 4)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        # self.CA1 = offset_attention(128)
        # self.CA2 = offset_attention(128)
        self.CEA1 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.CEA2 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.conv4 = nn.Conv1d(256, 256, 1)
        # self.CA3 = offset_attention(256)
        # self.CA4 = offset_attention(256)
        self.CEA3 = Multi_head_ExternalAttention(256, 256, self.point_scales, 4)
        self.CEA4 = Multi_head_ExternalAttention(256, 256, self.point_scales, 4)
        self.conv5 = nn.Conv1d(512, 512, 1)
        self.conv6 = nn.Conv1d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool1d(self.point_scales, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(1024)

    def forward(self, x):
        self.xyz = torch.transpose(x, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.EA(x, self.xyz)
        x = F.relu(self.bn2(self.conv2(x)))
        x_128 = F.relu(self.bn3(self.conv3(x)))
        temp00_x_128 = self.CEA1(x_128, self.xyz)
        temp01_x_128 = self.CEA2(temp00_x_128, self.xyz)
        x = torch.cat((temp00_x_128, temp01_x_128), dim=1)
        x_256 = F.relu(self.bn4(self.conv4(x)))
        temp00_x_256 = self.CEA3(x_256, self.xyz)
        temp01_x_256 = self.CEA4(temp00_x_256, self.xyz)
        x = torch.cat((temp00_x_256, temp01_x_256), dim=1)
        x_512 = F.relu(self.bn5(self.conv5(x)))
        x_1024 = F.relu(self.bn6(self.conv6(x_512)))
        x_128 = torch.squeeze(self.maxpool(x_128))
        x_256 = torch.squeeze(self.maxpool(x_256))
        x_512 = torch.squeeze(self.maxpool(x_512))
        x_1024 = torch.squeeze(self.maxpool(x_1024))
        L = [x_1024, x_512, x_256, x_128]
        x = torch.cat(L, 1)
        return x


class Latentfeature(nn.Module):
    def __init__(self, each_scales_size, point_scales_list):
        super(Latentfeature, self).__init__()
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales=self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(2, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(torch.unsqueeze(self.Convlayers1[i](x[0]), 2))
        for j in range(self.each_scales_size):
            outs.append(torch.unsqueeze(self.Convlayers2[j](x[1]), 2))
        latentfeature = torch.cat(outs, 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature, 1)

        return latentfeature


class Latentfeature_Transformer(nn.Module):
    def __init__(self, each_scales_size, point_scales_list):
        super(Latentfeature_Transformer, self).__init__()
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([ConvLayer_Transformer(
            point_scales=self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([ConvLayer_Transformer(
            point_scales=self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(2, 1, 1)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x):
        outs = []
        for i in range(self.each_scales_size):
            outs.append(torch.unsqueeze(self.Convlayers1[i](x[0]), 2))
        for j in range(self.each_scales_size):
            outs.append(torch.unsqueeze(self.Convlayers2[j](x[1]), 2))
        latentfeature = torch.cat(outs, 2)
        latentfeature = latentfeature.transpose(1, 2)
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
        latentfeature = torch.squeeze(latentfeature, 1)

        return latentfeature


# class offset_attention(nn.Module):
#     def __init__(self, channels):
#         super(offset_attention, self).__init__()
#         self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
#         self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
#         self.q_conv.weight = self.k_conv.weight
#         self.v_conv = nn.Conv1d(channels, channels, 1)
#         self.trans_conv = nn.Conv1d(channels, channels, 1)
#         self.after_norm = nn.BatchNorm1d(channels)
#         self.act = nn.ReLU()
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x = x + xyz
#         x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
#         x_k = self.k_conv(x)  # b, c, n
#         x_v = self.v_conv(x)
#         energy = torch.bmm(x_q, x_k)  # b, n, n
#         attention = self.softmax(energy)
#         attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
#         x_r = torch.bmm(x_v, attention)  # b, c, n
#         x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
#         x = x + x_r
#         return x


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
        x = (torch.transpose(x, 1, 2))
        B, N, C = x.shape
        x = torch.cat([x, xyz], dim=2)
        x = self.Q_linear(x).contiguous().view(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attention = self.LNorm(self.softmax(self.K_linear(x)))
        output = torch.transpose(self.O_Linear(self.V_Linear(attention).permute(0, 2, 1, 3).contiguous().view(B, N, C)), 1, 2)
        return output


class SuperDecoder(nn.Module):

    """
        num_dense:  16384
        latent_dim: 3840
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=3840, grid_size=4):
        super(SuperDecoder, self).__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            # nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            # nn.BatchNorm1d(512),
            nn.Conv1d(self.latent_dim + 3 + 2, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
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

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)

        feature_global = latent_feature.unsqueeze(2).expand(-1, -1, self.num_dense) # (B, 3840, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 3840+2+3, num_fine)

        fine = self.final_conv(feat) + point_feat  # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()


class AutoEncoder(nn.Module):
    def __init__(self, each_scales_size, point_scales_list, output_size):
        super(AutoEncoder, self).__init__()
        self.point_scales_list = point_scales_list
        self.each_scales_size = each_scales_size
        self.mlpEncoder = Latentfeature(self.each_scales_size, self.point_scales_list)
        self.transformerEncoder = Latentfeature_Transformer(self.each_scales_size, self.point_scales_list)
        self.fc = nn.Linear(output_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        mlpEncoderOutput = self.mlpEncoder(x)
        transformerEncoderOutput = self.transformerEncoder(x)
        x = torch.cat([mlpEncoderOutput, transformerEncoderOutput], dim=1)
        output = F.relu(self.bn(self.fc(x)))
        return output


class PreModel(nn.Module):
    def __init__(self, point_scales_list, num_dense=16384, latent_dim=3840, grid_size=4):
        super(PreModel, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.point_scales_list = point_scales_list
        self.Encoder = AutoEncoder(1, self.point_scales_list, self.latent_dim)
        self.Decoder = SuperDecoder(num_dense=16384, latent_dim=3840, grid_size=4)

    def forward(self, x):
        x = self.Encoder(x)
        coarse_output, fine_output = self.Decoder(x)
        return coarse_output, fine_output


if __name__ == '__main__':

    sim_data = Variable(torch.rand(2, 3, 2048))
    sim_data2 = Variable(torch.rand(2, 3, 1024))
    sim_data = sim_data.cuda()
    sim_data2 = sim_data2.cuda()
    extract = PreModel([2048, 1024])
    extract = extract.cuda()
    coarse_output, fine_output = extract([sim_data, sim_data2])
    print('input', sim_data.size())
    print('Output_coarse:', coarse_output.size())
    print('Output_fine', fine_output.size())
#    extract = Convlayer(1024)
#    out = extract(sim_data)
#    print('input', sim_data.size())
#    print('ConvLayer', out.size())
