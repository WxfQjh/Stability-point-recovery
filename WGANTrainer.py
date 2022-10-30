import argparse
from ast import arg
from curses import init_pair
import os
import numpy as np
import math
import sys
import utils

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
# from models import WGAN_EA
from models import WGAN_GFV
from models import Completion_EA as premodel
from dataset import ShapeNetGFV
# import LoadNpy
import logging
from tqdm import tqdm
import time
from visualization import plot_pcd_one_view
from metrics.metric import l1_cd
os.makedirs("images", exist_ok=True)


def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join("GanModel/logs",
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=2560,
                    help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int,
                    default=10, help="interval betwen image samples")
parser.add_argument('--data', metavar='DIR', default='/home/qjh/PCN-PyTorch/GFV_T/',
                    help='Path to Complete Point Cloud Data Set')
parser.add_argument('--category', type=str, default='cabinet', help='Category of global feature')
parser.add_argument('--split_value', default=0.9, help='Ratio of train and test data split')
parser.add_argument('--ckpt_path', type=str, default='GanModel/ckpt', help='The path of pretrained model')
parser.add_argument('--save_img', type=str, default='GanModel/ckpt/imgs_cabinet', help='The path of pretrained model')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
parser.add_argument('--pretrained', default='/home/qjh/PCN-PyTorch/log/PCWGAN_T_16384/all/checkpoints/model_best.pth.tar',
                    help='Use Pretrained Model for testing or resuming training')  # TODO

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10

# load decoder

network_data = torch.load(opt.pretrained)
model_decoder = premodel.PreModel()
model_decoder.Decoder.load_state_dict(network_data['state_dict_decoder'])
model_decoder.cuda()

# Initialize generator and discriminator
generator = WGAN_GFV.Generator()
discriminator = WGAN_GFV.Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(),
#              transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

train_dataset = ShapeNetGFV(opt.data, 'train', opt.category)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                           num_workers=opt.num_workers,
                                           shuffle=True,
                                           pin_memory=True)


# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
schedulerD = torch.optim.lr_scheduler.StepLR(
    optimizer_D, step_size=40, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(
    optimizer_G, step_size=40, gamma=0.2)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
logger = getLogger()


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------


# training save best
best_generator = 1e8
real_label = 1
fake_label = 0
batch_size = opt.batch_size
label = torch.FloatTensor(batch_size)
criterion = torch.nn.BCEWithLogitsLoss().cpu()
for epoch in range(opt.n_epochs):

    for i, (gfv, complete) in tqdm(enumerate(train_loader, 0),
                                   total=len(train_loader), smoothing=0.9):
        gfv = torch.squeeze(gfv)
        complete = torch.squeeze(gfv)

        b, _ = gfv.shape

        label.resize_([b, 1]).fill_(real_label)

        input = gfv.cuda()
        real = complete.cuda()
        label = label.cuda()

        generator = generator.train()
        discriminator = discriminator.train()
        ############################
        # (2) Update D network
        ###########################
        discriminator.zero_grad()
        # output = discriminator(real)
        real_validity = discriminator(real)

        # errD_real = criterion(output, label)
        # errD_real.backward()

        fake_gfv = generator(input)

        label.data.fill_(fake_label)

        # output = discriminator(fake_gfv.detach())
        fake_validity = discriminator(fake_gfv.detach())

        gradient_penalty = compute_gradient_penalty(
                        discriminator, real.data, fake_gfv.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        # errD_fake = criterion(output, label)

        # errD_fake.backward()
        d_loss.backward()
        # errD = errD_real + errD_fake
        optimizer_D.step()
        ############################
        # (3) Update G network
        ###########################
        # if i % opt.n_critic == 0:
        alpha = 0.001
        beta = 20
        generator.zero_grad()
        label.data.fill_(real_label)

        output = discriminator(input)
        errG_D_x_real = criterion(output, label)
        errG_D_x_real.backward()
        output = discriminator(fake_gfv.detach())
        errG_D_Gx_real = criterion(output, label)
        errG_D_Gx_real.backward()

        l1_loss = torch.sum(torch.abs(real - fake_gfv))
        l1_loss.backward()

        errG = (errG_D_x_real + errG_D_Gx_real) + alpha * l1_loss
        # errG.backward()

        optimizer_G.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
              % (epoch, opt.n_epochs, i, len(train_loader),
                 d_loss.data, errG.data))

        f = open('loss_GanTrainer.txt', 'a')
        f.write('\n'+'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                % (epoch, opt.n_epochs, i, len(train_loader),
                    d_loss.data, errG.data))

        f.close()
        schedulerD.step()
        schedulerG.step()

    if not os.path.exists(opt.ckpt_path):
        os.makedirs(opt.ckpt_path)
    if not os.path.exists(opt.save_img):
        os.makedirs(opt.save_img)
    if epoch % 1 == 0:
        torch.save({'epoch': epoch+1,
                    'state_dict': generator.state_dict()},
                   os.path.join(opt.ckpt_path, 'generator_model_best.pth.tar'))
        torch.save({'epoch': epoch+1,
                    'state_dict': discriminator.state_dict()},
                   os.path.join(opt.ckpt_path,
                                'discriminator_model_best.pth.tar'))
        with torch.no_grad():
            input = input.cuda()
            input_var = Variable(input, requires_grad=True)
            fake_gfv = generator(input)
            _, decoder_fake_out = model_decoder.Decoder(fake_gfv)
            complete = complete.cuda()
            complete = Variable(complete, requires_grad=True)
            _, decoder_complete_out = model_decoder.Decoder(complete)
            plot_pcd_one_view(os.path.join(opt.save_img,
                                           'epoch_{:03d}.png'.format(epoch)),
                              [decoder_fake_out[0].detach().cpu().numpy(),
                               decoder_complete_out[0].detach().cpu().numpy()],
                              ['Dense_fake', 'Dense_real'],
                              xlim=(-0.35, 0.35), ylim=(-0.35, 0.35),
                              zlim=(-0.35, 0.35))


# for epoch in range(opt.n_epochs):
#     for i, (gfv,complete) in enumerate(train_loader):
  
#         # Configure input
#         # real_imgs = Variable(complete.type(Tensor))

#         # ---------------------
#         #  Train Discriminator
#         # ---------------------

#         optimizer_D.zero_grad()

#         # Sample noise as generator input
#         z = Variable(Tensor(np.random.normal(
#             0, 1, (gfv.shape[0], opt.latent_dim))))
        

#         # Generate a batch of images
#         _, fake_imgs = generator(z)

#         fake_fps_idx = utils.farthest_point_sample(fake_imgs, 1024, RAN=False)
#         fake_fps = utils.index_points(fake_imgs, fake_fps_idx)
#         fake_fps =Variable(fake_fps,requires_grad=True)
#         real_fps_idx = utils.farthest_point_sample(complete, 1024, RAN=False)
#         real_fps = utils.index_points(complete, real_fps_idx)
#         real_fps =Variable(real_fps, requires_grad=True).cuda()
#         # Real images
#         # real_validity = discriminator(real_imgs)
#         real_validity = discriminator(real_fps)
#         # Fake images
#         # fake_validity = discriminator(fake_imgs)
#         fake_validity = discriminator(fake_fps)
#         # Gradient penalty
#         gradient_penalty = compute_gradient_penalty(
#             discriminator, real_fps.data, fake_fps.data)
#         # Adversarial loss
#         d_loss = -torch.mean(real_validity) + \
#             torch.mean(fake_validity) + lambda_gp * gradient_penalty

#         d_loss.backward()
#         optimizer_D.step()

#         optimizer_G.zero_grad()
 
#         # Train the generator every n_critic steps
#         if i % opt.n_critic == 0:

#             # -----------------
#             #  Train Generator
#             # -----------------

#             # Generate a batch of images
#             _, fake_imgs = generator(z)
#             # Loss measures generator's ability to fool the discriminator
#             # Train on fake images
#             fake_fps_idx = utils.farthest_point_sample(fake_imgs, 1024, RAN=False)
#             fake_fps = utils.index_points(fake_imgs, fake_fps_idx)
#             fake_fps = Variable(fake_fps, requires_grad=True)
#             # fake_validity = discriminator(fake_imgs)
#             fake_validity = discriminator(fake_fps)
#             g_loss = -torch.mean(fake_validity)

#             g_loss.backward()
#             optimizer_G.step()

#             logger.info(
#                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#                 % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
#             )

            
#             # if g_loss < best_generator:
#             best_generator = g_loss
#             print(opt.save_img)
#             if not os.path.exists(opt.save_img):
#                 os.makedirs(opt.save_img)
#             if batches_done % opt.sample_interval == 0:
#                 # save_image(fake_imgs.data[:25], "images/%d.png" %
#                 #         batches_done, nrow=5, normalize=True)
#                 plot_pcd_one_view(os.path.join(opt.save_img, 'epoch_{:03d}.png'.format(batches_done+epoch)),
#                                     [fake_imgs[0].detach().cpu().numpy(),
#                                     complete[0].detach().cpu().numpy()],
#                                     ['Dense', 'Ground Truth'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
#                 if not os.path.exists(opt.ckpt_path):
#                     os.makedirs(opt.ckpt_path)
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'model': "generatorModel",
#                     'state_dict': generator.state_dict(),
#                 }, os.path.join(opt.ckpt_path, 'generator_model_best.pth.tar'))
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'model': "discriminatorModel",
#                     'state_dict': discriminator.state_dict(),          
#                 }, os.path.join(opt.ckpt_path, 'discriminator_model_best.pth.tar'))

#             batches_done += opt.n_critic
