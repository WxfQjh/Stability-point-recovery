import torch
import torch.utils.data
import torch.nn.parallel
import time
import models

from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import os
import argparse
import datetime


from torch.autograd.variable import Variable
from utils import AverageMeter, get_n_params

from torch.utils.data.dataloader import DataLoader
# from dataset import ShapeNet
from dataset import ShapenetPartial
from models import Completion_EA as  premodel

np.random.seed(5)
torch.manual_seed(5)




parser = argparse.ArgumentParser(description= 'Point Cloud Training Autoencoder and Shapecompletion Training on Three Datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#arguments for Saving Models
parser.add_argument('--save_path',default='./GFV_T_P', help='Path to Data Set')
parser.add_argument('--save',default= True,help= 'Save Models or not ?')
parser.add_argument('--pretrained', default='/home/qjh/PCN-PyTorch/log/PCWGAN_T_16384/all/checkpoints/model_best.pth.tar',
                    help='Use Pretrained Model for testing or resuming training')  # TODO


# Arguments for Model Settings
parser.add_argument('-me','--model_encoder',default='pcn_transform_premodel',help='Chose Your Encoder Model Here',choices=['encoder_pointnet']) # TODO
parser.add_argument('-md','--model_decoder',default='decoder_sonet',help='Chose Your Decoder Model Here',choices=['decoder_sonet']) # TODO
parser.add_argument('-nt','--net_name',default='auto_encoder',help='Choose The name of your network',choices=['auto_encoder']) #TODO


# Arguments for Data Loader
#  TODO Add Path to Training Data here
parser.add_argument('-d', '--data', metavar='DIR', default='', help='Path to Complete Point Cloud Data Set')
parser.add_argument('-s','--split_value',default = None, help='Ratio of train and test data split')
parser.add_argument('-n', '--dataName', metavar='Data Set Name', default='shapenet')

# Arguments for Torch Data Loader
parser.add_argument('-b','--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('-w','--workers',type=int, default=16, help='Set the number of workers')



# Visualizer Settings
parser.add_argument('--name', type=str, default='GFV',help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_id', type=int, default=2000, help='window id of the web display')
parser.add_argument('--port_id', type=int, default=8099, help='Port id for browser')
parser.add_argument('--print_freq', type=int, default=10, help='Print Frequency')


# Setting for Decoder
#parser.add_argument('--output_pc_num', type=int, default=1280, help='# of output points')
parser.add_argument('--output_fc_pc_num', type=int, default=256, help='# of fc decoder output points')
parser.add_argument('--output_conv_pc_num', type=int, default=4096, help='# of conv decoder output points')
parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

parser.add_argument('--category', type=str, default='all', help='Category of point clouds')

# GPU settings
parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0, 1. -1 is no GPU')



args = parser.parse_args()

args.device = torch.device("cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu") # for selecting device for chamfer loss
torch.cuda.set_device(args.gpu_id)
print('Using A6000 GPU # :', torch.cuda.current_device())


def main(args):

     """------------------------------ Path to save the GFV files-------------------------------------------------- """
     

     if args.save == True:
           save_path = '{}'.format(
               args.model_encoder)
           time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
           save_path = os.path.join(time_stamp, save_path)
           save_path = os.path.join(args.dataName, save_path)
           save_path = os.path.join(args.save_path, save_path)
           print('==> Will save Everything to {}', save_path)

           if not os.path.exists(save_path):
                os.makedirs(save_path)

     """------------------------------------- Data Loader---------------------------------------------------------- """
    #  [train_dataset, valid_dataset] = Datasets.__dict__[args.dataName](input_root=args.data,
    #                                                               target_root=None,
    #                                                               split=args.split_value,
    #                                                               net_name=args.net_name,
    #                                                               input_transforms=None,
    #                                                               target_transforms=None,
    #                                                               co_transforms=None,
    #                                                               give_name =True)


    #  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,
    #                                            num_workers = args.workers,
    #                                            shuffle = True,
    #                                            pin_memory=True)

    #  valid_loader = torch.utils.data.DataLoader(valid_dataset,
    #                                            batch_size=args.batch_size,
    #                                            num_workers=args.workers,
    #                                            shuffle=False,
    #                                            pin_memory=True)

     train_dataset = ShapenetPartial('data/PCN', 'train', args.category)
    #  valid_dataset = ShapeNet('data/PCN', 'valid', args.category)
 
     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    #  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

     """----------------------------------------------Model Settings--------------------------------------------------"""

     print('Model:', args.model_encoder)



    # model.load_state_dict(torch.load(
    #     args.ckpt_path)['state_dict'])
     network_data = torch.load(args.pretrained)
    # model.encoder.load_state_dict(data['state_dict_encoder'])
     model_encoder = premodel.PreModel()
     model_encoder.Encoder.load_state_dict(network_data['state_dict_encoder'])
     model_encoder.to(args.device)
     args = get_n_params(model_encoder)
     print('| Number of Encoder parameters [' + str(args) + ']...')

    
     test_loss = test(train_loader, model_encoder, save_path)

     print('Average Loss :{}'.format(test_loss))


def test(train_loader, model_encoder, save_path):

    model_encoder.eval()     

    print('==> Will save Validation Clean GFV to {}', save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for _, (input, input_name) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        save_path_old = save_path
    
        input_name = input_name[0]          
        save_path = os.path.join(save_path, input_name[:17])
        root_name = os.path.basename(input_name)

        if not os.path.exists(save_path):
             os.makedirs(save_path)

        save_file = os.path.join(save_path, root_name)
     

        with torch.no_grad():

            input = input.cuda()

            input_var = Variable(input,requires_grad = True)

            encoder_out = model_encoder.Encoder(input_var)
   
            np.save(save_file, encoder_out.cpu())

        save_path = save_path_old

    return True



if __name__ =='__main__':
    main(args)
