## Stability point cloud recovery based on WGAN with Transformer

This is the Pytorch implementation for the paper "Stability point cloud recovery based on WGAN with Transformer"

## Getting Started
python version: python-3.7;  cuda version: cuda-11;  

## Datasets
 [PCN's dataset](https://github.com/wentaoyuan/pcn)  
    
## Pretrain Encoder-Decoder module
To pretrain the module: python train.py  

## Pretrain WGAN module
To pretrain the module: python WGANTrainer.py  

## Acknowledgements 
Our implementations use the code from the following repository:  
[PCN](https://github.com/wentaoyuan/pcn)     
[PointNet++](https://github.com/charlesq34/pointnet2)   
[PCT](https://github.com/MenghaoGuo/PCT)   
[External Attention](https://github.com/MenghaoGuo/EANet)
