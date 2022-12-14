# -*- coding: utf-8 -*-
"""DesignGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KQlvsqswUSpb-t_nR3fmo-iLxFaKpDEi
"""

# Commented out IPython magic to ensure Python compatibility.
### import necessary library
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from dataloader import data_loader
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import pickle
import model
import util
import os

# define loss function
def mse_loss(score,target=1):  
    dtype = type(score)  
    if target == 1:
        label = torch.ones(score.size(),requires_grad=False).to(device)
    elif target == 0:
        label = torch.zeros(score.size(),requires_grad=False).to(device)
    criterion = nn.MSELoss()
    loss = criterion(score, label)
    return loss

def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

# define DesignGAN class
class DesignGAN():
    def __init__(self,batch_size, num_epoch,load_weight=False):
        self.weight_dir = './output/weights'
        self.result_dir = './output/images'
        self.gen_image_path = './output/gen_images'
        self.img_size = 256
        self.batch_size = batch_size
        self.start_epoch = 0
        self.num_epoch = num_epoch
        self.save_interval = 50
        self.Rms_lr=2e-4
        self.weight_decay = 1e-5   # first = 6e-8
        self.Adam_lr=0.0002
        self.beta_1=0.5
        self.beta_2=0.999
        #self.lambda_kl=0.01
        self.lambda_img=3   #100
        self.criterionVGG = VGGLoss()
        self.z_dim=512
        self.load_weight = load_weight
        self.test_size = 1
        self.test_img_num = 1
        # Data type(Can use GPU or CPU)
        self.dtype = torch.cuda.FloatTensor
        if torch.cuda.is_available() != True:
            self.dtype = torch.FloatTensor
        # load data for training
        self.dloader = data_loader(batch_size=self.batch_size,shuffle=True)
        # load data for test
        self.t_dloader = data_loader(batch_size=self.test_size, shuffle=True)        
        # load model
        #self.M = model.Mapping(init_weights=True).type(self.dtype)
        #self.E = model.Extractor(z_dim=self.z_dim).to(device)
        self.G = model.Generator(init_weights=True).to(device)
        self.D = model.Discriminator(init_weights=True).to(device)
        # define optimizer
        self.optim_D = optim.Adam(self.D.parameters(),lr=self.Adam_lr,betas=(self.beta_1, self.beta_2))
        self.optim_G = optim.Adam(self.G.parameters(),lr=self.Adam_lr,betas=(self.beta_1, self.beta_2))
        #self.optim_E = optim.RMSprop(self.E.parameters(),lr=self.Rms_lr,weight_decay=self.weight_decay)
        #self.optim_E = optim.Adam(self.E.parameters(),lr=self.Adam_lr,betas=(self.beta_1, self.beta_2))

    # print model architecture
    def show_model(self):
        print('======================== Discriminator ========================')
        print(self.D)
        print('===========================================================\n\n')
        print('========================== Generator ==========================')
        print(self.G)
        print('===========================================================\n\n')
        # print('=========================== Encoder ===========================')
        # print(self.E)
        # print('===========================================================\n\n')

    def set_train_phase(self):
        self.D.train()
        self.G.train()
        #self.E.train()

    # load pre_trained weight
    def load_pretrained(self):
        self.D.load_state_dict(torch.load(os.path.join(self.weight_dir, 'D.pkl')))
        self.G.load_state_dict(torch.load(os.path.join(self.weight_dir, 'G.pkl')))
        #self.E.load_state_dict(torch.load(os.path.join(self.weight_dir, 'E.pkl')))
        
        # log_file = open('log.txt', 'r')
        # line = log_file.readline()
        # self.start_epoch = int(line)

    # save weight
    def save_weight(self, epoch=None):
        if epoch is None:
            d_name = 'D.pkl'
            g_name = 'G.pkl'
            #e_name = 'E.pkl'
        else:
            d_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D.pkl')
            g_name = '{epochs}-{name}'.format(epochs=str(epoch), name='G.pkl')
            #e_name = '{epochs}-{name}'.format(epochs=str(epoch), name='E.pkl')
            
        torch.save(self.D.state_dict(), os.path.join(self.weight_dir, d_name))
        torch.save(self.G.state_dict(), os.path.join(self.weight_dir, g_name))
        #torch.save(self.E.state_dict(), os.path.join(self.weight_dir, e_name))

    # set all optimizers' grad to zero
    def all_zero_grad(self):
        self.optim_D.zero_grad()
        self.optim_G.zero_grad()
        #self.optim_E.zero_grad()

    # train G, D
    def train(self):
        if self.load_weight is True:
            self.load_pretrained()

        self.set_train_phase()
        self.show_model()

        # execute training
        for epoch in range(self.start_epoch, self.num_epoch):
            for batch_idx, dset in enumerate(self.dloader):                
                # train D
                # encoded latent vector
                batch_size=self.batch_size
                # mu, log_var = self.E(dset.type(self.dtype))
                # std = torch.exp(log_var / 2)
                #random_z = torch.randn(batch_size, 512).type(self.dtype)
                #encoded_z = (random_z * std) + mu

                # run mapping
                #style_w = self.M(encoded_z)

                # real image
                real_img1 = dset.type(self.dtype)

                # generate fake image
                noise1 = torch.randn(batch_size,512).type(self.dtype)
                fake_img1 = self.G(real_img1, noise1)                

                # get score
                real_score1 = self.D(real_img1)
                fake_score1 = self.D(fake_img1.detach())

                # define the loss function of the discriminator 
                D_loss = mse_loss(real_score1, 1) + mse_loss(fake_score1, 0)

                # Update D
                self.all_zero_grad()
                D_loss.backward()
                self.optim_D.step()

                # Train G
                # encoded latent vector
                # real_img1 = dset.type(self.dtype)                
                # mu1, log_var1 = self.E(real_img1)
                # std = torch.exp(log_var1 / 2)
                # random_z1 = torch.randn(batch_size, 512).type(self.dtype)
                # encoded_z1 = (random_z1 * std) + mu

                # run mapping
                #style_w1 = self.M(encoded_z1)     

                # real image
                real_img2 = dset.type(self.dtype)           

                # generate fake image
                noise2 = torch.randn(batch_size,512).type(self.dtype)             
                fake_img2 = self.G(real_img2, noise2)
                
                # get score
                fake_score2 = self.D(fake_img2)

                # fool the discriminator
                G_GAN_loss = mse_loss(fake_score2, 1)

                # Reconstruction of ground truth image (|G(noise, z) - real|)
                recon_loss = L1_loss(fake_img1, real_img1)

                # VGGLoss of ground truth image (|G(noise, z) - real|)    

                vgg_loss = self.lambda_img*self.criterionVGG(fake_img2, real_img2)

                # define EG_loss
                G_loss = G_GAN_loss + vgg_loss + recon_loss
                #EG_loss = G_GAN_loss + KL_div + recon_loss
                self.all_zero_grad()
                G_loss.backward()
                #self.optim_E.step()
                self.optim_G.step()

                if batch_idx % 40 == 0:
                    print('[Epoch : %d / Iters : %d] => D_loss : %f / G_GAN_loss : %f / vgg_loss : %f / recon_loss : %f'\
#                             %(epoch, batch_idx, D_loss.data, G_GAN_loss.data, vgg_loss.data, recon_loss.data))

                    # Save intermediate result image
                    os.makedirs(self.result_dir,exist_ok=True)
                    result_img = util.make_img(self.t_dloader, self.G)
                    img_name = '{epoch}_{iters}.png'.format(epoch=epoch,iters=batch_idx) 
                    img_path = os.path.join(self.result_dir, img_name)

                    torchvision.utils.save_image(result_img, img_path, nrow=self.test_img_num+1)

            # Save intermediate weight
            if os.path.exists(self.weight_dir) is False:
                os.makedirs(self.weight_dir)

            if epoch % 5 == 0: 
                self.save_weight(epoch=epoch)

    def test(self, path, num):               
        #weight_path = './output/gen_weights'
        weight_path = path
        Gen = self.G
        Gen.load_state_dict(torch.load(os.path.join(weight_path, '435-G.pkl')))       
        Gen.eval()
        gen_num = num
        n = 0
        for n in range(gen_num):
            # dloader = self.test_dloader
            # # load real image
            # dloader = iter(dloader)
            # real_image = next(dloader)
            # real_image = real_image.to(device)

            fake_image = result_img = util.make_img(self.t_dloader, self.E, Gen)

            # eta = torch.FloatTensor(1,1,1,1).uniform_(0,1)
            # eta = eta.expand(1, real_image.size(1), real_image.size(2), real_image.size(3))
            # eta = eta.to(device)
            # gen_image = eta * real_image + ((1 - eta) * fake_image)
            img_name = '{}.png'.format(n+1)
            img_path = os.path.join('./output/gen_images', img_name)
            torchvision.utils.save_image(fake_image, img_path, nrow=self.test_img_num+1)   
            n += 1 

if __name__ == '__main__':
    gan = DesignGAN(batch_size=16,num_epoch=401,load_weight=False)
    gan.train()
    #gan.test(path='./output/gen_weights', num=3000)