#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:29:39 2021

@author: zcj
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import network.networks as network
import random
import util.helpers as helper
from argparse import ArgumentParser
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

parser = ArgumentParser()

parser.add_argument('--device', default='cuda:1', help='Device used for train')
parser.add_argument('--epochs', default=50, help='Number of train epochs')
parser.add_argument('--lr', default=5*1e-07, help='Learning rate')
parser.add_argument('--train_batch', default=64, help='The training size of data set')
parser.add_argument('--valid_batch', default=16, help='The validation size of data set')
parser.add_argument('--log', default=2, help='The file name for saving results')
parser.add_argument('--train_path', default='./data/train', help='The training path of training set')
parser.add_argument('--valid_path', default='./data/valid', help='The validation path of validation set')
parser.add_argument('--pos_weight', default=4, help='The weight for positive sample used in Loss')
parser.add_argument('--neg_weight', default=-20, help='The weight for negative sample used in Loss')
args = parser.parse_args()

epoch = args.epochs
learning_rate =args.lr
train_batch_size = args.train_batch
batch_size = args.valid_batch
log_number = args.log
device = torch.device(args.device)
training_path = args.train_path
validation_path = args.valid_path
pos_value = args.pos_weight
neg_value = args.neg_weight

#load data
reader_moving_image, reader_fixed_image,_,_ = helper.get_data_readers(
    '%s/moving_images'%training_path,
    '%s/fixed_images'%training_path)
validation_moving_image, validation_fixed_image, _, _ = helper.get_data_readers(
    '%s/moving_images'%validation_path, 
    '%s/fixed_images'%validation_path)

#load label
training_label,test_label = helper.get_label(pos_value,neg_value,reader_moving_image.num_data,validation_moving_image.num_data)

#load theta and phi pf augmentation data generated randomly
theta_train,phi_train,theta_valid,phi_valid = helper.get_theta_phi("%s/training_data_augmentation_parameters.csv"%training_path,"%s/valid_data_augmentation_parameters.csv"%validation_path,reader_moving_image.num_data,validation_moving_image.num_data)


'''
loss_result_array contains 10 rows
1: total training loss
2: total validation loss

3: mse loss of training for positive sample
4: mse loss of training for negative sample
5: augmentation loss of training for positive sample
6: augmentation loss of training for negative sample

7: mse loss of validation for positive sample
8: mse loss of validation for negative sample
9: augmentation loss of validation for positive sample
10: augmentation loss of validation for negative sample
'''
loss_result_array = np.zeros([10,epoch])

model = network.CNNNet()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for step in tqdm(range(epoch)):

    #-------------------------------------------training------------------------------------------------------#
    input_training_label = torch.from_numpy(training_label).to(device,torch.float)
    input_valid_label = torch.from_numpy(test_label).to(device,torch.float)
  
    mse_loss = nn.MSELoss()
    mse_loss_train_epock = 0
    pos_loss_train = 0
    neg_loss_train = 0
    pos_new_loss_train = 0
    neg_new_loss_train = 0
    
    model.train()
    for i in range(reader_moving_image.num_data):

        input_moving = torch.from_numpy(reader_moving_image.get_data([i])).to(device,torch.float)
        input_fixed = torch.from_numpy(reader_fixed_image.get_data([i])).to(device,torch.float)
        #get feature maps
        moving_training = model(input_moving)
        fixed_training = model(input_fixed)

        optimizer.zero_grad()

        #compute loss
        #mse loss
        mse_loss_train = mse_loss(moving_training,fixed_training) * input_training_label[i]
        #augmentation data loss
        roll_moving=torch.roll(moving_training,shifts=[int(theta_train[i]),int(phi_train[i])],dims=[3,4])
        new_loss_train = mse_loss(roll_moving,fixed_training)
        

        if mse_loss_train < -2:
            train_loss = -2 + new_loss_train
        else:  
            train_loss = mse_loss_train + new_loss_train

        
        #train_loss = mse_loss_train + new_loss_train
        train_loss.backward()
        
        #torch.nn.utils.clip_grad_norm(model.parameters(), 10)
        optimizer.step()
        torch.cuda.empty_cache()
        
        # save loss
        mse_loss_train_epock =mse_loss_train_epock + mse_loss_train + new_loss_train
        print("mse_loss%d: %s new_loss: %s"%(i,mse_loss_train.item(),new_loss_train.item()))
        if mse_loss_train>0:
            pos_loss_train += mse_loss_train
        else:
            neg_loss_train += mse_loss_train
        
        if i<reader_moving_image.num_data/2:
            pos_new_loss_train += new_loss_train
        else:
            neg_new_loss_train += new_loss_train
        #if mse_loss_train<0 and len(neg_los)<=step:
        #    neg_los.append(mse_loss_train)

    loss_result_array[0,step] = mse_loss_train_epock/train_batch_size
    loss_result_array[2,step] = pos_loss_train/train_batch_size*2
    loss_result_array[3,step] = neg_loss_train/train_batch_size*2
    loss_result_array[4,step] = pos_new_loss_train/train_batch_size*2
    loss_result_array[5,step] = neg_new_loss_train/train_batch_size*2


    # -------------------------------------------validation------------------------------------------------------#
    mse_loss_valid_epock = 0
    pos_loss_valid = 0
    neg_loss_valid = 0
    pos_new_loss_valid = 0
    neg_new_loss_valid = 0
    model.eval()
    with torch.no_grad():
        for i in range(validation_moving_image.num_data):
            input_moving = torch.from_numpy(validation_moving_image.get_data([i])).to(device,torch.float)
            input_fixed = torch.from_numpy(validation_fixed_image.get_data([i])).to(device,torch.float)
            
        
            #get feature maps
            moving_valid = model(input_moving)
            fixed_valid = model(input_fixed)

            #compute loss
            mse_loss_valid = mse_loss(moving_valid,fixed_valid) * input_valid_label[i]
            roll_moving=torch.roll(moving_valid,shifts=[int(theta_valid[i]),int(phi_valid[i])],dims=[3,4])
            new_loss_valid = mse_loss(roll_moving,fixed_valid)
            valid_loss = mse_loss_valid + new_loss_valid

        
            # save loss
            mse_loss_valid_epock =mse_loss_valid_epock + mse_loss_valid + new_loss_valid
            print("mse_loss%d: %s new_loss: %s"%(i,mse_loss_valid.item(),new_loss_valid.item()))
            if mse_loss_valid>0:
                pos_loss_valid += mse_loss_valid
            else:
                neg_loss_valid += mse_loss_valid
        
            if i<validation_moving_image.num_data/2:
                pos_new_loss_valid += new_loss_valid
            else:
                neg_new_loss_valid += new_loss_valid
        #if mse_loss_train<0 and len(neg_los)<=step:
        #    neg_los.append(mse_loss_train)

        loss_result_array[1,step] = mse_loss_valid_epock/batch_size
        loss_result_array[6,step] = pos_loss_valid/batch_size*2
        loss_result_array[7,step] = neg_loss_valid/batch_size*2
        loss_result_array[8,step] = pos_new_loss_valid/batch_size*2
        loss_result_array[9,step] = neg_new_loss_valid/batch_size*2
    
    print('Step %d: train_loss=%f validation_loss=%f ' %
          (step,
           loss_result_array[0,step],
           loss_result_array[1,step]))

    #save model
    if step in range(0, epoch, 10):  
        save_path = './data/model%s.pt'%log_number
        torch.save(model,save_path)
        print("Model saved in: %s" % save_path)


#torch.save(model,save_path)
#print("Model saved in: %s" % save_path)

# save results
helper.plot_loss(loss_result_array,epoch,log_number)
helper.save_loss(loss_result_array,log_number)


