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
import util.networks as network
import random
import util.helpers as helper

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
epoch = 50
learning_rate =1e-05
train_batch_size = 64
batch_size = 16
log_number = 21
device = torch.device("cuda:1")

reader_moving_image, reader_fixed_image,_,_ = helper.get_data_readers(
    './data/train/new_mr_images',
    './data/train/new_us_images')
validation_moving_image, validation_fixed_image, _, _ = helper.get_data_readers(
    './data/test/new_mr_images', 
    './data/test/new_us_images')

theta_train,phi_train,theta_valid,phi_valid = helper.get_theta_phi("./data/train/train_result.csv","./data/test/test_result.csv",reader_moving_image.num_data,validation_moving_image.num_data)
training_label,test_label = helper.get_label(1,-3,reader_moving_image.num_data,validation_moving_image.num_data)

'''
theta_train = [0,0,0,0,-9,-4,-5,-7]
phi_train = [0,0,0,0,-5,-7,-4,-2]
theta_valid = [0,0,0,0,-4,-3,-6,-8]
phi_valid = [0,0,0,0,-7,-8,-2,-5]
'''


        

traing_set = {}
traing_set['images'] = reader_moving_image.get_data()
traing_set['labels'] = np.array(theta_train)

num_minibatch = int(reader_moving_image.num_data/train_batch_size)
train_indices = [i for i in range(reader_moving_image.num_data)]
num_minibatch_test = int(validation_moving_image.num_data/batch_size)
test_indices = [i for i in range(validation_moving_image.num_data)]

loss_result_array = np.zeros([10,epoch])

#training_images = torch.utils.data.DataLoader(reader_moving_image.get_data(), batch_size=8, shuffle=False, num_workers=0)
#training_images = {x: torch.utils.data.DataLoader(traing_set[x],batch_size=4,shuffle=False,num_workers=4) for x in ['images', 'labels']}

model = network.CNNNet()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
'''
for step in tqdm(range(epoch)):
    #prepare trainFeed and validationFeed
    print('training step:')
    mse_loss = nn.MSELoss()
    for i, images in enumerate(training_images):
        for iter in range(images.shape[0]):
            moving_feature = model(images)
'''
model.train()
for step in tqdm(range(epoch)):

    if step in range(0, epoch, num_minibatch):
        random.shuffle(train_indices)
    if step in range(0, epoch, num_minibatch_test):
        random.shuffle(test_indices)
        
    case_indices1=[]
    case_indices2=[]
    for i in train_indices:
        #if len(case_indices1)<batch_size:
        #    case_indices1.append(i)
        
        
        if i <=(reader_moving_image.num_data/2)-1:
            if len(case_indices1)<(train_batch_size//2):
                case_indices1.append(i)
        else:
            if len(case_indices2)<(train_batch_size//2):
                case_indices2.append(i)
        
                
    case_indices= case_indices1+case_indices2
    random.shuffle(case_indices)
    case_indices_validation1=[]
    case_indices_validation2=[]
    
    for i in test_indices:
        #if len(case_indices_validation1)<batch_size:
        #    case_indices_validation1.append(i)
        
        if i <=(validation_moving_image.num_data/2)-1:
            if len(case_indices_validation1)<(batch_size//2):
                case_indices_validation1.append(i)
        else:
            if len(case_indices_validation2)<(batch_size//2):
                case_indices_validation2.append(i)
        
                
    case_indices_validation= case_indices_validation1+case_indices_validation2
    random.shuffle(case_indices_validation)
    
    input_training_label = torch.from_numpy(training_label).to(device,torch.float)
    input_valid_label = torch.from_numpy(test_label).to(device,torch.float)
    print("train_indice: %s"%case_indices)
    mse_loss = nn.MSELoss()
    mse_loss_train_epock = 0
    pos_loss_train = 0
    neg_loss_train = 0
    pos_new_loss_train = 0
    neg_new_loss_train = 0
    
    model.train()
    for i in range(reader_moving_image.num_data):
    #for i in case_indices: 

        input_moving = torch.from_numpy(reader_moving_image.get_data([i])).to(device,torch.float)
        input_fixed = torch.from_numpy(reader_fixed_image.get_data([i])).to(device,torch.float)
        
        
        #get feature maps
        moving_training = model(input_moving)
        fixed_training = model(input_fixed)
        optimizer.zero_grad()
        #print("Outside: input size", input_moving.size(),"output_size", moving_training.size())

        #compute loss
        mse_loss_train = mse_loss(moving_training,fixed_training) * input_training_label[i]

        
        roll_moving=torch.roll(moving_training,shifts=[int(theta_train[i]),int(phi_train[i])],dims=[3,4])
        new_loss_train = mse_loss(roll_moving,fixed_training)
        
        
        if mse_loss_train < -0.7:
            train_loss = -0.7 + new_loss_train
        else:  
            train_loss = mse_loss_train + new_loss_train
        
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
    
    
    
    
    
    print("validation_indice: %s"%case_indices_validation)
    mse_loss_valid_epock = 0
    pos_loss_valid = 0
    neg_loss_valid = 0
    pos_new_loss_valid = 0
    neg_new_loss_valid = 0
    model.eval()
    with torch.no_grad():
        for i in range(validation_moving_image.num_data):
        #for i in case_indices_validation: 
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

    if step in range(0, epoch, 100):  
        save_path = './data/model%s.pt'%log_number
        torch.save(model,save_path)
        print("Model saved in: %s" % save_path)






torch.save(model,save_path)
print("Model saved in: %s" % save_path)
helper.plot_loss(loss_result_array,epoch,log_number) 

helper.save_loss(loss_result_array,log_number)


