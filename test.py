#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:28:01 2021

@author: zcj
"""

import time
import util.helpers as helper
import util.networks as network
import util.data_process as ProcessData
import numpy as np
import torch
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--device', default='cuda:1', help='Device used for test')
parser.add_argument('--test_path', default='./data/valid', help='The test path of test set')
parser.add_argument('--checkpoints', default='./network/pretrain.pt', help='The training path of training set')
parser.add_argument('--save_path', default='./inference/', help='The test path of data set')
args = parser.parse_args()

test_path = args.test_path
device = torch.device(args.device)
pretrained_model = args.checkpoints
save_path = args.save_path


reader_moving_image, reader_fixed_image,_,_ = helper.get_data_readers(
    '%s/moving_images'%test_path,
    '%s/fixed_images'%test_path)


model = torch.load(pretrained_model)
model.eval()
with torch.no_grad():
    #for i in range(reader_moving_image.num_data):
    for i in range(1):

        input_moving = torch.from_numpy(reader_moving_image.get_data([i])).to(device,torch.float)
        input_fixed = torch.from_numpy(reader_fixed_image.get_data([i])).to(device,torch.float)
        
        #get feature maps
        predict_moving = model(input_moving)
        predict_fixed = model(input_fixed)
        
        #helper.write_images(predict_moving, save_path, 'predict_feature_moving%s'%i)
        #helper.write_images(predict_fixed, save_path, 'predict_feature_fixed%s'%i) 
        
        predict_theta,predict_phi = ProcessData.get_theta_phi(predict_moving, predict_fixed)    
        ProcessData.write_image_in_cartesian(input_moving, save_path, 'predict_moving%s.nii.gz'%i,predict_theta,predict_phi,'moving')
        ProcessData.write_image_in_cartesian(input_fixed, save_path, 'predict_fixed%s.nii.gz'%i,predict_theta,predict_phi,'fixed')
        



