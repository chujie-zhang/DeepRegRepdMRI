#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:28:01 2021

@author: zcj
"""

import time
import util.helpers as helper
import util.networks as network
import numpy as np
import torch

reader_moving_image, reader_fixed_image,_,_ = helper.get_data_readers(
    './data/test/new_mr_images',
    './data/test/new_us_images')

device = torch.device("cuda:1")
model = torch.load('./data/model19.pt')
model.eval()


#for i in [0,1,4,5]:
total_time = 0
with torch.no_grad():
    for i in range(reader_moving_image.num_data):
        single_start = time.time()
        input_moving = torch.from_numpy(reader_moving_image.get_data([i])).to(device,torch.float)
        input_fixed = torch.from_numpy(reader_fixed_image.get_data([i])).to(device,torch.float)
        
        #get feature maps
        predict_moving = model(input_moving)
        predict_fixed = model(input_fixed)
    
    
        single_end = time.time()
        print('time:', single_end-single_start)
        total_time = total_time + single_end-single_start
 
        
        helper.write_images(predict_moving, './result/test/', 'predict_moving%s'%i)
        helper.write_images(predict_fixed, './result/test/', 'predict_fixed%s'%i)

print('total time:',total_time)  


