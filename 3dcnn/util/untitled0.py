#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:29:39 2021

@author: zcj
"""

import numpy as np 
a = np.ones([8,1])
theta_train = [0,0,0,0,-9,-4,-5,-7]
phi_train = [0,0,0,0,-5,-7,-4,-2]

a = np.expand_dims(a,axis=[2,3])

for i in range(8):
    print('before',a)
    a[i,:,0,:] = theta_train[i]
    a[i,:,:,0] = phi_train[i]
    print('after',a)
    

a = np.ones([8,1,180,180,360,1])



