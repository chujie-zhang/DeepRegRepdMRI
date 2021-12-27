#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:34:01 2021

@author: zcj
"""
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '/gpu:0'

from os.path import join 
from os.path import expanduser
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti,save_nifti
from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
import dipy
import time
import matplotlib.pyplot as plt
import math
import time
import copy
import re
import numba as nb
from numba import jit, cuda

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def find_origin_point(x,y,z):
    return x//2,y//2,z//2


def Cartesian_polar(x,y,z):
    radius = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/(radius))
    #phi = np.arctan(abs(y)/abs(x))
    phi = np.arccos(abs(x) / np.sqrt(x * x + y * y))
    theta = math.degrees(theta)
    phi = math.degrees(phi)
    
    if x>0 and y>0:
        phi=phi
    elif x<0 and y>0:
        
        phi = 180-phi
    elif x<0 and y<0:
        phi = 180+phi
    elif x>0 and y<0:
        phi = 360-phi
    
    return radius,theta,phi

#@nb.jit(nopython=True)
def polar_Cartesian(radius,theta,phi):
    theta,phi=math.radians(theta),math.radians(phi)
    x=radius*np.sin(theta)*np.cos(phi)
    y=radius*np.sin(theta)*np.sin(phi)
    z=radius*np.cos(theta)
    return x,y,z


def nn_3d(origin_x,origin_y,origin_z,volume,volume_number,is_polar_to_cartesian):
    x = math.floor(origin_x)
    y = math.floor(origin_y)
    z = math.floor(origin_z)
    
    cgamma = origin_x - x
    calpha = origin_y - y
    cbeta = origin_z - z
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    if(gamma < cgamma):
        x += 1
    if(alpha < calpha):
        y += 1
    if(beta < cbeta):
        z += 1
    
    if is_polar_to_cartesian:
        new_x = x+volume.shape[0]//2
        new_y = y+volume.shape[1]//2
        new_z = z+volume.shape[2]//2
        if ((0<= new_x < volume.shape[0]) and (0<= new_y < volume.shape[1]) and (0<= new_z < volume.shape[2])):
            return True,volume[new_x,new_y,new_z,volume_number]
        else:
            return False,0
    else:
        
        if ((0<= x < volume.shape[0]) and (0<= y < volume.shape[1]) and (0<= z < volume.shape[2])):
            return True,volume[x,y,z,volume_number]
        else:
            return False,0

#@nb.jit(nopython=True)
def trilinear_3d(origin_x,origin_y,origin_z,volume,volume_number,is_polar_to_cartesian):
    if is_polar_to_cartesian:
        x = math.floor(origin_x) + volume.shape[0]//2
        y = math.floor(origin_y) + volume.shape[1]//2
        z = math.floor(origin_z) + volume.shape[2]//2
        cgamma = origin_x + volume.shape[0]//2- x
        calpha = origin_y + volume.shape[1]//2- y
        cbeta = origin_z + volume.shape[2]//2- z
    else:
        x = math.floor(origin_x)
        y = math.floor(origin_y) 
        z = math.floor(origin_z)   
        cgamma = origin_x - x
        calpha = origin_y - y
        cbeta = origin_z - z
        
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    result = 0
    inside = 0
    ns = volume.shape[0]
    nr = volume.shape[1]
    nc = volume.shape[2]
    
    if not (-1 < x < ns and -1 < y < nr and -1 < z < nc):
        return False,0
    
    
    if (y >= 0) and (z >= 0) and (x >= 0):
        result = alpha * beta * gamma * volume[x, y, z,volume_number]
        inside += 1
    else:
        result = 0
    # ---top-right
    z += 1
    if (y >= 0) and (z < nc) and (x >= 0):
        result += alpha * cbeta * gamma * volume[x, y, z,volume_number]
        inside += 1
    # ---bottom-right
    y += 1
    if (y < nr) and (z < nc) and (x >= 0):
        result += calpha * cbeta * gamma * volume[x, y, z,volume_number]
        inside += 1
    # ---bottom-left
    z -= 1
    if (y < nr) and (z >= 0) and (x >= 0):
        result += calpha * beta * gamma * volume[x, y, z,volume_number]
        inside += 1
    x += 1
    if(x < ns):
        y -= 1
        if (y >= 0) and (z >= 0):
            result += alpha * beta * cgamma * volume[x, y, z,volume_number]
            inside += 1
        z += 1
        if (y >= 0) and (z < nc):
            result += alpha * cbeta * cgamma * volume[x, y, z,volume_number]
            inside += 1
        # ---bottom-right
        y += 1
        if (y < nr) and (z < nc):
            result += calpha * cbeta * cgamma * volume[x, y, z,volume_number]
            inside += 1
        # ---bottom-left
        z -= 1
        if (y < nr) and (z >= 0):
            result+= calpha * beta * cgamma * volume[x, y, z,volume_number]
            inside += 1
    
        
        
    if inside == 8:
        return True,result
    else:
        return False,0
    
#@nb.jit(nopython=True)
def transformAllVolumesToPolar(data):
    new_image=np.zeros([180,180,360,data.shape[3]])
    radius_array=[]
    theta_array=[]
    phi_array=[]
    #centre_x,centre_y,centre_z = find_origin_point(data.shape[0],data.shape[1],data.shape[2])
    #centre_r,centre_theta,centre_phi = find_origin_point(new_image.shape[0],new_image.shape[1],new_image.shape[2])
    
    for v in range(new_image.shape[3]):
        for r in range(new_image.shape[0]):
            for theta in range(new_image.shape[1]):
                for phi in range(new_image.shape[2]):
                    i,j,k = polar_Cartesian(r,theta,phi)
                    
                    #flag,result = nn_3d(i,j,k,data,v,True)
                    flag,result = trilinear_3d(i,j,k,data,v,True)
                    
                    if flag:
                        new_image[r,theta,phi,v] = result
                        

                    radius_array.append(r)
                    theta_array.append(theta)
                    phi_array.append(phi)

    return new_image,radius_array,theta_array,phi_array


def transformAllVolumesToCartesian(data):
    new_image=np.zeros([144,168,110,data.shape[3]])
    x_array=[]
    y_array=[]
    z_array=[]
    centre_x,centre_y,centre_z = find_origin_point(data.shape[0],data.shape[1],data.shape[2])
    centre_r,centre_theta,centre_phi = find_origin_point(new_image.shape[0],new_image.shape[1],new_image.shape[2])
    
    for v in range(new_image.shape[3]):
        for x in range(new_image.shape[0]):
            for y in range(new_image.shape[1]):
                for z in range(new_image.shape[2]):
                    if x-new_image.shape[0]//2==0 and y-new_image.shape[1]//2==0:
                        continue
                    r,theta,phi = Cartesian_polar(x-new_image.shape[0]//2,y-new_image.shape[1]//2,z-new_image.shape[2]//2) 
                    #flag,result = nn_3d(r,theta,phi,data,v,False)
                    flag,result = trilinear_3d(r,theta,phi,data,v,False)
                    if flag:
                        new_image[x,y,z,v]=result

                    x_array.append(r)
                    y_array.append(theta)
                    z_array.append(phi)

    return new_image,x_array,y_array,z_array

def plot(data,x,y):
    plt.figure('Showing the datasets')
    plt.subplot(1, 2, 1).set_axis_off()
    plt.imshow(data[:, :, 40, 10].T, cmap='gray', origin='lower')
    plt.subplot(1, 2, 2).set_axis_off()
    plt.imshow(data[:, :, 50, 10].T, cmap='gray', origin='lower')
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()
    #plt.savefig('data.png', bbox_inches='tight')


def generate_data():
    home = expanduser('~')
    dname = join(home, 'HCP_retest', '103818_3T_Diffusion_unproc','103818','unprocessed','3T','Diffusion')
    fdwi1 = join(dname, 'eddy_unwarped_images.nii.gz')
    fdwi2 = join(dname, 'my_hifi_images_even.nii.gz')
    fbval = join(dname, '103818_3T_DWI_dir95_RL.bval')
    fbvec = join(dname, '103818_3T_DWI_dir95_RL.bvecs')
    
    static_data, static_affine, static_img = load_nifti(fdwi1, return_img=True)
    moving_data, moving_affine, moving_img = load_nifti(fdwi2, return_img=True)
    
    moving_data_inPolar,radius_array,theta_array,phi_array = transformAllVolumesToPolar(moving_data)
    
    
    with open(fbval, "r") as f:  # open file
        data=f.readlines() 
        origin=[]
        temp = re.findall(r"-?\d+\.?\d*",str(data))
    origin=np.array(list(map(float, temp)))
    index_id = []
    for i in range(origin.shape[0]):
        if origin[i]<=30:
            index_id.append(i)

    for j in range(origin.shape[0]):
        if j in index_id:
            save_nifti("../data/case"+str(j), moving_data_inPolar[:,:,:,j].reshape([moving_data.shape[0],moving_data.shape[1],moving_data.shape[2],1]), moving_affine)
    

start = time.time()
generate_data()
#cuda.synchronize()
end = time.time()
print("The total time is: ",(end-start)/60,"min")
    

