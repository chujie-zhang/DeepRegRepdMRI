from os.path import join
from os.path import expanduser
import numpy as np
from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.image import load_nifti
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
import pandas as pd
import re
from dipy.io.image import save_nifti
from dipy.align.metrics import SSDMetric, CCMetric, EMMetric
import nibabel as nib
import math
import random


def Cartesian_polar(mean_theta, mean_phi, x, y, z):
    '''
    Cartesian coordinate to spherical polar coordinate
    '''
    radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / (radius))
    # phi = np.arctan(abs(y)/abs(x))
    phi = np.arccos(abs(x) / np.sqrt(x * x + y * y))
    theta = math.degrees(theta) + mean_theta
    phi = math.degrees(phi) + mean_phi

    if x > 0 and y > 0:
        phi = phi
    elif x < 0 and y > 0:

        phi = 180 - phi
    elif x <= 0 and y <= 0:
        phi = 180 + phi
    elif x > 0 and y < 0:
        phi = 360 - phi

    return radius, theta, phi


def polar_Cartesian(radius, theta, phi):
    '''
    spherical polar coordinate to Cartesian coordinate
    '''
    theta, phi = math.radians(theta), math.radians(phi)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z


def nn_3d(origin_x, origin_y, origin_z, volume, volume_number, is_polar_to_cartesian):
    '''
    interpolation method: Nearest Neighbor
    '''
    x = math.floor(origin_x)
    y = math.floor(origin_y)
    z = math.floor(origin_z)

    cgamma = origin_x - x
    calpha = origin_y - y
    cbeta = origin_z - z
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    if (gamma < cgamma):
        x += 1
    if (alpha < calpha):
        y += 1
    if (beta < cbeta):
        z += 1

    if is_polar_to_cartesian:
        new_x = x + volume.shape[0] // 2
        new_y = y + volume.shape[1] // 2
        new_z = z + volume.shape[2] // 2
        if ((0 <= new_x < volume.shape[0]) and (0 <= new_y < volume.shape[1]) and (0 <= new_z < volume.shape[2])):
            return True, volume[new_x, new_y, new_z, volume_number]
        else:
            return False, 0
    else:

        if ((0 <= x < volume.shape[0]) and (0 <= y < volume.shape[1]) and (0 <= z < volume.shape[2])):
            return True, volume[x, y, z, volume_number]
        else:
            return False, 0


def trilinear_3d(origin_x, origin_y, origin_z, volume, volume_number, is_polar_to_cartesian):
    '''
    interpolation method: trilinear
    '''
    if is_polar_to_cartesian:
        x = math.floor(origin_x) + volume.shape[0] // 2
        y = math.floor(origin_y) + volume.shape[1] // 2
        z = math.floor(origin_z) + volume.shape[2] // 2
        cgamma = origin_x + volume.shape[0] // 2 - x
        calpha = origin_y + volume.shape[1] // 2 - y
        cbeta = origin_z + volume.shape[2] // 2 - z
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
        return False, 0

    if (y >= 0) and (z >= 0) and (x >= 0):
        result = alpha * beta * gamma * volume[x, y, z, volume_number]
        inside += 1
    else:
        result = 0
    # ---top-right
    z += 1
    if (y >= 0) and (z < nc) and (x >= 0):
        result += alpha * cbeta * gamma * volume[x, y, z, volume_number]
        inside += 1
    # ---bottom-right
    y += 1
    if (y < nr) and (z < nc) and (x >= 0):
        result += calpha * cbeta * gamma * volume[x, y, z, volume_number]
        inside += 1
    # ---bottom-left
    z -= 1
    if (y < nr) and (z >= 0) and (x >= 0):
        result += calpha * beta * gamma * volume[x, y, z, volume_number]
        inside += 1
    x += 1
    if (x < ns):
        y -= 1
        if (y >= 0) and (z >= 0):
            result += alpha * beta * cgamma * volume[x, y, z, volume_number]
            inside += 1
        z += 1
        if (y >= 0) and (z < nc):
            result += alpha * cbeta * cgamma * volume[x, y, z, volume_number]
            inside += 1
        # ---bottom-right
        y += 1
        if (y < nr) and (z < nc):
            result += calpha * cbeta * cgamma * volume[x, y, z, volume_number]
            inside += 1
        # ---bottom-left
        z -= 1
        if (y < nr) and (z >= 0):
            result += calpha * beta * cgamma * volume[x, y, z, volume_number]
            inside += 1

    if inside == 8:
        return True, result
    else:
        return False, 0


def transformOneVolumesToPolar(data, volume_id):
    '''
    transfrom images from Cartesian coordinate system to spherical polar coordinate system
    '''
    new_image = np.zeros([180, 180, 360, data.shape[3]])
    # new_image=np.zeros([180,180,360])
    for r in range(new_image.shape[0]):
        for theta in range(new_image.shape[1]):
            for phi in range(new_image.shape[2]):
                i, j, k = polar_Cartesian(r, theta, phi)

                # flag,result = nn_3d(i,j,k,data,volume_id,True)
                flag, result = trilinear_3d(i, j, k, data, volume_id, True)

                if flag:
                    # new_image[r,theta,phi,volume_id] = result
                    new_image[r, theta, phi, 0] = result

    return new_image


def transformAllVolumesToCartesian(data, mean_theta, mean_phi):
    '''
    transfrom images from spherical polar coordinate system to Cartesian coordinate system
    '''
    #new_image = np.zeros([144, 168, 110, data.shape[3]])
    new_image = np.zeros([144, 168, 110, 1])
    x_array = []
    y_array = []
    z_array = []
    data = data.reshape([data.shape[0],data.shape[1],data.shape[2],1])
    for v in range(new_image.shape[3]):
        for x in range(new_image.shape[0]):
            for y in range(new_image.shape[1]):
                for z in range(new_image.shape[2]):
                    if x - new_image.shape[0] // 2 == 0 and y - new_image.shape[1] // 2 == 0:
                        continue
                    r, theta, phi = Cartesian_polar(mean_theta, mean_phi, x - new_image.shape[0] // 2,
                                                    y - new_image.shape[1] // 2, z - new_image.shape[2] // 2)
                    # theta += mean_theta
                    # phi += mean_phi
                    flag, result = nn_3d(r, theta, phi, data, v, False)
                    #flag,result = trilinear_3d(r,theta,phi,data,v,False)
                    if flag:
                        new_image[x, y, z, v] = result

                    x_array.append(r)
                    y_array.append(theta)
                    z_array.append(phi)

    return new_image



def compute_euclidean_metric(moving, fixed):
    return np.sqrt(np.sum(np.square(moving - fixed)))

def transform_data(data):
    data = 255 * (data - np.min(data)) / (np.max(data) - np.min(data))
    return data.astype(np.uint8)


def find_parameters(origin_moving, origin_fixed):
    '''
    find shifted parameter
    '''

    # first step: normalizaton
    origin_moving = origin_moving - np.mean(origin_moving)
    origin_fixed = origin_fixed - np.mean(origin_fixed)
    moving = origin_moving / np.sqrt(np.sum(np.square(origin_moving)))
    fixed = origin_fixed / np.sqrt(np.sum(np.square(origin_fixed)))

    moving = moving[10:len(moving) - 10]
    fixed = fixed[10:len(fixed) - 10]

    # second step: find shifted parameter
    result_list = []
    for i in range(10):
        new_moving = np.roll(moving, i * -1)
        difference = np.mean(np.abs(fixed - new_moving))
        result_list.append(difference)
    result_list = np.array(result_list)
    target_parameter = np.argwhere(result_list == np.min(result_list))
    return int(target_parameter)


def match(predict_moving, predict_fixed):
    '''
    compute shifted theta and phi between predicted moving image and predicted fixed image
    '''
    # transform value to 0-255
    predict_moving = transform_data(predict_moving)
    predict_fixed = transform_data(predict_fixed)

    moving_theta_array = np.mean(predict_moving, axis=1)
    fixed_theta_array = np.mean(predict_fixed, axis=1)
    moving_phi_array = np.mean(predict_moving, axis=0)
    fixed_phi_array = np.mean(predict_fixed, axis=0)

    # compute theta
    theta = find_parameters(moving_theta_array, fixed_theta_array)
    # compute phi
    phi = find_parameters(moving_phi_array, fixed_phi_array)
    return theta, phi

def get_theta_phi(predict_moving, predict_fixed):
    predict_moving = predict_moving[0, ...][0,...].to('cpu').numpy()
    predict_fixed = predict_fixed[0, ...][0,...].to('cpu').numpy()
    
    final_theta = 0
    final_phi = 0
    for i in range(predict_moving.shape[0]):
        theta,phi = match(predict_moving[9,:,:],predict_fixed[9,:,:])
        final_theta += theta
        final_phi += phi
    return final_theta/predict_moving.shape[0], final_phi/predict_moving.shape[0]


def write_image_in_cartesian(input_, file_path, file_prefix, predict_theta,predict_phi,flag):
    if file_path is not None:
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        input_ = input_[0, ...][0,...].to('cpu').numpy()
        if flag == 'fixed':
            input_polar = input_
        elif flag== 'moving': 
            input_polar = np.roll(input_,[-1*int(predict_theta),-1*int(predict_phi)],axis=[1,2])
        input_cartesian = transformAllVolumesToCartesian(input_polar,0,0)
        input_cartesian = input_cartesian[:,:,:,0]
        #input_cartesian = nib.Nifti1Image(input_cartesian,affine) 
        #nib.save(input_cartesian,join(file_path+file_prefix))
        save_nifti(file_path+file_prefix,input_cartesian,affine)


    
    
    
    

