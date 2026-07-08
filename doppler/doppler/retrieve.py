#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:44:14 2019

@author: km357

"""
import numpy as np
from scipy.ndimage import uniform_filter

def border_particle_vel(DopplerSpecdB, orderedDopplerVel, lowThreshold_dB = -40,
        axis = 1):

    flag_valid = (DopplerSpecdB>lowThreshold_dB+6)
    s_spec = DopplerSpecdB.shape
    s_vel = orderedDopplerVel.shape
    if s_spec == s_vel:
        VelMat = np.copy(orderedDopplerVel)
    else:
        VelMat = np.copy(np.broadcast_to(orderedDopplerVel,s_spec))

    VelMat[~flag_valid] = np.nan
    min_vel = np.nanmin(VelMat,axis)
    max_vel = np.nanmax(VelMat,axis)
    return min_vel, max_vel

def border_particle_vel_SNR(SNR_dB, orderedDopplerVel, axis = 1):

    flag_valid = (SNR_dB>6)
    s_spec = SNR_dB.shape
    s_vel = orderedDopplerVel.shape
    if s_spec == s_vel:
        VelMat = np.copy(orderedDopplerVel)
    else:
        VelMat = np.copy(np.broadcast_to(orderedDopplerVel,s_spec))

    VelMat[~flag_valid] = np.nan
    min_vel = np.nanmin(VelMat,axis)
    max_vel = np.nanmax(VelMat,axis)
    return min_vel, max_vel

def particle_vel_SNR(SNR_dB, orderedDopplerVel, axis = 0, thres = 6, 
                            size = 7,):
    dv = orderedDopplerVel[1]-orderedDopplerVel[0]
    rad = (size-1)/2
    ker_size = np.prod(size)    
    flag_valid = (uniform_filter((SNR_dB>thres).astype(float), size = size)>
                  1-(1/ker_size)/2)        
    s_spec = SNR_dB.shape
    s_vel = orderedDopplerVel.shape
    if s_spec == s_vel:
        VelMat = np.copy(orderedDopplerVel)
    else:
        VelMat = np.copy(np.broadcast_to(orderedDopplerVel,s_spec))
        
    VelMat = np.ma.array(VelMat, mask = ~flag_valid)   
    min_vel = VelMat.min(axis = axis) - dv*rad
    max_vel = VelMat.max(axis = axis) + dv*rad
    return min_vel, max_vel