#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:00:29 2019

@author: km357
"""

import numpy as np
import h5py
from scipy.interpolate import  RegularGridInterpolator
import os

def mu_dm_2_sigma(mu,dm):
    return dm/np.sqrt(4+mu)

def dm_matrosov(ddv):
    dm = np.full(ddv.shape,-1.)
    flag = (ddv>1) & (ddv<2.4)
    s_ddv = ddv[flag]
    dm[flag] = 1.338-0.977*s_ddv + 0.678*s_ddv**2 - 0.079*s_ddv**3
    flag = (ddv<=1) & (ddv>0)
    dm[flag] = 0.47 + 0.49 * np.power(ddv[flag],0.54)
    return dm

def mu_williams(dm):
    return np.minimum(18,(11.1* (dm**(-0.72))) -4)
def sigm_williams(dm):
    return 0.3 * np.power(dm,1.36)

def load_brenda_data(data_path):
    h5_name = os.path.join(data_path, 'mean_dm_mu_sigma_dist_weight.h5')
    brenda_data = {}
    hf1 = h5py.File(h5_name, 'r')
    for var in hf1.keys():
        brenda_data[var] = hf1[var][:]
    hf1.close()
    return brenda_data

def load_leo_data(data_path):
    dis_stats={}
    for instr in ['2DVD','Pars']:   
        dis_stats[instr] ={}
        h5_file = os.path.join(data_path, 'Dm_Sm_LUT_%s.h5' % instr )
        hf1 = h5py.File(h5_file, 'r')         
        for var in hf1.keys():
            # print(var)
            dis_stats[instr][var] = hf1[var][:]
        hf1.close()
    return dis_stats


leo_data_path = os.path.join('/data/doppler/DSD_datasets', 'LeoDataset')
brenda_data_path = os.path.join('/data/doppler/DSD_datasets', 'BrendaData')

leo_data    = load_leo_data(leo_data_path)
brenda_data = load_brenda_data(brenda_data_path)

dm_leo   = {}
sigm_leo = {}
std_sigm_leo = {}
dm_leo_XW ={}
std_dm_leo ={}
for inst in ['Pars', '2DVD']:   
    dm_leo[inst] = RegularGridInterpolator(
            (leo_data[inst]['ddv_KaW_grid'],leo_data[inst]['ddv_XKa_grid']), 
            leo_data[inst]['dm_for_ddv_KaW_XKa'].T, bounds_error = False)    
    std_dm_leo[inst] = RegularGridInterpolator(
            (leo_data[inst]['ddv_KaW_grid'],leo_data[inst]['ddv_XKa_grid']), 
            leo_data[inst]['dm_std_for_ddv_KaW_XKa'].T, bounds_error = False) 
    sigm_leo[inst] = RegularGridInterpolator(
            (leo_data[inst]['ddv_KaW_grid'],leo_data[inst]['ddv_XKa_grid']), 
            leo_data[inst]['sm_for_ddv_KaW_XKa'].T, bounds_error = False)
    std_sigm_leo[inst] = RegularGridInterpolator(
            (leo_data[inst]['ddv_KaW_grid'],leo_data[inst]['ddv_XKa_grid']), 
            leo_data[inst]['sm_std_for_ddv_KaW_XKa'].T, bounds_error = False)
    dm_leo_XW[inst] = RegularGridInterpolator(
            (leo_data['2DVD']['ddv_XW_grid'],), 
            leo_data[inst]['dm_for_ddv_XW'], bounds_error = False) 


dm_brenda = RegularGridInterpolator((brenda_data['DDV_ka_w_test'],
                                     brenda_data['DDV_x_ka_test']), 
                            brenda_data['near_dm'].T,
                            bounds_error = False,fill_value = np.nan)
    
sigma_brenda = RegularGridInterpolator((brenda_data['DDV_ka_w_test'],
                                     brenda_data['DDV_x_ka_test']), 
                            brenda_data['near_sigm'].T,
                            bounds_error = False,fill_value = np.nan)
