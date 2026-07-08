#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:28:33 2019

@author: km357
"""

import os
from  scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
#RectBivariateSpline, interp1d, Rbf

import numpy as np
import h5py

    
def load_lut(bands = None, hydrometeros = None, ):
    base_dir = os.path.dirname(__file__)
    lut_dir = os.path.join(base_dir,'LUT')
    lut_dir_rain = os.path.join(lut_dir,'rain')
    lut_dir_snow = os.path.join(lut_dir,'snow')
    rain_MODELS = ['mie','tmat','tmat_cant']
    print(lut_dir)
    rain_files={}
    rain_files['mie'] = {'w': 'Rain_SratiowithTDop94GHz.mat', 
                  'ka': 'Rain_SratiowithTDop35p5GHz.mat', 
                  'ku': 'Rain_SratiowithTDop13p6GHz.mat', 
                  'x': 'Rain_SratiowithTDop9p6GHz.mat',
                  's': 'Rain_SratiowithTDop3GHz.mat'}
    rain_files['tmat'] = {'w': 'Rain_SratiowithTandTmatrix94GHz.mat', 
                  'ka': 'Rain_SratiowithTandTmatrix35GHz.mat', 
                  'ku': 'Rain_SratiowithTandTmatrix13GHz.mat', 
                  'x': 'Rain_SratiowithTandTmatrix9p6GHz.mat'}
    
    rain_files['tmat_cant'] = { 'c': 'rain_C_band_with_temp_T_matrix_leinonen.h5',
                  's': 'rain_S_band_with_temp_T_matrix_leinonen.h5',
                  'ka': 'rain_Ka_band_with_temp_T_matrix_leinonen.h5', 
                  'ku': 'rain_Ku_band_with_temp_T_matrix_leinonen.h5', 
                  'x': 'rain_X_band_with_temp_T_matrix_leinonen.h5'}

    snow_files = ['SSRGA_LUTvMarch2018.mat',
              'Spheres_iceLUTvMarch2018.mat']
    if hydrometeros is None:
        hydrometeros=['rain', 'snow', 'cloud']
    if bands is None:
            bands = ['w','ka','ku','x']
    
    if 'rain' in hydrometeros:
        rain = {}
        for model in rain_MODELS:
            rain[model] = {}
            for band in bands:
                if band in rain_files[model].keys(): 
                    rain[model][band] = {}
                else:
                    continue
                if model == 'mie' or model =='tmat':
                    fileLUT = loadmat(os.path.join(lut_dir_rain,rain_files[model][band]))    
                    dm = fileLUT['LUT4Forward_water'][0,0]['D_m'].ravel()
                    mu = fileLUT['LUT4Forward_water'][0,0]['mu'].ravel()
                    temp = fileLUT['LUT4Forward_water'][0,0]['Temp'].ravel()
                    precip = fileLUT['LUT4Forward_water'][0,0]['Flux_perunitmass1000mb'][:,:,0]
                    dopp = fileLUT['LUT4Forward_water'][0,0]['vDoppler']
                    refl = fileLUT['LUT4Forward_water'][0,0]['Z_coeff_watunitmass']
                    ext = fileLUT['LUT4Forward_water'][0,0]['Ext_coeff_watunitmass']
                    g = fileLUT['LUT4Forward_water'][0,0]['g']
                    S = fileLUT['LUT4Forward_water'][0,0]['S']
                    alb = fileLUT['LUT4Forward_water'][0,0]['albedo']
                elif model == 'tmat_cant':
                    fileLUT = h5py.File(os.path.join(
                            lut_dir_rain,rain_files[model][band]),'r')    

                    dm = fileLUT['dm'][:].ravel()
                    mu = fileLUT['mu'][:].ravel()
                    temp = fileLUT['temp'][:].ravel()
                    precip = fileLUT['massflux'][:]
                    dopp = fileLUT['doppler_1000hPa'][:]
                    refl = fileLUT['refl'][:]
                    ext = fileLUT['ext'][:]
                    g = fileLUT['asym'][:]
                    S = np.zeros_like(fileLUT['asym'])#fileLUT['ext_2_bscat_ratio'][:]
                    alb = fileLUT['ssa'][:]
                    fileLUT.close()
                
                rain[model][band]['precip'] = RegularGridInterpolator((dm,mu), precip, 
                         method='linear', bounds_error=False, fill_value=np.nan)
                rain[model][band]['dopp'] = RegularGridInterpolator((dm,mu,temp), dopp, 
                         method='linear', bounds_error=False, fill_value=np.nan)
                rain[model][band]['refl'] = RegularGridInterpolator((dm,mu,temp), refl, 
                         method='linear', bounds_error=False, fill_value=np.nan)
                rain[model][band]['ext'] = RegularGridInterpolator((dm,mu,temp), ext, 
                         method='linear', bounds_error=False, fill_value=np.nan)
                rain[model][band]['alb'] = RegularGridInterpolator((dm,mu,temp), alb, 
                         method='linear', bounds_error=False, fill_value=np.nan)
                rain[model][band]['g'] = RegularGridInterpolator((dm,mu,temp), g, 
                         method='linear', bounds_error=False, fill_value=np.nan)
                rain[model][band]['S'] = RegularGridInterpolator((dm,mu,temp), S, 
                         method='linear', bounds_error=False, fill_value=np.nan)
            
#            dm_test = np.linspace(0.1,5,101)
#            temp_test = 280*np.ones_like(dm_test)
#            plt.figure()
#            for mm in range(-3,7,2):
#                mu_test = mm*np.ones_like(dm_test)
#                plt.plot(dm_test, rain[model][band]['dopp']((dm_test,mu_test,temp_test)))
        if 'snow' in hydrometeros:
            band_freq = {'s': 3., 'c': 5., 'x': 9.6, 'ku': 13.6, 'ka': 35.5,
                         'w': 94., 'g': 200.}
            snow = {}
            for f_name in snow_files:
                fileLUT = loadmat(os.path.join(lut_dir_snow,f_name))
                n_models = fileLUT['LUT4Forward'][0,0]['MODEL'].shape[1]
                
                for ii in range(n_models):
                    model = fileLUT['LUT4Forward'][0,0]['MODEL'][0,ii][0]
                    snow[model]={}
                    dm = fileLUT['LUT4Forward'][0,0]['D_mwatereqPSD'][0,ii].ravel()*1e3
                    temp = fileLUT['LUT4Forward'][0,0]['Temp'].ravel()
                    freq = fileLUT['LUT4Forward'][0,0]['Freq'].ravel()
                    
                    press_norm = np.sqrt(0.7)
                    precip = fileLUT['LUT4Forward'][0,0]['Flux_700mb'][0,ii]*press_norm                
                    dopp = fileLUT['LUT4Forward'][0,0]['v_Doppler700mb'][0,ii]*press_norm
                    
                    refl = fileLUT['LUT4Forward'][0,0]['Z_coeff_snowunitmass'][0,ii]-30
                    ext = fileLUT['LUT4Forward'][0,0]['Ext_coeff_snowunitmass'][0,ii]
                    g = fileLUT['LUT4Forward'][0,0]['g'][0,ii]
                    S = fileLUT['LUT4Forward'][0,0]['S'][0,ii]
                    alb = fileLUT['LUT4Forward'][0,0]['albedo'][0,ii]
                    
                    for band in bands:
                        ind_band = np.argmin((freq-band_freq[band])**2)
                        if (freq[ind_band]-band_freq[band])**2<1:
                            snow[model][band]={}
                        else:
                            print('no %s-band data for snow %s' % (band, model))
                            continue

                        snow[model][band]['precip'] = RegularGridInterpolator(
                                (dm,temp), precip.T, method='linear',
                                  bounds_error=False, fill_value=np.nan)
                        snow[model][band]['dopp'] = RegularGridInterpolator(
                                (dm,temp), dopp[ind_band,:,:].T, method='linear', 
                                 bounds_error=False, fill_value=np.nan)
                        snow[model][band]['refl'] = RegularGridInterpolator(
                                (dm,temp), refl[ind_band,:,:].T, method='linear',
                                  bounds_error=False, fill_value=np.nan)
                        snow[model][band]['ext'] = RegularGridInterpolator(
                                (dm,temp), ext[ind_band,:,:].T, method='linear',
                                  bounds_error=False, fill_value=np.nan)
                        snow[model][band]['alb'] = RegularGridInterpolator(
                                (dm,temp), alb[ind_band,:,:].T, method='linear',
                                  bounds_error=False, fill_value=np.nan)
                        snow[model][band]['g'] = RegularGridInterpolator(
                                (dm,temp), g[ind_band,:,:].T, method='linear',
                                  bounds_error=False, fill_value=np.nan)
                        snow[model][band]['S'] = RegularGridInterpolator(
                                (dm,temp), S[ind_band,:,:].T, method='linear',
                                  bounds_error=False, fill_value=np.nan)
                
    return snow, rain
snow, rain = load_lut()   


def doppler(dm, mu = 3, temp = 288.15, band = 'ka', pressure = None, wind = None,
             rain_flag = None, snow_flag = None, snow_model = 'leinonenA0p2kgm2',
             rain_model = 'tmat'):
    
    """
    Doppler radar simulator,
    for 2d data we assume that rows of the array are the altitude levels
    dm is the mean mass weighted melted diameter in [mm]
    pressure in hPa
    altitude in km
    wind in m/s
    band is one of 's','x','ku','ka','w'
    rain_flag: array of Booleans 
    snow_flag: array of Booleans
    snow_model is one of : 'leinonenA0p2kgm2',
    rain_model is one of 'tmat', 'mie', 'tmat_cant'
    """
    
    if (rain_flag is None) and (snow_flag is None):
        rain_flag = np.full(dm.shape,True)
     
    if np.isscalar(mu):
        MU = mu*np.ones(dm.shape)
    elif mu.shape ==dm.shape:
        MU = np.copy(mu)
    if np.isscalar(temp):
        TEMP = temp*np.ones(dm.shape)
    elif temp.shape ==dm.shape:
        TEMP = np.copy(temp)
      
    doppler_vel = np.full(dm.shape, np.NaN)
    doppler_vel[rain_flag] = rain[rain_model][band]['dopp']((
            dm[rain_flag],MU[rain_flag],TEMP[rain_flag]))
    if not (snow_flag is None):
         doppler_vel[snow_flag] = snow[snow_model][band]['dopp']((
            dm[snow_flag],TEMP[snow_flag]))
    if not (pressure is None):
        press_corr = np.sqrt(1000/pressure)
        doppler_vel *=  press_corr
    if not (wind is None):
        doppler_vel += wind
    
    return doppler_vel



