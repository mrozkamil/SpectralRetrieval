#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:53:59 2020

@author: km357
"""

import xarray as xr
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import convolve
def hildebrand(raw_spec, spec_conv_wind = 1, freq_axis = 2, averaging_cycles = 1):
    
    ''' Hildebrand Sekhon noise floor estimation algorithm
       Input parameters:
       
       raw_spec - is an array of spectra
       spec_conv_wind - number of spectral line to be averaged, default value =1.
       freq_axis - axis of spectral lines, set to 2 if not given
       averaging_cycles - number of the cycles for the spectrum extimate
            
            NaN values in the input array are not allowed!  
       Output parameters:
       out - is a matrix of mean spectral noise power
    '''
    
    data_shape = np.array(raw_spec.shape)
    
    # In order to improve performance neighboring spectal lines can be optionally
    # averaged. In this case w has to be set to a value larger than 1. By default
    # no spectral averaging is applied (w = 1).
    ker_shape = np.ones_like(data_shape)
    ker_shape[freq_axis] = spec_conv_wind
    ker = np.ones(ker_shape)/spec_conv_wind
    spectrum = convolve(raw_spec, ker, mode = 'wrap')
    
    out = np.zeros(np.delete(data_shape, freq_axis))
    found_noise = np.full(out.shape, False)
    spectrum_sorted = np.sort(spectrum, axis = 2)
        
    n= data_shape[freq_axis]
    while np.any(~found_noise) & (n > 2):
        already_found = found_noise[:]
        s_red = spectrum_sorted[:,:,:n]
         
        # first raw moment (mean spectral power)
        p      = np.sum(s_red, axis = freq_axis)/n;       
        # second raw momemt
        q      = np.sum(s_red**2, axis = freq_axis)/n;      
        # second central moment (variance of spectral power)
        q      = q - p**2;     
        # expected variance of spectral power normalized by q 
        #(for white noise r2 is 1 in the case of signal present r2 < 1);
        r2     = p**2/q/spec_conv_wind/averaging_cycles
        n      += -1
        
        found_noise = (r2>1) | found_noise
        new_estimates = found_noise & (~already_found)
        out[new_estimates] = p[new_estimates]
    return out


def loadData(file_list, precise_time = False):
    if isinstance(file_list, list):
        test_file = file_list[0]
    elif isinstance(file_list, str):
        test_file = file_list

    with Dataset(test_file) as f:
        var_list = list(f.variables)

    if 'time' in var_list: time_name = 'time'
    elif 'Time' in var_list: time_name = 'Time'

    Data = xr.open_mfdataset(file_list,combine='by_coords', chunks = {time_name:1},
         compat='override', coords = 'minimal', data_vars = 'minimal')

    if ('time' not in Data.variables) and ('Time' in Data.variables):
        Data = Data.rename_dims({'Time': 'time'})
        Data = Data.rename({'Time': 'time'})
        
    for key in Data.variables:
        upper_keys_list = []
        for attr in Data[key].attrs.keys():
            if attr != attr.lower():
                upper_keys_list.append(attr)
        for upper_key in upper_keys_list:
            Data[key].attrs[upper_key.lower()] = Data[key].attrs[upper_key]          

    # if ('Units' in Data.time.attrs) and ('units' not in Data.time.attrs):
    #      Data.time.attrs['units'] = Data.time.attrs['Units']
    if ('long_name' in Data.time.attrs):
        Data.time.attrs['units'] = Data.time.long_name.replace(
            'time in sec', 'seconds').replace('Time in sec', 'seconds')
    Data.time.attrs['units'] = Data.time.attrs['units'].replace(
        'Number of seconds', 'seconds').replace(' 00:00:00 [UTC]','').replace(
            ' 00:00 UTC','')
    Data = xr.decode_cf(Data,decode_times=True)

    if precise_time:
        for key in Data.variables:
            if 'long_name' in Data[key].attrs:
                ln =  Data[key].attrs['long_name']            
            elif 'name' in Data[key].attrs: 
                ln =  Data[key].attrs['name']
            elif 'units' in Data[key].attrs: 
                ln =  Data[key].attrs['units']
            else:
                continue
            # print(key)
            # print(Data[key].attrs['long_name'])
            if ('millisecond' in ln ) or ('Millisecond' in ln):
                # print(key)
                time_delta =  Data[key].values*1e3*np.timedelta64(1,'us')
            if ('microsec' in ln) or ('Microsec' in ln) or (
                'Micro Seconds' in  ln   ):
                # print(key)
                time_delta =  Data[key].values*np.timedelta64(1,'us')                
        Data = Data.assign_coords({"time": Data.time + time_delta })
    return Data

def getSpect(dataSet, ch, noise_remove = True):
    #ch = o or x
    rangeLen = dataSet.range.shape[0]
    try:
        timeLen = dataSet.time.shape[0]
    except:
        timeLen = 1
    rangeData = dataSet.range.values.reshape(1, rangeLen, 1)
    radConst = dataSet.RadarConst.values.reshape(timeLen,1,1)
    SNRCorFa = dataSet['SNRCorFaC'+ch].values.reshape(timeLen,rangeLen,1)
    dv = 2.*dataSet.NyquistVelocity.values.item()/dataSet.nfft.values.item()
    # SPC = dataSet['SPCc'+ch].values

    SPC = dataSet['SPCc'+ch].values*1.
    HSD = dataSet['HSDc'+ch].values*1.
    HSD = HSD.reshape(timeLen, rangeLen, 1)
    if noise_remove:
        SPC = SPC-HSD
    npw = dataSet.npw1.values.reshape(timeLen,1,1)
    calSPC = radConst*SNRCorFa*(rangeData**2/5000.**2)*SPC/npw
    return calSPC/dv, SPC/HSD

def getSpectW(wData, ch = 0, noise_remove = True):    
    chirp = 'C%d' % (ch+1)
    rangeLen = wData[chirp+'Range'].shape[0]
    dopplerLen = wData[chirp+'Vel'].size
    dv = 2.*wData.MaxVel.values[ch]/dopplerLen
    try:
        timeLen = wData.time.shape[0]
    except:
        timeLen = 1
    # print(wData[chirp+'VSpec'].shape)
    specDataV = wData[chirp+'VSpec'].values*1.
    specDataV = specDataV.reshape(timeLen,rangeLen,dopplerLen)
    if not noise_remove:
        return specDataV/dv, np.full_like(specDataV, np.nan)
    
    aver_cycles = wData.AvgNum.values/wData.DoppLen.values
    noise_lin = hildebrand(specDataV, spec_conv_wind = 1, 
                    freq_axis = 2,averaging_cycles = aver_cycles[ch])
    
    calSPC = specDataV-noise_lin[:,:,np.newaxis]
    SNR = calSPC/noise_lin[:,:,np.newaxis]    
    return calSPC/dv, SNR

def getSelectedSpec(dataSet,sel_time, ch = 'o', vel_lims = (-2.,10.),
                    noise_remove = True):
    pd_time = pd.to_datetime(sel_time)
    sel_Data = dataSet.sel({"time": pd_time}, method = 'nearest')  
    vel = getSortedVel(dataSet, ch = ch, vel_lims = vel_lims)
    if ch in [0,1,2,3]:
        chirp = 'C%d' % (ch+1)  
        sel_Data = sel_Data.sel({chirp+'Vel' : vel})
        spec, snr = getSpectW(sel_Data, ch = ch, noise_remove= noise_remove)
    else:              
        sel_Data = sel_Data.sel(doppler = vel)
        spec, snr = getSpect(sel_Data, ch = ch)       
    i_sort = np.argsort(vel)
    return spec[:,:,i_sort], snr[:,:,i_sort], vel[i_sort]

def getSortedVel(dataSet, ch = 'o', vel_lims = (-2.,10.)):
    if ch in [0,1,2,3]:
        chirp = 'C%d' % (ch+1)
        vel= dataSet[chirp+'Vel'].values 
    else:
        vel= dataSet.doppler.values
    vel = np.sort(vel) 
    if not (vel_lims is None): 
        flag = (vel>vel_lims[0]) & (vel<=vel_lims[1])
        vel = vel[flag]        
    return vel
