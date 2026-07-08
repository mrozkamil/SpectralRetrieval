#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:39:14 2019

@author: km357
"""
import numpy as np
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