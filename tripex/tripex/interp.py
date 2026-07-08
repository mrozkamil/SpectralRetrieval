#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:53:59 2020

@author: km357
"""


import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline


def interp2(x,y,z,xi,yi):
    fl = np.isfinite(z)
    z[~fl] = -99.9
    
    d0 = np.datetime64('2000-01-01')
    dd = np.timedelta64(1000,'ms')
    if np.issubdtype(x.dtype, np.datetime64):
        xx = (x-d0)/dd
        xxi = (xi-d0)/dd
    else: xx = x; xxi = xi
    if np.issubdtype(y.dtype, np.datetime64):
        yy = (y-d0)/dd
        yyi = (yi-d0)/dd
    else: yy = y; yyi = yi
        
    
    tmp_i = RectBivariateSpline(xx, yy, z, kx =1, ky=1,s=0)
    zi = tmp_i(xxi,yyi,  grid=True)
    if np.all(fl):
        return zi
    else:
        tmp_i = RectBivariateSpline(xx, yy, fl.astype('float'), kx =1, ky=1,s=0)
        fi = tmp_i(xxi,yyi,  grid=True)
        zi[fi<.75] = np.nan
        return zi
    
def interp1(x,z,xi,):
    fl = np.isfinite(z)
    z[~fl] = -99.9
    
    d0 = np.datetime64('2000-01-01')
    dd = np.timedelta64(1000,'ms')
    if np.issubdtype(x.dtype, np.datetime64):
        xx = (x-d0)/dd
        xxi = (xi-d0)/dd
    else: xx = x; xxi = xi    
        
    
    tmp_i = UnivariateSpline(xx, z, k =1, s=0)
    zi = tmp_i(xxi, )
    if np.all(fl):
        return zi
    else:
        tmp_i = UnivariateSpline(xx, fl.astype('float'), k =1,s=0)
        fi = tmp_i(xxi, )
        zi[fi<.75] = np.nan
        return zi