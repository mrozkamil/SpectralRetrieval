#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:53:59 2020

@author: km357
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
# import os
from skimage import restoration
from scipy.ndimage import uniform_filter

# from scipy.interpolate import splrep, splev
# from scipy.linalg import lu
from scipy.interpolate import BSpline

# from scipy.signal import wiener


# class _ksq():
#     def __init__(self, temp = 278.):  
#         from scipy.interpolate import InterpolatedUnivariateSpline as IUS
#         self.speed_of_light =299792458.0
#         home = os.path.expanduser('~')
#         data_ri_path = os.path.join(home, 'Data/LUT_pytmatrix/refractive/')
#         f_name = 'water_refractive_index_t%d.dat' % (temp,)
        
#         ri_table= np.loadtxt(os.path.join(data_ri_path, f_name), comments = '%')
#         self.Ksq = IUS(ri_table[:,0], ri_table[:,3], k =1)
#         self.freq = {'L': 0.915, 'S': 2.8, 'C': 5.6, 'X': 9.6,
#             'Ku': 13.6, 'K': 24.,'Ka': 35.5, 'W': 94., 'G': 220.}
        
#         self.wave_length = {key: self.speed_of_light/self.freq[key]*1e-6 
#                             for key in self.freq}
        
#     def freq_to_wave_length(self,freq):
#         return self.speed_of_light/freq*1e-6
    
#     def __call__(self, x, unit = 'GHz'):
#         if unit=='GHz':
#             wl = self.freq_to_wave_length(x)
#         elif unit =='mm':
#             wl = np.copy(x)       
#         else:
#             print(unit)
#             print('only input in [mm] or [GHz] is accepted')
#             return
#         return self.Ksq(wl)
    
# Ksq = _ksq()

def gamma_mu(dm,sm):
    return (dm/sm)**2-4

def dB(val):
    return 10*np.log10(val)
def invdB(val):
    return np.power(10, val/10)
    
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def map_R_to_int(x, y_min, y_max):
    dy = y_max-y_min
    return sigmoid(x)*dy +y_min
    
def map_int_to_R(y, y_min, y_max):
    dy = y_max-y_min
    y_0_1 = (y-y_min)/dy
    return np.log(y_0_1)- np.log(1-y_0_1)
    
def BB_ext_Matr(rr= 1.,band = 'w'):
    if band == 'w':
      return 2.6*rr**0.87
    if band == 'ka':
      return 0.66*rr**1.1
    if band == 'x':
      return 0.048*rr**1.05
def simple_gauss_deconv(x,y,dx, width = 0.1, mean_x = None):
    if mean_x is None:
        mean_x = np.trapz(y*x,x)/np.trapz(y,x)
    sc = 1-width/(x[-1]-x[0])
    x_new = (x-mean_x)*sc+mean_x
    # y_new = y/(1-width)    
    int_y_new = np.cumsum(y, )*dx
    int_y_old = np.interp(x,x_new,int_y_new)    
    y_new_old_res = np.diff(int_y_old,prepend=y[0])/np.diff(x,prepend=1.)   
    return y_new_old_res

def deconv(y, ker):    
    if (len(y.shape)==1) and (len(ker.shape)==1) :
        Y = y[np.newaxis,:]
        K = ker[np.newaxis,:]
    else:
        Y = y[:]
        K = ker[:]
    y_res = restoration.richardson_lucy(Y, K, iterations=30, clip = False,)
    # y_res = restoration.unsupervised_wiener(Y, K, clip = False,)
    # y_res = restoration.wiener(Y, K, clip = False,balance = 1e-3)
    return y_res

def ffill(arr,mask=None):
    '''Solution provided by Divakar.'''
    if mask is None:
        mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def mfill(data, mask=None, order = 'C'):
    if mask is None:
        bad_indexes = np.isnan(data)
    else:
        bad_indexes = mask
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(
        np.nonzero(np.ravel(bad_indexes, order = order))[0], 
        np.nonzero(np.ravel(good_indexes, order = order))[0], good_data)
    data[bad_indexes] = interpolated
    return data
    


def calc_parts_and_shifts(time_shift, vel, height ):
    '''
    time shift in [s]
    vel is Doppler velocity domain in [m/s], e.g. sampled particle speed
    height is a distance from the radar in [m]
    '''
    delta_height = np.median(np.diff(height))
    vert_displacement = time_shift*vel
    fract_displaced = vert_displacement/delta_height

    part_down = np.mod(np.abs(fract_displaced[np.newaxis,np.newaxis,:]),1.0)
    part_up = 1-part_down

    tmp_vect = (-(np.ceil( np.abs( fract_displaced)))*np.sign(
                        fract_displaced)).astype(int)
    part_up_shift = tmp_vect[np.newaxis,np.newaxis,:]

    tmp_vect = (-(np.floor( np.abs( fract_displaced)))*np.sign(
                        fract_displaced)).astype(int)
    part_down_shift = tmp_vect[np.newaxis,np.newaxis,:]

    return ([part_up, part_up_shift, np.unique(part_up_shift)],
            [part_down, part_down_shift, np.unique(part_down_shift)])

def evolve_spectrum(spec, time_shift, vel, height, parts_and_shifts = None,
                    plot = False, ):
    if parts_and_shifts is None:
        parts_and_shifts =  calc_parts_and_shifts(time_shift, vel, height )

    spec_sed = np.zeros_like(spec)
    for part, shift, uniq_sh in parts_and_shifts:
        spec_tmp = spec*part
        for sh in uniq_sh:
            flag = (shift==sh)
            tmp_arr = np.where(flag,spec_tmp,0.)
            spec_sed += np.roll(tmp_arr,sh, axis = 1)

    if plot:
        plt.figure()
        plt.pcolormesh(vel, height, 10*np.log10(spec_sed[0]),
                    cmap = 'jet', vmin = -30, vmax = 10)
        plt.ylim(0,5e3)
        plt.xlim(-2,9)
        plt.colorbar()
        plt.show(block = False)
    return spec_sed

def average_spectrum(spec, dopp, vel_min,):
    
    ind_low = np.interp(vel_min, dopp, np.arange(dopp.size))
    ind_low_or = np.round(ind_low).astype(int)
    ind_where_shift = np.where(np.abs(vel_min)<9.)
    sp_mod = np.copy(spec)
    for rr,cc in zip(ind_where_shift[0],ind_where_shift[1]):            
        sp_mod[rr,cc,:] = np.roll(spec[rr,cc,:],-ind_low_or[rr,cc],)
               
    sp_aver = np.mean(sp_mod, axis = 0)
    vel_min_new = np.copy(vel_min)
    vel_min_new[np.abs(vel_min)>=9.] = np.nan
    mean_vel = np.nanmean(vel_min_new, axis = 0)    
    ind_low = np.interp(mean_vel, dopp, np.arange(dopp.size))
    ind_low[~np.isfinite(ind_low)] = 0
    ind_low_or = np.round(ind_low).astype(int)
    ind_where_shift = np.where(ind_low>0)
    
    
    for rr in ind_where_shift[0]:            
        sp_aver[rr,:] = np.roll(sp_aver[rr,:],ind_low_or[rr],)
    return sp_aver
        
def LocalLinFit(X_arr, Y_arr, W_arr = None, kernel_size = 30, 
                Lam = None, return_cc = False):
    """

    Parameters
    ----------
    X_arr : numpy ndarray
        An array of x coordinates.
    Y_arr : numpy ndarray
        An array of y coordinates.
    W_arr : numpy ndarray, optional
        An array of weights for linear fit. The default is None.
    kernel_size : numpy ndarray or int, optional
        window size used for regression. The default is 30.
    Lam : int or array like, optional
        the regularization term. The default is None.
    return_cc : bool
        if True the correlation coefficient is also returned

    Returns
    -------
    alpha : numpy ndarray
        scaling factor next to x.
    beta : numpy ndarray
        constant term.
    locW : numpy ndarray
        the sum of weights.
    cc : numpy ndarray
        the correlation coefficient
    y = alpha X + beta

    """
    if W_arr is None:
        W_arr = np.ones(X_arr.shape)       

    locW = uniform_filter(W_arr, size = kernel_size)    
    if np.isscalar(Lam):
        norm_beta = Lam
        norm_alpha = Lam
    elif isinstance(Lam, (list, tuple, np.ndarray)):
        norm_beta = Lam[1]
        norm_alpha = Lam[0]
    else:
        norm_beta = 0
        norm_alpha = 0
    locWn = locW+norm_beta
    
    
    locX = uniform_filter(W_arr*X_arr, size = kernel_size)
    locY = uniform_filter(W_arr*Y_arr, size = kernel_size)
    locXY = uniform_filter(W_arr*Y_arr*X_arr, size = kernel_size)
    
    locXX = uniform_filter(W_arr*(X_arr**2), size = kernel_size)
    locYY = uniform_filter(W_arr*(Y_arr**2), size = kernel_size)

    alpha =(locWn*locXY -locX*locY)/(
        (locXX+norm_alpha)*locWn - locX**2)    
  
    beta = (locY - (alpha*locX))/locWn
    if return_cc:
        cc = (locW*locXY -locX*locY)/np.sqrt((
        locXX*locW - locX**2)*(locYY*locW - locY**2))
        return alpha,beta, locW, cc
    # beta[beta<0] = 0.
    return alpha,beta, locW

def sens(r,a,b):
    return a + b*10*np.log10(r)
def get_sens(Z,Range_km, time_axis =0):    
    Z[~np.isfinite(Z)] = np.nan
    Z[Z<-99.] = np.nan
    ZZ = np.moveaxis(Z, time_axis, 0)
    Z_bins = np.arange(-90,60,0.5)
    hist_arr = np.array([np.histogram(ZZ[:,ii].ravel(), bins = Z_bins)[0] 
                 for ii in range(ZZ.shape[1])])
    hist_cum = np.cumsum(hist_arr, axis = 1)
    ind_min = np.argmax(hist_cum>25, axis = 1)
    ind = (ind_min>0) & (Range_km>1.) 
    prc = Z_bins[ind_min]
     
    # prc = np.nanpercentile(Z, q = 0.0001, axis = time_axis)    
    # ind = (np.isfinite(prc)) & (Range_km>1.)    
    
    popt, pcov = curve_fit(sens,
            Range_km[ind],prc[ind], p0 = np.array([-60,2]))   
    print(popt)    
    # plt.figure()
    # plt.plot(Range_km,prc)
    # plt.plot(Range_km,sens(Range_km,*popt))
    # plt.grid()
    return (lambda r: sens(r,*popt), pcov)
def calc_std(SNR,M=10):
    return 4.343/np.sqrt(M)*(1+10**(-SNR/10))


def spline_mat(nodes, x = None, k = 1, inner_only = True):
    """
    Calculates the spline matrix for the smooth expansion
    
    Parameters
    ----------
    nodes : :class: 'numpy.ndarray'
        Defines spline nodes
    x : :class: 'numpy.ndarray';
        the domain of basis functions
    k : 'int'
        the order of the spline polynomial
    inner_only: bool
        determines if the basis functions with the peaks outside the domain are
        summed up or not
    Returns
    -------
    spline_matrix : `numpy.ndarray`
        Expansion coeficients: size = (n_points,n_functions)
    """
    if x is None:
        x = np.linspace(nodes[0],nodes[-1], 128)[1:] 
        
    basis_size = nodes.size - k -1
    tmp_arr = np.zeros((x.size,basis_size,))
    peaks = np.zeros((basis_size,))
    
    for ii in  range(basis_size):
        t = nodes[ii:ii+2+k]                
        be = BSpline.basis_element(t = t, extrapolate = False)
        tmp_arr[:,ii] = be(x)
        peaks[ii] = np.mean(t) 
    tmp_arr[np.isnan(tmp_arr)] = 0.
    
    if inner_only:
        ind_within = np.where((peaks>=x[0]) & (peaks<x[-1]))[0]
        ind_below = np.where((peaks<x[0]))[0]
        ind_above = np.where((peaks>=x[-1]))[0]
        
        if ind_below.size>0:
            firs_element = np.sum(tmp_arr[:,ind_below], axis = 1,
                                  keepdims = True)           
        else:
            firs_element = np.zeros((x.size,0))
            
        if ind_above.size>0:
            last_element = np.sum(tmp_arr[:,ind_above], axis = 1,
                                  keepdims = True)            
        else:
            last_element = np.zeros((x.size,0))            
        
        basis = np.concatenate((firs_element, 
                    tmp_arr[:,ind_within],last_element), axis = 1)
       
    else:
        basis = tmp_arr
    
    w = np.trapz(basis, x=x, axis = 0)
    wx = np.trapz(basis*x[:, np.newaxis], x=x, axis = 0)
    
    
    ind_non_zero = np.where(w>1e-9)[0]
    basis_peaks = wx/w    
    final_basis_size = ind_non_zero.size
    
    # basis_peaks = np.zeros(final_basis_size)
    # for ii in range(final_basis_size):
    #     basis_peaks[ii] = (np.trapz(basis[:,ii]*x, x=x)/
    #                        np.trapz(basis[:,ii], x=x))
    
    plt.figure()
    plt.plot(peaks, np.ones(basis_size),'x')
    plt.plot(basis_peaks[ind_non_zero], np.ones(final_basis_size),'o')
    for ii in ind_non_zero:
        plt.plot(x, basis[:,ii])
        
    return basis[:,ind_non_zero], basis_peaks[ind_non_zero], final_basis_size, x

def funct_2_proj_matrix(basis_functions, weights = None, tikhonov_norm = None, 
        eps = 1e-6, ):
    """
    Calculates the projection matrix from a set of basis functions, ie: 
        P = (B^T B)^-1 B^T  
    where B is an array of the basis functions, if weights are given:
        P = (B^T W B)^-1 B^T W
        
    if Tikhonov normalization term is given:
        P = (B^T W B + T)^-1 B^T W
    Parameters
    ----------

    basis_functions : :class:`numpy:numpy.ndarray`
        basis functions, the i-th column is the evaluation of the i-th 
        basis function at the sampling space
        basis_functions.shape = (n,m); m - the number of the basis functions
                                       n - the number of sampling points
                                       
    weights  : :class:`numpy:numpy.ndarray`
        default value = None
        the inverse of the covariance matrix of the observations, 
        must be symmmetric and positive, in case it is a vector, transformed to 
        a diagonal matrix
        
    tikhonov_norm : :class:`numpy:numpy.ndarray`
        default value = None
        the Tikhonov normalization matrix 
        must be symmmetric and positive
        
    eps: float
        default value 1e-5
        the threshold for detecting ill posed problem, i.e. when 
            det(B^T W B + T)<eps
    Returns
    -------
    Proj_matrix : `numpy.ndarray`
        Projection matrix: size = (m, m)
    """
    X = basis_functions
    if weights is not None:
        if weights.size == weights.shape[0]:
            W = np.diag(weights)
        elif weights.shape[1] == X.shape[0]:
            W = weights
    else:
        W = np.eye(X.shape[0])
        
    if tikhonov_norm is None:
        T = np.zeros((X.shape[1], X.shape[1]))
    elif isinstance(tikhonov_norm, (float,int)):
        T = np.eye(X.shape[1])*tikhonov_norm
    else: 
        T = tikhonov_norm
        
    XTX = np.matmul(X.T,np.matmul(W,X)) + T
    
    if np.abs(np.linalg.det(XTX))<eps:
        # print('problem ill posed, use tikhonov_norm to fix this;' + 
        #     'tikhonov_norm must be a scalar or a square matrix of ' + 
        #     'the size of x, current eigenvalues of XTX = ' + 
        #      np.array2string(np.linalg.eigvals(XTX), 
        #      formatter={'float_kind':lambda x: "%.6f," % x}))
        return np.linalg.lstsq(XTX, np.matmul(X.T,W))[0]
    else:
        return np.linalg.solve(XTX, np.matmul(X.T,W))

def project(data, X, weights = None,  tikhonov_norm = None):
    """
    Calculates the coeficients of the linear expansion of 
    the data: 
        data = sum_i c_i *basis_funct_i
    
    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray` 
        An array of the sampling points,
        data.shape = (n,p); 
    basis_functions : :class:`numpy:numpy.ndarray`
        basis functions, the i-th column is the evaluation of the i-th 
        basis function at the sampling space
        basis_functions.shape = (n,m); m - the number of the basis functions
                                       n - the number of sampling points
    weights  : :class:`numpy:numpy.ndarray`
        default value = None
        the inverse of the covariance matrix of the observations, 
        must be symmmetric and positive, in case it is a vector, transformed to 
        a diagonal matrix
        
    tikhonov_norm : :class:`numpy:numpy.ndarray`
        default value = None
        the Tikhonov normalization matrix 
        must be symmmetric and positive
    Returns
    -------
    exp_coeff : `numpy.ndarray`
        Expansion coeficients: size = (m, p)
    """
    XX = funct_2_proj_matrix(basis_functions= X, weights = weights, 
            tikhonov_norm = tikhonov_norm)
    return np.matmul(XX, data)
    
def expand(series, expansion_matrix):
    """
    Calculates the value of the linear expansion of a series with a given set of basis functions: 
        data = sum_i c_i *basis_funct_i
    
    Parameters
    ----------
    series : :class:`numpy:numpy.ndarray` 
        An array of the expansion coefficients,
        series.shape = (m,p); 
    basis_functions : :class:`numpy:numpy.ndarray`
        basis functions, the i-th column is the evaluation of the i-th 
        basis function at the sampling space
        basis_functions.shape = (n,m); m - the number of the basis functions
                                       n - the number of sampling points
    Returns
    -------
    data : `numpy.ndarray`
        Expansion coeficients: size = (n, p)
    """
    return np.matmul(expansion_matrix, series)

def difference_mat(n):
    T = np.eye(n)- np.eye(n,k=1)
    T[-1,:] = 0
    return T

def reg_matrix(n, diff = 1):
    A = difference_mat(n)
    return np.matmul(np.linalg.matrix_power(A,diff).T, 
                     np.linalg.matrix_power(A,diff))
    

def roll_pad(data, shift, constant=0., axis = 0):
    """
    Shifts the array in two dimensions while setting rolled values to constant
    :param data: The 2d numpy array to be shifted
    :param shift: The shift 
    :param constant: The constant to replace rolled values with
    :return: The shifted array with "constant" where roll occurs
    """
    shifted_data = np.swapaxes(data, axis, 0)
    shifted_data = np.roll(shifted_data, shift, axis = 0)    
    if shift < 0:
        shifted_data[shift:] = constant
    elif shift > 0:
        shifted_data[:shift] = constant
    return np.swapaxes(shifted_data,0, axis) 
    





