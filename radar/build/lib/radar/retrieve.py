#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:36:57 2019

@author: km357
"""
import numpy as np
from . import simulate
from scipy.spatial import cKDTree
#from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def dm_from_ddv_ka_w(ddv):
    flag = ddv<1
    dm = np.zeros(flag.shape)
    s_ddv = ddv[~flag]
    dm[~flag] = 1.338-0.977*s_ddv + 0.678*s_ddv**2 - 0.079*s_ddv**3
    dm[flag] = 0.47 + 0.49 * np.power(ddv[flag],0.54)
    return dm

def _sim_ddv_x_ka(DM,MU,T):
    return (simulate.rain['tmat']['x']['dopp']((DM,MU,T)) - 
                   simulate.rain['tmat']['ka']['dopp']((DM,MU,T)))
def _sim_ddv_ka_w(DM,MU,T):
    return (simulate.rain['tmat']['ka']['dopp']((DM,MU,T)) - 
                    simulate.rain['tmat']['w']['dopp']((DM,MU,T)))

def _mu_williams(dm):
    return np.minimum(18,(11.1* (dm**(-0.72))) -4)

def _generate_DDV_tree():
    dm_test = np.arange(1., 3.55,0.05)
    mu_test = np.arange(-1, 18, 0.5)
    temp_test = np.arange(275, 301,5)
    
    DM,MU,T = np.meshgrid(dm_test,mu_test,temp_test)
    
    LUT_ddv_x_ka = _sim_ddv_x_ka(DM,MU,T)        
    LUT_ddv_ka_w = _sim_ddv_ka_w(DM,MU,T)
    
    dm_test = np.arange(0.5, 1.5,0.05)
    mu_test = _mu_williams(dm_test)
    DM_will,T_will = np.meshgrid(dm_test,temp_test)
    MU_will,_ = np.meshgrid(mu_test,temp_test)
    
    LUT_will_ddv_x_ka = _sim_ddv_x_ka(DM_will,MU_will,T_will)        
    LUT_will_ddv_ka_w = _sim_ddv_ka_w(DM_will,MU_will,T_will)
    
    ddv_x_ka_all = np.concatenate((LUT_will_ddv_x_ka.ravel(),LUT_ddv_x_ka.ravel(),))
    ddv_ka_w_all = np.concatenate((LUT_will_ddv_ka_w.ravel(),LUT_ddv_ka_w.ravel(),))
    temp_all = np.concatenate((T_will.ravel(),T.ravel(),))
    
    dm_all = np.concatenate((DM_will.ravel(),DM.ravel(),))
    mu_all = np.concatenate((MU_will.ravel(),MU.ravel(),))
    
    scal = 2.5
    data_tree= np.stack((ddv_x_ka_all*scal, ddv_ka_w_all, temp_all), axis = 1)
    my_tree = cKDTree (data_tree)
    return my_tree, dm_all, mu_all, scal

_ddv_tree = _generate_DDV_tree()
    
def dm_mu_rain_first_guess(ddv_ka_w,ddv_x_ka, temp, ddv_tree = _ddv_tree):
    if ddv_tree is None:
        ddv_tree = _generate_DDV_tree()
    my_tree = ddv_tree[0]
    DM = ddv_tree[1]
    MU = ddv_tree[2]
    scal = ddv_tree[3]
    
    data_search = np.stack((ddv_x_ka.ravel()*scal, ddv_ka_w.ravel(), temp.ravel()), axis = 1)    
        
    out = my_tree.query(data_search)
    flag = (out[1]==DM.size)
    out[1][flag]= 0
    DM_ret = DM[out[1].reshape(ddv_x_ka.shape)]
    MU_ret = MU[out[1].reshape(ddv_x_ka.shape)]
    DM_ret[flag.reshape(ddv_x_ka.shape)] = np.nan
    MU_ret[flag.reshape(ddv_x_ka.shape)] = np.nan
    return DM_ret, MU_ret
 
    
def dm_rain_from_3f_dopp(dopp_meas, temp, std_dopp_meas,
                         bands = ['x','ka','w'], step = 30, niter = 10,
                         method = 'CG'):
    
    dopp_shape = dopp_meas[bands[0]].size
    tmp_x = np.arange(dopp_shape)
    init = step//2
    tx = tmp_x[init::step]
    
    spl = splrep(tmp_x, np.zeros_like(tmp_x), task =-1, t = tx)
    len_series = spl[1].size
    exp_series = lambda series: splev(tmp_x, (spl[0],series,spl[2]))
    
    def _vect_to_trip_f_dopp(x,bands =  ['x','ka','w']):
        l = x.size//3
        dopp={}
        for ii, band in enumerate(bands):
            dopp[band] = x[ii*l:(ii+1)*l]
        return dopp
    def _vect_to_vars(x, n_vars=3):
        l = x.size//n_vars
        return [x[ii*l:(ii+1)*l] for ii in range(n_vars)]
    
    def _trip_f_dopp_to_vect(dopp):
        return np.concatenate([dopp[band] for band in list(dopp.keys())])
    
    def _CF_measurements(simul, meas, weight):
        tmp = weight*(meas-simul)**2
        return np.nansum(tmp)/np.sum(~np.isnan(tmp))
    
    def _CF_Williams(dm,mu, std_sigma = 0.058):
        sigm = dm/np.sqrt(mu+4)
        return np.nansum((sigm-0.3)**2)*np.power(std_sigma,-2)/sigm.size
    def _test_CG(x0):
        all_x_i = [x0[0]]
        all_y_i = [x0[1]]
        f = lambda x: _CF_Williams(x[0],x[1])
        def store(X):
            x, y = X
            all_x_i.append(x)
            all_y_i.append(y)
        minimize(f, x0, method="BFGS", callback=store, options={"gtol": 1e-12})
        return all_x_i, all_y_i
    def _plot_CG():
        D_test =np.arange(0.01,4,0.01)
        MU_test = np.arange(-1,18,0.1)
        
        CF_W = np.array([[_CF_Williams(D_test[ii], MU_test[jj]) for ii in range(D_test.size)]
            for jj in range(MU_test.size)])
        plt.figure()
        plt.pcolormesh(D_test, MU_test,np.log10(CF_W))
        plt.colorbar()
        for dm_t,mu_t in zip ([1,2,3],[-1,3,15]):
            all_x_i, all_y_i = _test_CG(np.array([dm_t,mu_t]))
            plt.plot(all_x_i, all_y_i,'-b')
            plt.plot(all_x_i, all_y_i,'or')
    
    def _CF_wind(wind, std_wind = 0.25):
        return np.nansum(wind**2)*np.power(std_wind,-2)/wind.size
    
    def _forward_model(dm, mu, temp, wind, bands = ['x','ka','w']):
        dopp_sim = {}
        for band in bands:
            dopp_sim[band] = simulate.rain['tmat'][band]['dopp']((dm,mu,temp)) + wind
        return dopp_sim
    
    def _x_to_y(x,temp,bands = ['x','ka','w']):
        x_unkn = _vect_to_vars(x)
        dopp_sim =_forward_model(x_unkn[0], x_unkn[1], temp, x_unkn[2], bands)
        return _trip_f_dopp_to_vect(dopp_sim)
    

    weight_level = {}
    for band in bands:        
        weight_level[band] = 1/std_dopp_meas[band]**2
    
    ddv_x_ka = dopp_meas['x'] - dopp_meas['ka']
    ddv_ka_w = dopp_meas['ka'] - dopp_meas['w']
#        plt.figure()
#        plt.plot(ddv_x_ka)
#        plt.plot(ddv_ka_w)        
    
    dm_near, mu_near = dm_mu_rain_first_guess(ddv_ka_w, ddv_x_ka, temp)
    dm_near[np.isnan(dm_near)] = 0.05
    mu_near[np.isnan(mu_near)] = 18.
    
    spl_dm = splrep(tmp_x, dm_near, task =-1, t = tx)
    spl_mu = splrep(tmp_x, mu_near, task =-1, t = tx)
    
#        dm_sm = exp_series(spl_dm[1])
#        mu_sm = exp_series(spl_mu[1]) 
    
    
    
    CF_meas = lambda x: _CF_measurements(_trip_f_dopp_to_vect(
            _forward_model(dm=exp_series(x[:len_series]), 
                           mu=exp_series(x[len_series:2*len_series]),
                           temp=temp, 
                       wind = exp_series(x[2*len_series:3*len_series]))), 
                     _trip_f_dopp_to_vect(dopp_meas),
                     _trip_f_dopp_to_vect(weight_level))
    CF_all = lambda x: (CF_meas(x) +
                    0*_CF_Williams(dm=exp_series(x[:len_series]), 
                           mu=exp_series(x[len_series:2*len_series]))+
                    _CF_wind( exp_series(x[2*len_series:3*len_series])))
    

    x0=np.concatenate((spl_dm[1],spl_mu[1],np.zeros(len_series)))
    
    print(CF_meas(x0))
    res = minimize(CF_all, x0, method = method, options = {'maxiter': niter,
                                                       'disp': True})
    print(CF_meas(res.x))
    return [exp_series(_vect_to_vars(res.x,3)[ii]) for ii in range(3)]
#        res = minimize(CF_x, x0, method = 'Nelder-Mead', options = {'maxfev': 1e4,
#                                                           'disp': True})
    
         
    
#        plt.figure()
#        plt.plot(dm_near)
#        plt.plot(dm_sm)
#        plt.plot(exp_series(res.x[:len_series]))
#        
#        plt.figure()
#        plt.plot(mu_near)
#        plt.plot(mu_sm)
#        plt.plot(exp_series(res.x[len_series:len_series*2]))
#        
#        plt.figure()
#        plt.plot(exp_series(res.x[2*len_series:len_series*3]))
#        plt.figure()
#        plt.plot(res.jac[:len_series])
#        plt.plot(res.jac[len_series:len_series*2])
#        plt.plot(res.jac[len_series*2:len_series*3])
#        
#    
#        
#        dopp_sim = _forward_model(dm=dm_sm, mu=mu_sm, temp=temp_lev, 
#                           wind = np.zeros_like(dm_sm))
#        _CF_measurements(_trip_f_dopp_to_vect(dopp_sim), 
#                         _trip_f_dopp_to_vect(dopp_meas_lev),
#                         _trip_f_dopp_to_vect(weight_level))
#        
#        
#        doppx_sim = simulate.rain['tmat']['x']['dopp']((dm_sm,mu_sm,temp_lev))
#        doppka_sim = simulate.rain['tmat']['ka']['dopp']((dm_sm,mu_sm,temp_lev))
#        doppw_sim = simulate.rain['tmat']['w']['dopp']((dm_sm,mu_sm,temp_lev))
#        
#        plt.figure()
#        plt.plot(doppx_meas)
#        plt.plot(doppx_sim)
#        plt.plot(doppka_meas)
#        plt.plot(doppka_sim)
#        
#        
#        
#        
#        
#    expand_ser = lambda vv: splev(drange, [x_0[0], vv,x_0[2]])
#    
#    form_x = lambda x0 : np.concatenate(x0, axis = -1) 
#    unravel_x = lambda x: [x[ii*x_0[1].size:x_0[1].size*(ii+1)] for ii in range(3)]
#    form_y = form_x
#    unravel_y = lambda y: [y[ii*drange.size:drange.size*(ii+1)] for ii in range(len(bands))] 
#    
#    dm_0 = np.copy(x_0[1])        
#    mu_0 = np.copy(x_0[1])*3
#    wind_0 = np.copy(x_0[1])*0
#    
#    xb = form_x([dm_0,mu_0,wind_0]) 
#    
#    
#    dsize = data['ku']['zku'].shape
#    
#    for xx in range(dsize[0]):
#        temp = -(data['ku']['alt_array'][xx,:].data-4000)*6.5*1e-3 +273.15
#        rain_flag = (temp>276) & (data['ku']['alt_array'][xx,:].data>=0)
#        ind_bott = np.argmin((data['ku']['alt_array'][xx,:])**2)
#        press = simulate._standard_atmos_press(data['ku']['alt_array'][xx,:].data, 
#                    T_b = temp[ind_bott])*1e-2
#                                               
#        dopp_sim = lambda var_list: [simulate.doppler(dm = var_list[0], mu = var_list[1], 
#                    temp = temp, band = band, pressure = press, wind = var_list[2],
#             rain_flag = rain_flag, snow_flag = None) for band in bands]                          
#        y_m = form_y([data['ku']['dopp'+band].data for band in bands])
#        w_y = (y_m >10).astype(float)
#        
#        
#        forward_model = lambda x: form_y(dopp_sim([expand_ser(var) for var in unravel_x(x)]))
#        
#        CF = lambda x: np.nansum(w_y*(forward_model(x)-y_m)**2) + np.nansum((xb-x)**2)
#        out = minimize(CF,xb)
        
        
        