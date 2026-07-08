#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:20:34 2020

@author: km357
"""
import numpy as np
from scipy.optimize import minimize, Bounds
import os, sys
home = os.path.expanduser('~')
sys.path.append(os.path.join(home, 'Documents', 'Python3', 'my_modules', 'tripex'))
import tripex
sys.path.append(os.path.join(home, 'Documents', 'Python3', 'my_modules', 'doppler'))
import doppler

class spectra_simulator():
    def __init__(self, spec_sim, D_nodes= None, k = 2, ref_band = 'ka',
                 max_Deq = 5e-3, max_Deq_bin_size = 1e-3,
                 min_vel_res = 0.2, vel_bins = np.linspace(-1,10,111),
                 turb_vect = np.arange(0,.5,0.025)): 
        self.doppler_bins = vel_bins              
        self.spec_fun = spec_sim
        
        self.bands = list(self.spec_fun.d_sp.keys())
        band = ref_band
               
        # D = self.spec_fun.d_sp[band]   
        D = np.linspace(0,max_Deq, 501)[1:]        
        up_lim = D[-1]
        if D_nodes is None:    
            V = self.spec_fun.vel_sp_intp(D[-1])
            V_tmp = np.arange(0,V, min_vel_res)[1:]
            D_tmp = self.spec_fun.get_diam[band](V_tmp)                
            tmp_list = [t for t in np.arange(-1,1e-6,0.1)*1e-3]
            for tmp_node in D_tmp:
                if (tmp_node - tmp_list[-1]<max_Deq_bin_size) & (
                        tmp_node<up_lim):                    
                    tmp_list.append(tmp_node)
            while tmp_list[-1]<up_lim:
                tmp_list.append(tmp_list[-1]+max_Deq_bin_size)
                    
            D_nodes_hyd = np.array(tmp_list)
        else:
            D_nodes_hyd = D_nodes
            
        self.psd_gen = psd_gen(
            D_nodes = D_nodes_hyd, D = D, k = k,)
            # funct = (lambda x: np.power(x*1e3,-6)) )
            # funct = (lambda x: 8e3*np.exp(x*4e3)) ) 
        print('PSD expressed using %d basis functions' % (
             self.psd_gen.psd_basis_size))
        # print(self.psd_gen.psd_basis.shape)
        # print(self.psd_gen.psd_peaks)
        self.psd_gen.PR_basis = np.zeros(self.psd_gen.psd_basis_size)
        self.psd_gen.Dm_basis = np.zeros(self.psd_gen.psd_basis_size)
        self.psd_gen.WC_basis = np.zeros(self.psd_gen.psd_basis_size)
        self.psd_gen.Dm_x_WC_basis = np.zeros(self.psd_gen.psd_basis_size)
        vel = self.spec_fun.vel_sp_intp(self.psd_gen.D) 
        for ii in range(self.psd_gen.psd_basis_size):  
            psd = self.psd_gen.psd_basis[:,ii]
            wc = doppler.psd.WC(psd=psd,D= self.psd_gen.D)
            rr = doppler.psd.RR(psd=psd,D= self.psd_gen.D,vel = vel)
            dm = doppler.psd.Dm(psd=psd,D= self.psd_gen.D)
            self.psd_gen.WC_basis[ii] = wc
            self.psd_gen.PR_basis[ii] = rr
            self.psd_gen.Dm_basis[ii] = dm
            self.psd_gen.Dm_x_WC_basis[ii] = dm*wc
        self.turb_vect = turb_vect        
        self.spec_sim = self._spec_sim(psd_gen = self.psd_gen,
                                               spec_fun = self.spec_fun,
                                               turb_vect = self.turb_vect,
                                               vel_bins = self.doppler_bins)
        self.Dm_vect = np.arange(0.1,3.5,0.1)*1e-3
        mat_size = (self.doppler_bins.size, self.Dm_vect.size, 
                    self.turb_vect.size)
        
        self.DSR_basis = {}
        self.DFR_val = {}
        self.N0_0dBZ = {}
        for band1, band2 in zip(['x','ka'],['ka','w']):            
            dfr_str = '%s_%s' % (band1, band2)

            self.DSR_basis[dfr_str] = np.zeros(mat_size)
            self.DFR_val[dfr_str] = np.zeros(mat_size[1:])
            self.N0_0dBZ[band1] = np.zeros(mat_size[1:])
            
            for indd,dd in enumerate(self.Dm_vect):
                for tt, turb in enumerate(self.turb_vect):
                    dsr,dfr, n0 = self.get_DSR (Dm = dd, turb_q = turb,
                        band1 = band1, band2 = band2, )
                    
                    self.DSR_basis[dfr_str][:,indd,tt] = dsr
                    self.DFR_val[dfr_str][indd,tt] = dfr
                    self.N0_0dBZ[band1][indd,tt] = n0
                    
    def get_PR(self, psd_c):
        return np.dot(psd_c, self.psd_gen.PR_basis)
    
    
    def get_exp_psd(self,Dm = 1e-3, N= 8e6, refined = True):
        psd_q = N*np.exp(-4*self.psd_gen.D/Dm)
        psd_q[self.psd_gen.D>2.5*Dm] = 0
        psd_c = self.psd_gen.get_basis_rep(psd_q, refined = refined)
        psd_c[psd_c<1e-6] = 0
        return psd_c
        
    def get_DSR (self, Dm = 1e-3, turb_q = 0., band1 = 'ka', band2 = 'w', ):
        psd_c = self.get_exp_psd(Dm = Dm, N= 8e6)
        spec_1 = self.spec_sim(psd_c,band = band1, turb_q = turb_q)                        
        spec_2 = self.spec_sim(psd_c,band = band2, turb_q = turb_q)
        # print(spec_1.shape)
        # print(self.doppler_bins.shape)
        z1 = np.trapz(spec_1,x = self.doppler_bins)
        z2 = np.trapz(spec_2,x = self.doppler_bins)
        dwr = 10*np.log10(z1/z2)
                                                   
        nan_flag  = ((spec_1<1e-4) | (spec_2<1e-4))                                            
        dsr  = 10*np.log10(spec_1/spec_2)
        dsr[nan_flag] = np.nan
        return dsr, dwr, 8e6/z1
    
    def get_lowest_detectable_vel(self,  Dm = 1e-3, PR = 0.1,  
                                  band = 'ka', turb_q = 0., thres = -30):
        psd_c = self.get_exp_psd(Dm = Dm, N= 8e6)       
        rr = np.dot(self.psd_gen.PR_basis,psd_c)
        scale_fact =  PR/rr        
        spec = self.spec_sim(scale_fact*psd_c,band = band, 
                                 turb_q = turb_q)
        
        return np.min(self.spec_sim.vel[spec>10**(thres/10)])
           
    class _spec_sim():
        def __init__(self, psd_gen, spec_fun, vel_bins = np.linspace(-1,10,111),
                     turb_vect = np.arange(0,.5,0.025), ):
                        
            self.turb_vect = turb_vect
            self.Tikhonov = {0: np.eye(psd_gen.psd_basis_size)}
            for oo in range(1,5):
                self.Tikhonov[oo] =  tripex.utils.reg_matrix(
                             n = psd_gen.psd_basis_size, diff = oo) 
                
            self.spec_basis = {}
            self.ext_basis = {}
            self.refl_basis = {}
            self.vel = vel_bins
            bands = spec_fun.d_sp.keys()
            for band in bands: 
                print('generating spectral basis at the %s-band' % band)
                self.spec_basis[band]= np.zeros((self.vel.size, 
                                            psd_gen.psd_basis_size,
                                            self.turb_vect.size))
                self.ext_basis[band]= np.zeros((psd_gen.psd_basis_size,))
                self.refl_basis[band]= np.zeros((psd_gen.psd_basis_size,))
                # low_freq_K2_corr = 0.93/spec_fun.K2[band]                
                for ii in range(psd_gen.psd_basis_size):
                    self.ext_basis[band][ii] = spec_fun.calc_ext(band, 
                                D = psd_gen.D, psd= psd_gen.psd_basis[:,ii])
                    self.refl_basis[band][ii] = spec_fun.calc_Z(band, 
                                D = psd_gen.D, psd= psd_gen.psd_basis[:,ii])
                    for tt, turb in enumerate(self.turb_vect):
                        self.spec_basis[band][:,ii, tt] = (
                                spec_fun(Vq= self.vel,
                                band= band, D = psd_gen.D,  
                                psd = psd_gen.psd_basis[:,ii], 
                                turb = turb, wind = 0.0,  integ = 'prec')) 
                # self.spec_basis[band] *= low_freq_K2_corr
                                  
        def __call__(self, psd_c, band = 'ka', turb_q=0,  ):
            tt = np.argmin(np.abs(self.turb_vect-turb_q))
            bs_array = self.spec_basis[band][:,:,tt]
            return np.matmul(bs_array, psd_c,)   
        
        def get_ext(self,  psd_c, band = 'ka',):
            bs_array = self.ext_basis[band]
            return np.dot(bs_array, psd_c,)
        def get_refl(self,  psd_c, band = 'ka',):
            bs_array = self.refl_basis[band]
            return np.dot(bs_array, psd_c,)
        
        def get_basis_rep(self, ys, weights = None, turb_q = [0,],
                          bands = ['ka',], reg_par = 1e-6,  ):
            
            X_list = []
            for turb, band in zip(turb_q, bands):
                tt = np.argmin(np.abs(self.turb_vect-turb))
                X_list.append (self.spec_basis[band][:,:,tt]) 
                
            X = np.concatenate(X_list, axis = 0)
            y = np.concatenate(ys, axis =0)
            if weights is not None:                
                W = np.diag(np.concatenate(weights, axis =0))                
            else:
                W = np.eye(X.shape[0])
            
            proj_mat = np.real(tripex.utils.funct_2_proj_matrix(
                    basis_functions=X,  weights = W,  
                        tikhonov_norm = reg_par, eps = 1e-12, ))            
            return np.matmul(proj_mat, y), X
        

class psd_gen():
    def __init__(self,D_nodes, D = np.linspace(0,5, 501)[1:], 
                 k = 1, inner_only = True, funct = None):
        
        (psd_basis,self.psd_peaks, self.psd_basis_size, 
                 self.D) = tripex.utils.spline_mat(
            nodes = D_nodes, x = D, k = k, inner_only = inner_only)   
        if funct is None:
            self.psd_basis = psd_basis
        else:
            self.psd_basis = funct(self.psd_peaks)[np.newaxis,:]*psd_basis
        self.proj_matrix = tripex.utils.funct_2_proj_matrix(self.psd_basis)
 
    def __call__(self,node_values):
        return np.matmul(self.psd_basis, node_values, )
        
    def get_basis_rep(self, psd, refined = True):
        a_0 = np.matmul(self.proj_matrix, psd, )
        if refined:            
            def CF(a):            
                y_s = np.matmul(self.psd_basis, a)
                dy = (y_s-psd)
                CF_o = np.dot(dy,dy)       
                return CF_o 
        
            def CF_grad(a):
                y_s = np.matmul(self.psd_basis, a)
                dy = (y_s-psd)
                CF_o = 2*np.matmul(self.psd_basis.T,dy)
                return CF_o
            lb = 1e-6*np.ones(a_0.shape)
            ub= 1e15*np.ones(a_0.shape)
            db = ub-lb
            a_0 = np.minimum(np.maximum(a_0,lb+0.01*db), ub-0.01*db)
            bounds = Bounds(lb,ub)            
            
            a_best = minimize(CF, x0 = a_0, jac = CF_grad, method = 'L-BFGS-B',             
                              options = {'maxiter':1500,},  bounds = bounds,)
            return a_best.x
        else:
            return a_0



         