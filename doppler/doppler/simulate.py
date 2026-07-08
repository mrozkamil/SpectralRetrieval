#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:05:22 2019

@author: km357
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light  
from scipy.signal import savgol_filter
from scipy.ndimage import convolve
from netCDF4 import Dataset

main_dir = os.path.dirname(os.path.abspath(__file__))

def dB(val):
    return 10*np.log10(val)
def invdB(val):
    return np.power(10, val/10)
    
# def SpectStd(SNR, spectral_width, M=10, tau_s =4096*1e-4, 
#                   wave_length= 0.031229166):
#     tau_i = wave_length/(4*np.sqrt(np.pi)*spectral_width)
#     M_i = np.minimum(M, M*tau_s/tau_i)
#     return np.sqrt(1/M_i+(1/SNR**2+2/SNR)/M)

def SpectStdDb(SNR, spectral_width, M=10, tau_s =4096*1e-4, 
                  wave_length= 0.031229166):
    tau_i = wave_length/(4*np.sqrt(np.pi)*spectral_width)
    M_i = np.minimum(M, M*tau_s/tau_i)
    return dB(1+np.sqrt(1/M_i+(1/SNR**2+2/SNR)/M))
#    return dB(np.exp(1))*np.sqrt(1/M_i+(1/SNR**2+2/SNR)/M)



def vel_atlas(x, alpha= 1.173, beta = 1.204, gamma = 1676.106):
    return alpha-beta*np.exp(-gamma*x)
    
def rain_terminal_velocity():   
    D_smpl = np.array([0.007826,0.009126,
                        0.02, 0.03, 0.04, 0.05,
                       0.06, 0.07, 0.08, 0.09, 0.1,
                       0.12, 0.14, 0.16, 0.18, 0.2,
                       0.22, 0.24, 0.26, 0.28, 0.3,
                       0.32, 0.34, 0.36, 0.38, 0.4,
                       0.42, 0.44, 0.46, 0.48, 0.5,
                       0.52, 0.54, 0.56, 0.58])*1e1
    V_smpl = np.array([18, 25,
                        72, 117, 162, 206,
                       247, 287, 327, 367, 403,
                       464, 517, 565, 609, 649,
                       690, 727, 757, 782, 806,
                       826, 844, 860, 872, 883,
                       892, 898, 903, 907, 909,
                       912, 914, 916, 917])*1e-2
    
    D_Stocks = np.linspace(0,0.07,8)
    V_Stocks = 30*D_Stocks**2
    
    params = curve_fit(vel_atlas, D_smpl[-15:], V_smpl[-15:],
                           bounds=([9.,9.,0.], [12., 35., 3.]))[0]
    D_ext = np.arange(6,10,0.2)
    V_ext = vel_atlas(D_ext, *params)
    
    DD = np.concatenate((D_Stocks,D_smpl,D_ext))*1e-3
    VV = np.concatenate((V_Stocks,V_smpl,V_ext))
    return IUS(DD,VV), IUS(VV,DD)

def get_am_bm_av_bv(model, elwp =0.):
    if model in ['A','B','C']:        
        ELWP = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
        ind_lwp = np.argmin(np.abs(elwp-ELWP))
        
        a_m = {'A': [0.015, 0.373, 0.249, 0.211, 0.270, 0.576],
               'B': [0.015, 0.0354, 0.0545, 0.126, 0.369, 1.7],
               'C': [469.],}
        
        b_m = {'A': [2.08,  2.44,  2.30,  2.140, 2.080, 2.120],
               'B': [2.08,  2.06,  2.05,  2.03, 2.11, 2.3],
               'C': [3.36],}
        
        a_v = {'A': [2.02 , 6.35 , 5.49 , 6.05 , 8.16 , 12.11],
                'B': [2.02 , 2.42 , 2.92 , 4.49 , 7.52 , 13.07],
                'C': [53.14,]} #
        
        b_v = {'A': [0.14 , 0.28 , 0.23 , 0.21 , 0.23 , 0.28],
                'B': [0.14 , 0.13 , 0.13 , 0.16 , 0.22 , 0.28],
                'C': [0.52]}
        return (a_m[model][ind_lwp], b_m[model][ind_lwp],
                a_v[model][ind_lwp], b_v[model][ind_lwp])
    elif model in ['column', 'dendrite', 'mixcolumndend', 'needle', 'plate']:
        a_m = {'column': 0.074, 'dendrite': 0.027, 'mixcolumndend':0.017,
               'needle': 0.028, 'plate':0.076};
        b_m = {'column': 2.15, 'dendrite': 2.22, 'mixcolumndend':1.95,
               'needle': 2.11, 'plate':2.22}
        
        a_v = {'column': 23.416, 'dendrite': 24.348, 'mixcolumndend':21.739,
               'needle': 17.583, 'plate':30.966}
        
        b_v = {'column': 0.534, 'dendrite': 0.698, 'mixcolumndend':0.580,
               'needle': 0.557, 'plate':0.635};
        return a_m[model], b_m[model], a_v[model], b_v[model]
    
def get_alph_beta_gamma_vel(model, elwp =0.):
    if model in ['A','B','C']: 
        ELWP = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
        ind_lwp = np.argmin(np.abs(elwp-ELWP))
        
        # alpha_m = {'A': [1.14, 2.19, 2.22, 2.61, 3.21, 4.01],
        #        'B': [1.14, 1.44, 1.70, 2.32, 3.06, 4.05],
        #        'C': [4.5],}
        # beta_m = alpha_m
        # gamma_v = {'A': [2976.18 , 948.68 , 1163.36 , 1304.12 , 998.97 , 685.56],
        #         'B': [2976.18 , 2521.74 , 2277.15 , 1686.23 , 1129.30 , 690.99],
        #         'C': [589.07,]} #
                
        alpha_m = {'A': [0.88, 2.16, 2.09, 2.43, 3.06, 3.96],
               'B': [0.88, 1.25, 1.53, 2.29, 3.25, 4.59],
               'C': [6.03],}
        beta_m = alpha_m
        gamma_v = {'A': [1626.17 , 660.76, 936.83 , 1400.49 , 1199.37 , 860.00],
                'B': [1626.17 , 1874.71 , 2144.21 , 1707.05 , 1161.20 , 715.88],
                'C': [443.07,]} #
        
        
        return alpha_m[model][ind_lwp], beta_m[model][ind_lwp], gamma_v[model][ind_lwp]
    
    elif model in ['column', 'dendrite', 'mixcolumndend', 'needle', 'plate']:
        alpha_m = {'column': 1.55, 'dendrite': 0.791, 'mixcolumndend':1.173,
               'needle': 1.061, 'plate':1.299};
        beta_m = {'column': 1.576, 'dendrite': 0.820, 'mixcolumndend':1.204,
               'needle': 1.09, 'plate':1.336}        
        gamma_v = {'column': 1559.496, 'dendrite': 1708.274, 'mixcolumndend':1676.106,
               'needle': 1858.711, 'plate':1421.028}
        
        return alpha_m[model], beta_m[model], gamma_v[model]

rain_vel,rain_vel_inv = rain_terminal_velocity()

def get_vel_unc(model, elwp =0.):
    if model in ['A','B','C']: 
        ELWP = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
        ind_lwp = np.argmin(np.abs(elwp-ELWP))
        std = {'A': [0.08, 0.15, 0.14, 0.15, 0.17, 0.20],
               'B': [0.08, 0.08, 0.10, 0.13, 0.15, 0.17],
               'C': [0.16,],}
        return std[model][ind_lwp]
    elif model in ['column', 'dendrite', 'mixcolumndend', 'needle', 'plate']:
        return 0.08
               

class object():
    pass
    

class spectrum:
    def __init__(self, d_sp, v_sp_funct, z_sp, ext_sp,
    bands = ['w','ka','x'], psd_sampl_diam = None,
    vel_unc = 0.):
        
        freqs = {'s': 2.8, 'c': 5.6, 'x': 9.6, 'ku': 13.6, 'ka': 35.5, 'w': 94., 'g': 200.}
        self.vel_std = vel_unc
        self.d_sp  = {}
        self.refl_sp = {}
        self.vel_sp  = {}  
        self.ext_sp = {}
        self.get_diam = {}
        self.refl_sp_intp = {}
        self.vel_sp_intp = lambda x: v_sp_funct(x)
        self.dD_dV, self.dD_dV_intp = {}, {}
        self.dV_dD, self.dV_dD_intp = {}, {}
        
        self.bands = bands
        self.freqs = [freqs[band] for band in bands]
        
        self.reg_vel = object()
        self.reg_vel.vel_sp = {}
        self.reg_vel.d_sp =  {}
        self.reg_vel.refl_sp = {}
        self.reg_vel.dD_dV = {}
        
        self.dN_dNi_flag = False
        
        for (ff, band) in enumerate(self.bands):
            print('%s band' % band)
            
            self.refl_sp[band] = z_sp[ff]
            self.ext_sp[band] = ext_sp[ff]
            self.d_sp[band] = d_sp[ff]
            self.refl_sp_intp[band] = IUS(self.d_sp[band],self.refl_sp[band],
                                        ext=3)
            self.vel_sp[band] = self.vel_sp_intp(self.d_sp[band])   
            dV = savgol_filter(self.vel_sp[band],window_length = 5,
                                    polyorder =3, deriv = 1)
            dD = savgol_filter(self.d_sp[band],window_length = 5,
                                    polyorder =3, deriv = 1)
            self.dD_dV[band] = dD/dV
            self.dD_dV_intp[band] = IUS(self.d_sp[band],self.dD_dV[band],
                                        ext=3)
            self.dV_dD[band] = dV/dD
            self.dV_dD_intp[band] = IUS(self.d_sp[band],self.dV_dD[band],
                                        ext=3)
            
            self.get_diam[band] =IUS(self.vel_sp[band],self.d_sp[band],
                                        ext=3)
            
            vel_range = self.vel_sp[band].max()-self.vel_sp[band].min()
            or_samp_num = self.vel_sp[band].size
            sampling_points = int((vel_range+4.)/vel_range*or_samp_num)
            
           
            
            self.reg_vel.vel_sp[band] = np.linspace(-2.,
                                self.vel_sp[band].max()+2., sampling_points)
                                
            self.reg_vel.d_sp[band] = np.interp(self.reg_vel.vel_sp[band],
                        self.vel_sp[band], self.d_sp[band], 
                        left = 0, right = 0)
            
            self.reg_vel.refl_sp[band] = self.refl_sp_intp[band](
                                self.reg_vel.d_sp[band])
            self.reg_vel.dD_dV[band] = self.dD_dV_intp[band](
                                self.reg_vel.d_sp[band])
            
            if psd_sampl_diam:
                self.dN_dNi_flag = True
                self.dN_dNi = {}
                self.dN_dNi[band] = np.zeros(self.reg_vel.vel_sp[band].size,
                                        psd_sampl_diam.size)
                for ix in range(psd_sampl_diam.size):
                    d_psd = np.zeros_like(psd_sampl_diam)
                    d_psd[ix] = 1.
                    self.dN_dNi[band][:,ix] = np.interp(self.reg_vel.d_sp[band], 
                                psd_sampl_diam, d_psd,
                                left=0, right=0)
            
        # for band in bands:
        #     
        #     plt.figure()
        #     plt.plot( self.d_sp[band], self.vel_sp[band])
        #     plt.plot( self.reg_vel.d_sp[band], self.reg_vel.vel_sp[band], '--')
        #     
        #     plt.figure()
        #     plt.plot( self.d_sp[band], self.refl_sp[band])
        #     plt.plot( self.reg_vel.d_sp[band], self.reg_vel.refl_sp[band], '--')
        #     
        #     plt.figure()
        #     plt.plot( self.d_sp[band], self.dD_dV[band])
        #     plt.plot( self.reg_vel.d_sp[band], self.reg_vel.dD_dV[band], '--')
        #     
        #     plt.show()
            
        
                    

    def _gauss(self,x,mu,s):
        return 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*s**2))
    def _dgauss_ds(self,x,mu,s):
        return self._gauss(x,mu,s)*((x-mu)**2 - s**2)/s**3
            
    def __call__(self, Vq, band, D, psd, turb = 0.0, wind = 0.0,
                 integ = 'slow', pressure = 1e3,
                 plot = False,  derivatives = False, calc_all_der = True):
    
        
        if not np.isclose(pressure, 1e3):
            corr_fact = np.power(pressure*1e-3,0.4)
        else:
            corr_fact = 1.
            
        tot_turb = np.sqrt(turb**2+self.vel_std**2)
            
        if integ == 'prec':
            psd_i = np.interp(self.d_sp[band], D, psd, left = 0, right = 0)
            d_eq = np.copy(self.d_sp[band])
            refl = np.copy(self.refl_sp[band])
            spec_turb_free_diam = psd_i*refl
            vel_rain = np.copy(self.vel_sp[band])
            vel_rain *= 1/corr_fact
        
            if derivatives:
                dS_dN = np.zeros((Vq.size, psd.size))
                dS_dturb = np.zeros((Vq.size, 1))
                dS_dwind = np.zeros((Vq.size, 1))
                d_spec_turb_free = refl
                
            if tot_turb>1e-2:
                spec = np.zeros_like(Vq)
                for (indv,v) in enumerate(Vq-wind):                 
                    ker = self._gauss(vel_rain,v,tot_turb)  
                    spec[indv] = np.trapz(spec_turb_free_diam*ker, d_eq)                 
                if derivatives:
                    dker = self._dgauss_ds(vel_rain,v,tot_turb)
    
                    dS_dturb[indv,0] = np.trapz(spec_turb_free_diam*dker, d_eq)
                    ker_dwind = self._gauss(vel_rain,v-0.01,tot_turb)
                    spec_dwind = np.trapz(spec_turb_free_diam*ker_dwind, d_eq)
                    dS_dwind[indv,0] = 1e2*(spec_dwind-spec[indv])
                    if calc_all_der:
                        for ix in range(psd.size):
                            d_psd = np.zeros_like(psd)
                            d_psd[ix] = 1.
                            tmp_der = np.interp(d_eq, D, d_psd, left=0, right=0)
                            tmp_der = d_spec_turb_free*tmp_der
                            dS_dN[indv,ix] = np.trapz(tmp_der*ker, d_eq)
            else:
                spec = np.interp(Vq, vel_rain+wind, 
                       spec_turb_free_diam*corr_fact*self.dD_dV[band],
                       left=0, right=0)    
                # weight = np.trapz(ker, vel_rain)               
                        
        elif integ == 'slow':
            
            psd_i = np.interp(self.reg_vel.d_sp[band], D, psd, 
                        left = 0, right = 0)
                                
            refl = np.copy(self.reg_vel.refl_sp[band])
            dD_dV = np.copy(self.reg_vel.dD_dV[band])
            vel_rain = np.copy(self.reg_vel.vel_sp[band])
            dD_dV *= corr_fact
            vel_rain *= 1/corr_fact
            spec_turb_free = psd_i*refl*dD_dV
            
            if tot_turb>1e-2:
                ker = self._gauss(vel_rain[np.abs(vel_rain)<=4*tot_turb], 0, tot_turb)
                weight = np.sum(ker)
                ker /= weight
            else:
                ker = np.ones(1)
                weight = 1.
            
            spec_prec = convolve(spec_turb_free,ker, mode = 'nearest')
            spec = np.interp(Vq, vel_rain+wind, spec_prec,left=0, right=0)
            
                
            if derivatives:
                dS_dN = np.zeros((Vq.size, psd.size))
                dS_dturb = np.zeros((Vq.size, 1))
                dS_dwind = np.zeros((Vq.size, 1))
                d_spec_turb_free = refl*dD_dV
                if calc_all_der:
                
                    if self.dN_dNi_flag:
                        
                        tmp_der = d_spec_turb_free[np.newaxis,:]*self.dN_dNi[band]
                        dspec_prec = convolve(tmp_der,ker[np.newaxis,:], 
                                    mode = 'nearest')
                                    
                        f_out = interp1d(vel_rain+wind, dspec_prec, axis=1,
                                fill_value = 0.)
                        dS_dN = f_out(Vq)
                        
                    else:
                        for ix in range(psd.size):
                            d_psd = np.zeros_like(psd)
                            d_psd[ix] = 1.
                            tmp_der = np.interp(self.reg_vel.d_sp[band], 
                                        D, d_psd,left=0, right=0)
                            tmp_der = d_spec_turb_free*tmp_der 
                            dspec_prec = convolve(tmp_der,ker, mode = 'nearest')
                            dS_dN[:,ix] = np.interp(Vq, vel_rain+wind, dspec_prec,
                                                left=0, right=0)
                if tot_turb>1e-2:
                    dker = self._dgauss_ds(vel_rain[np.abs(vel_rain)<4*tot_turb], 0,
                                tot_turb)/weight
                    dspec_prec = convolve(spec_turb_free, dker, mode = 'nearest')
                    dS_dturb[:,0] = np.interp(Vq, 
                            vel_rain+wind, dspec_prec, 
                            left=0, right=0)
                
                spec_dwind = np.interp(Vq, vel_rain+wind+0.01, 
                                    spec_prec,left=0, right=0)
                dS_dwind[:,0] = 1e2*(spec_dwind-spec)
                
        if plot:
            plt.figure()
            plt.plot(vel_rain, dB(spec_turb_free))
            # if integ == 'slow':
            #     plt.plot(V_prec, dB(spec_interp))
            #     plt.plot(V_prec, dB(spec_prec))
            plt.plot(Vq, dB(spec),'x')
            plt.ylim(-50,20) 
            
        if derivatives:
            return spec, dS_dN, dS_dwind, dS_dturb
        else:
            return spec
    def calc_Z(self,band, D, psd):
        psd_i = np.interp(self.d_sp[band], D, psd, left = 0, right = 0)
        refl = np.copy(self.refl_sp[band])
        Z = np.trapz(psd_i*refl,self.d_sp[band])
        return Z
    def calc_ext(self,band, D, psd):
        psd_i = np.interp(self.d_sp[band], D, psd, left = 0, right = 0)
        refl = np.copy(self.ext_sp[band])
        Z = np.trapz(psd_i*refl,self.d_sp[band])
        return Z
        
    def calc_MDV(self,band, D, psd):
        psd_i = np.interp(self.d_sp[band], D, psd, left = 0, right = 0)
        refl = np.copy(self.refl_sp[band])
        vel_rain = np.copy(self.vel_sp[band])
        MDV = np.trapz(psd_i*refl*vel_rain,self.d_sp[band]
                )/np.trapz(psd_i*refl,self.d_sp[band])
        return MDV
    


class spectrum_rain(spectrum):      
    def _read_scattering_file(self,freq, T, data_dir ):           
        if isinstance(freq, float):
            freq_str = ('%.2f' % freq).replace('.','p')
        if isinstance(freq, int):
            freq_str = ('%d' % freq)
        f_name = 'water_tmatrix_t%d_%sGHz_wobbly.dat' % (T,freq_str)
        print('Reading: %s' % f_name)
        return np.loadtxt(os.path.join(data_dir,f_name), comments='%')
    def _wave_length(self,freq = 94):
        return speed_of_light/freq*1e-9

    def __init__(self, bands = ['w','ka','x'], T=278):
        self.K2 = {'s': 0.93, 'c': 0.93,'x': 0.928,'ku': 0.925, 'ka': 0.895,  'w': 0.765, 'g': 0.594} 
        freqs = {'s': 2.8, 'c': 5.6, 'x': 9.6, 'ku': 13.6, 'k': 24., 'ka': 35.5, 'w': 94., 'g': 200.}
        K2_lf = self.K2['s']
        
        scat_dir = os.path.join(main_dir, 'LUT', 'rain')
        
        z_sp, d_sp, ext_sp = [], [], []
        for band in bands:
            freq = freqs[band]
#            print('Frequency of %1.1f GHz' % freq)
            wl = self._wave_length(freq = freq)            
            cons = wl**4/(np.pi**5*K2_lf)*1e12;
            scat_tmp = self._read_scattering_file(freq = freq, T=T, 
                                                  data_dir = scat_dir)
            
            z_sp.append( scat_tmp[:,4]*cons)
            ext_sp.append(scat_tmp[:,2]*1e-6)
            d_sp.append( scat_tmp[:,0]*2*1e-3)
            v_sp_funct,_ = rain_terminal_velocity() 
        
        super().__init__(d_sp, v_sp_funct, z_sp, ext_sp, bands = bands, vel_unc = 0.)
        
class spectrum_snow(spectrum):      
    def _read_sps(self,elwp , model, data_path , vel_fit_type):
        if model in ['A','B']:
            single_particle_file = os.path.join(data_path,
                        'SSRGA_LUT_Leinonen_sps_new.nc')
            ELWP = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
            ind_lwp = np.argmin(np.abs(elwp-ELWP))
            mod = 'Lein_%s_%d' % (model, ind_lwp+1)
        if model == 'C':
            single_particle_file = os.path.join(data_path,
                        'SSRGA_LUT_Leinonen_sps_new.nc')
            mod = 'Lein_%s' % (model, )
            
        elif model in ['column', 'dendrite', 'mixcolumndend', 'needle', 'plate']:
            single_particle_file = os.path.join(data_path,
                       'SSRGA_LUT_Leonie.nc')
            mod_num = {'column': 1,'dendrite': 2, 'mixcolumndend':3,
                       'needle': 4, 'plate':5}
            mod = 'Leon_%d' % mod_num[model]
            
        mass_vel_par = get_am_bm_av_bv(model, elwp)  
        atlas_par = get_alph_beta_gamma_vel(model, elwp )
        dictionary = {'a_m': mass_vel_par[0],  'b_m': mass_vel_par[1],
                      'a_v': mass_vel_par[2],  'b_v': mass_vel_par[3],
                     'alpha_v': atlas_par[0],  'beta_v': atlas_par[1],
                     'gamma_v': atlas_par[2],}
        
        data= Dataset(single_particle_file)
        
        variables = ['Dlong', 'mass','s_backsca', 's_sca',
                's_ext', 'z_back',]
        
        dtmp = np.squeeze(data.variables['Dlong_'+mod][:].filled(np.nan))
        ind_valid = np.isfinite(dtmp)
        
        for var in variables[:]:
#            print(var)
            var_name = '%s_%s' % (var,  mod)
            dictionary[var] = np.squeeze(
                    data.variables[var_name][ind_valid].filled(np.nan))
          
        dictionary['Rad_freq'] =  np.squeeze(
                data.variables['Rad_freq'][:].filled(np.nan))
        seperator = ''
        dictionary['MODEL'] = seperator.join([l[0].decode('ascii') for l in 
                   data.variables['MODEL_'+mod][:].filled()])
        data.close()
        dictionary['D_eq'] = 0.1*np.cbrt(dictionary['mass']*6/np.pi)
        
        if vel_fit_type == 'power':
            dictionary['vel'] = dictionary['a_v'] * np.power(dictionary['Dlong'],
                              dictionary['b_v'])
        elif vel_fit_type == 'atlas':
            dictionary['vel'] = vel_atlas(dictionary['D_eq'], 
                      dictionary['alpha_v'], dictionary['beta_v'],  
                      dictionary['gamma_v'])
        
        return dictionary
     
    def _wave_length(self,freq = 94):
        return speed_of_light/freq*1e-9

    def __init__(self, bands = ['w','ka','x'], 
                 elwp = 0., model = 'B', vel_fit_type = 'power'):
        self.K2 = {'s': 0.93, 'c': 0.93,'x': 0.93,'ku': 0.93, 'ka': 0.93, 
                   'w': 0.93, } 
        freqs = {'s': 2.8, 'c': 5.6, 'x': 9.6, 'ku': 13.6, 'ka': 35.5, 'w': 94, 'g':200}
        K2_lf = 0.93
        scat_dir = os.path.join(main_dir, 'LUT', 'snow')
        scat_tmp = self._read_sps( elwp , model , scat_dir, vel_fit_type)        
        z_sp, d_sp, ext_sp = [], [], []
        for band in bands:
            freq = freqs[band]
            wl = self._wave_length(freq = freq)
            cons = wl**4/(np.pi**5*K2_lf)*1e18;
#            print('Frequency of %1.1f GHz' % freq)
            i_fr = np.argmin(np.abs(scat_tmp['Rad_freq'][:]-freq))
            
            tmp_z = np.copy(scat_tmp['s_backsca'][:,i_fr])*cons
            ind_nan = np.isnan(tmp_z)
            tmp_z[np.where(ind_nan)] = np.interp(scat_tmp['D_eq'][ind_nan],
                      scat_tmp['D_eq'][~ind_nan], tmp_z[~ind_nan])  
            z_sp.append( tmp_z)
            
            tmp_z = np.copy(scat_tmp['s_ext'][:,i_fr])
            ind_nan = np.isnan(tmp_z)
            tmp_z[np.where(ind_nan)] = np.interp(scat_tmp['D_eq'][ind_nan],
                      scat_tmp['D_eq'][~ind_nan], tmp_z[~ind_nan])  
            ext_sp.append( tmp_z)
            
            d_sp.append( scat_tmp['D_eq'])
            
        v_sp_funct = IUS(scat_tmp['D_eq'], scat_tmp['vel'])
        vel_std = get_vel_unc(model, elwp = elwp)
        print('model: %s; lwp: %.1f kg/m^2' % (model, elwp))
        super().__init__(d_sp, v_sp_funct, z_sp, ext_sp, bands = bands,vel_unc = vel_std)
        self.model = model
        self.elwp = elwp
        self.dmax_sp = scat_tmp['Dlong']
