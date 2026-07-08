import os, sys
home = os.path.expanduser('~')
sys.path.append('/data/doppler/CodeShare/doppler')
import doppler
sys.path.append('/data/doppler/CodeShare/tripex')
import tripex
from tripex.utils import dB, invdB, refractive

# import itur
# import itur.models.itu676 as itu676
from scipy.integrate import cumtrapz
import sys
import numpy as np
np.seterr(all='ignore')
import xarray as xr
import matplotlib.pyplot as plt
import numdifftools as nd
from glob import glob

from scipy.constants import speed_of_light
from scipy.signal import savgol_filter
from scipy.optimize import minimize, Bounds
from scipy.ndimage import uniform_filter

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# from matplotlib.colors import LogNorm
unit_t = np.timedelta64(1000,'ms')

# =============================================================================
test_flag = True

method = 'L-BFGS-B'
method = 'SLSQP'
# =============================================================================

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

retrievals_path = '/data/doppler/CodeShare/tripex/scripts'
sys.path.append(retrievals_path)
import retrievals


freqs = {'L': 0.915, 'S': 2.8, 'C': 5.6, 'X': 9.6,
         'Ku': 13.6, 'K': 24.,'Ka': 35.5, 'W': 94., 'G': 220.}

    
def map_between_bounds(x,lb,ub,min_dist = 0.):
    dlu = ub-lb
    return np.minimum(np.maximum(x,lb+min_dist*dlu),ub-min_dist*dlu)

def form_x_OE (dsd, pia_ka, pia_w, turb, wind):
    tmp_arr = np.array([pia_ka, pia_w, turb, wind])
    return np.concatenate([dsd, tmp_arr])
    
def form_y_OE (obs_struc):
    return np.concatenate([obs_struc[band] for band in bands])

def separate_x_OE(x_oe):
    dsd = x_oe[:-4]
    pia_ka, pia_w, turb, wind = x_oe[-4:]
    return dsd, pia_ka, pia_w, turb, wind


def CF_obs(y_sim, y_obs, var_inv_obs):
    delta_y = y_sim - y_obs
    return np.sum(var_inv_obs*(delta_y**2))
    
    
def CF_cont(dsd):
    return np.sum(np.diff(dsd,2)**2)

def runningVar(mat, size):
    mat_sq_mean = uniform_filter(mat**2, size = size)
    mean_mat_sq = uniform_filter(mat, size = size)**2
    return mat_sq_mean-mean_mat_sq

def simulate_spectra(psd_c,spectra_sim, pia_ka = 0., pia_w =0., 
                     turb=0, wind=0, dv = 0.041282654):
    ind_roll = np.round(wind/dv).astype(int) 
    spec, spec_lin = {},{}
    for band in bands:
        spec_lin[band] = tripex.utils.roll_pad(
            spectra_sim.spec_sim(psd_c, band = band, turb_q = turb),
            shift = ind_roll)               
    for band in ['ka','w', 'x']:
        spec[band] = dB(spec_lin[band])
    spec['ka'] += - pia_ka
    spec['w'] += - pia_w              
    for band in bands:
        flag_low = (~np.isfinite(spec[band]) | (spec[band]<-60 ))
        spec[band][flag_low] = -99.
    return spec, spec_lin

def CF_gamma(x, spectra_sim, x_b , var_inv_b , y_obs , var_inv_obs, 
            dv = 0.041282654, return_CFo = False, plot_dfr = False):
                        
    dm, pia_ka, pia_w, turb, wind = x[:]
   
    psd_lin_t = doppler.psd.gamma_psd_met(DD,WC=1e-3, Dm=dm*1e-3 , mu=0)
    psd_lin_t[DD>2.5*dm] = 0.
    psd_c = spectra_sim.psd_gen.get_basis_rep(psd_lin_t,refined = False)
    psd_c[psd_c<1e-9] = 1e-9  

    spec,_=simulate_spectra(psd_c,spectra_sim, pia_ka = pia_ka, pia_w =pia_w, 
                     turb=turb, wind=wind, dv = dv)
    
    dfr_tmp ={}
    for band1 in bands[1:]:
        for band2 in bands[:-1]:
            if band1==band2: continue
            tmp_str = '%s_%s' % (band1,band2)     
            dfr_tmp [tmp_str] = spec[band1] - spec[band2]  
    if plot_dfr:
        for ic,dfr_str in enumerate(['x_ka','x_w','ka_w']):
            plt.plot(dopp_bins['ka'], dfr_tmp[dfr_str],'--',
                     color = 'C%d' % ic, label = tmp_str)
    
    y_sim = np.concatenate([dfr_tmp[dfr_str] for dfr_str in 
                       ['x_ka','x_w','ka_w']])     
    
    CF_o = np.sum(0.5*var_inv_obs*(y_obs-y_sim)**2)
    CF_b = np.sum(0.5*var_inv_b*(x-x_b)**2)
    CF_pia = 0.5*1*(pia_ka*3-pia_w)**2
    if return_CFo:
        return CF_o+CF_b, CF_o/np.sum(var_inv_obs)
    else:
        return CF_o+CF_b+CF_pia


def CF_gamma_refl(x, spectra_sim, x_b , var_inv_b , y_obs , var_inv_obs, 
            dv = 0.041282654, refl_obs = None, refl_var_inv = None, 
            return_CFo = False, plot_dfr = False):
                        
    db_N0,dm, pia_ka, pia_w, turb, wind = x[:]
    N0 = invdB(db_N0)
    psd_c = spectra_sim.get_exp_psd(Dm = dm*1e-3, N = N0,
                                    refined = False)
    psd_c[psd_c<1e-9] = 1e-9  

    spec,_=simulate_spectra(psd_c,spectra_sim, pia_ka = pia_ka, pia_w =pia_w, 
                     turb=turb, wind=wind, dv = dv)
    
    dfr_tmp ={}
    for band1 in bands[1:]:
        for band2 in bands[:-1]:
            if band1==band2: continue
            tmp_str = '%s_%s' % (band1,band2)     
            dfr_tmp [tmp_str] = spec[band1] - spec[band2]  
    if plot_dfr:
        for ic,dfr_str in enumerate(['x_ka','x_w','ka_w']):
            plt.plot(dopp_bins['ka'], dfr_tmp[dfr_str],'--',
                     color = 'C%d' % ic, label = tmp_str)
    
    y_sim = np.concatenate([dfr_tmp[dfr_str] for dfr_str in 
                       ['x_ka','x_w','ka_w']])     
    
    CF_o = np.sum(0.5*var_inv_obs*(y_obs-y_sim)**2)
    CF_b = np.sum(0.5*var_inv_b*(x-x_b)**2)
    
    if refl_obs is not None:
        pia_dict = {'x':0.,'ka':pia_ka, 'w':pia_w}
        Ze_sim ={band: dB(spectra_sim.spec_sim.get_refl(psd_c, band = band)) 
                 for band in bands}
        Z_sim = {band: Ze_sim[band]-pia_dict[band] for band in bands}
        refl_sim = np.array([Z_sim[band] for band in bands]) 
        dz = refl_sim-refl_obs
        CF_z = 0.5*np.dot(dz**2, refl_var_inv)
        CF_o += CF_z
    
    if return_CFo:
        return CF_o+CF_b, CF_o/np.sum(var_inv_obs)
    else:
        return CF_o+CF_b
    
def CF_spline(x, spectra_sim, y_obs, var_inv_obs, x_b, var_inv_b, 
              PIA_ka=None, PIA_w=None, Turb=None, Wind=None,
              refl_obs = None, refl_var_inv = None, dv = 0.041282654,
              return_comp = False):
    
    if PIA_ka is None:  pia_ka = x[-4] 
    else: pia_ka = PIA_ka*1.
    if PIA_w is None:  pia_w = x[-3] 
    else: pia_w = PIA_w*1.
    if Turb is None:  turb = x[-2] 
    else: turb = Turb*1.
    if Wind is None:  wind = x[-1] 
    else: wind = Wind*1.  
   
    psd_db = x[:-4]*1.
    psd_c = invdB(psd_db) 
    
    spec,_=simulate_spectra(psd_c,spectra_sim, pia_ka = pia_ka, pia_w =pia_w, 
                     turb=turb, wind=wind, dv = dv)   
    
    y_sim = np.concatenate([spec[band] for band in bands])            
    dy = y_obs-y_sim
    CF_meas = 0.5*np.dot(dy**2, var_inv_obs)
    CF_back =  0.5*np.sum(var_inv_b*(x_b-x)**2)
    T_mat = spectra_sim.spec_sim.Tikhonov[2]*.025
    CF_cont = 0.5*np.dot(np.matmul(T_mat,psd_db),psd_db)
    CF_z = 0.
    if refl_obs is not None:
        pia_dict = {'x':0.,'ka':pia_ka, 'w':pia_w}
        Ze_sim ={band: dB(spectra_sim.spec_sim.get_refl(psd_c, band = band)) 
                 for band in bands}
        Z_sim = {band: Ze_sim[band]-pia_dict[band] for band in bands}
        refl_sim = np.array([Z_sim[band] for band in bands]) 
        dz = refl_sim-refl_obs
        CF_z = 0.5*np.dot(dz**2, refl_var_inv)
       
    CF_all = CF_meas+CF_back+CF_cont+ CF_z  
    # print('CF_o: %.1f; CF_b: %.1f; CF_c: %.1f;' % (CF_meas, CF_back, CF_cont )) 
    if return_comp:
        return CF_meas, CF_back, CF_cont, CF_z
    else:
        return CF_all


def CF_spline_grad(x, spectra_sim, y_obs, var_inv_obs, x_b, var_inv_b, 
              PIA_ka=None, PIA_w=None, Turb=None, Wind=None,
              refl_obs = None, refl_var_inv = None, dv = 0.041282654,
              return_comp = False):   
    
    if PIA_ka is None:  pia_ka = x[-4] 
    else: pia_ka = PIA_ka*1.
    if PIA_w is None:  pia_w = x[-3] 
    else: pia_w = PIA_w*1.
    if Turb is None:  turb = x[-2] 
    else: turb = Turb*1.
    if Wind is None:  wind = x[-1] 
    else: wind = Wind*1.  
   
    psd_db = x[:-4]*1.
    psd_c = invdB(psd_db) 
    
    ind_roll = np.round(wind/dv).astype(int)
    ind_turb = np.argmin(np.abs(spectra_sim.spec_sim.turb_vect-turb))
    
    spec,spec_lin=simulate_spectra(psd_c,spectra_sim, pia_ka = pia_ka, pia_w =pia_w, 
                     turb=turb, wind=wind, dv = dv)   
    
    y_sim = np.concatenate([spec[band] for band in bands])    
    y_sim_lin = np.concatenate([spec_lin[band] for band in bands])    
    y_sim_lin[y_sim_lin<1e-12] = 1e-12
    dy = y_obs-y_sim
    CF_meas = 0.5*np.dot(dy**2, var_inv_obs)
    mat0 = np.concatenate([tripex.utils.roll_pad(
        spectra_sim.spec_sim.spec_basis[band][:,:,ind_turb],
        shift = ind_roll, axis = 0) for band in bands])
    
   
    mat1 = mat0/y_sim_lin[:,np.newaxis]            
    mat1[~np.isfinite(mat1)] = 0.  
    
    dCF_meas_dpsd = -np.matmul(mat1.T,dy*var_inv_obs)*psd_c    
    
    if Turb is None:
        specp,_=simulate_spectra(psd_c,spectra_sim, pia_ka = pia_ka, pia_w =pia_w, 
                         turb=turb+0.01, wind=wind, dv = dv)             
        yp_sim = np.concatenate([specp[band] for band in bands]) 
        dyp = y_obs-yp_sim            
        CF_meas_p = 0.5*np.dot(dyp**2, var_inv_obs)            
        dCF_meas_dturb = (CF_meas_p-CF_meas)/0.01
    else:
        dCF_meas_dturb = 0.
        
    if Wind is None:
        yp_sim = np.concatenate([tripex.utils.roll_pad(spec[band],1)
                                 for band in bands])           
        dyp = y_obs-yp_sim            
        CF_meas_p = 0.5*np.dot(dyp**2, var_inv_obs)            
        dCF_meas_dwind = (CF_meas_p-CF_meas)/dv
    else:
        dCF_meas_dwind = 0.
        
    if PIA_ka is None:
        d_vect = np.concatenate([int(band=='ka')*np.ones(spec[band].size)
                                     for band in bands])
        dCF_meas_dpiaka = np.dot(d_vect,var_inv_obs*dy)
    else:
        dCF_meas_dpiaka = 0.
    if PIA_w is None:
        d_vect = np.concatenate([int(band=='w')*np.ones(spec[band].size)
                                 for band in bands])
        dCF_meas_dpiaw = np.dot(d_vect,var_inv_obs*dy)
    else:
        dCF_meas_dpiaw =0.
        
    dCFz_dpia_ka = 0.
    dCFz_dpia_w = 0.
    if refl_obs is not None:
        pia_dict = {'x':0.,'ka':pia_ka, 'w':pia_w}
        Ze_lin_sim ={band: spectra_sim.spec_sim.get_refl(psd_c, band = band) 
                 for band in bands}
        Ze_sim ={band: dB(Ze_lin_sim[band])  for band in bands}
        Z_sim = {band: Ze_sim[band]-pia_dict[band] for band in bands}
        refl_sim = np.array([Z_sim[band] for band in bands]) 
        dz = refl_sim-refl_obs
                      
        dCFz_dPSD = sum([spectra_sim.spec_sim.refl_basis[band]*psd_c/
                        Ze_lin_sim[band]*refl_var_inv[ind_band]*
                        dz[ind_band] for (ind_band, band) in enumerate(bands)])
        if PIA_ka is None:
            dCFz_dpia_ka = -dz[1]*refl_var_inv[1]
        if PIA_w is None:
            dCFz_dpia_w = -dz[0]*refl_var_inv[0]
        
    else:
        dCFz_dPSD = 0.        
                                 
    CF_o_grad = np.concatenate([dCF_meas_dpsd+dCFz_dPSD, np.array([
        dCF_meas_dpiaka+dCFz_dpia_ka,
        dCF_meas_dpiaw+dCFz_dpia_w,
        dCF_meas_dturb,dCF_meas_dwind])])
    
    CF_back_grad =  (var_inv_b*(x-x_b))            
    T_mat = spectra_sim.spec_sim.Tikhonov[2]*.025
    
    CF_cont_grad = np.concatenate(
        [np.matmul(T_mat,psd_db), np.zeros(4)])
   
    CF_all = CF_o_grad+CF_back_grad+CF_cont_grad
    if return_comp:
        return CF_o_grad, CF_back_grad, CF_cont_grad
    else:
        return CF_all

    
    
# =============================================================================
# load data
# =============================================================================

date_str = '20181124'
range_shift_x = 17.
dataPath = os.path.join('/data', 'doppler' ,'TRIPEx','LV0', date_str)

retrPath = os.path.join('/data', 'doppler','TRIPEx','LV0','Retr_V4', date_str, 'rain', method)
if not os.path.exists(retrPath):
    os.makedirs(retrPath)
    
figPath = os.path.join(retrPath,'figs')
if not os.path.exists(figPath):
    os.makedirs(figPath)
    
fileNames = {}
fileNames['w'] = glob(os.path.join(dataPath +'/wBand', '%s_*_P09_ZEN.LV0.nc' % date_str[2:]))
fileNames['x'] = glob(os.path.join(dataPath+'/xBand', 
        '%s_[0-9][0-9][0-9][0-9][0-9][0-9].znc' % date_str))
fileNames['ka'] = glob(os.path.join(dataPath+'/kaBand', 
        '%s_[0-9][0-9][0-9][0-9].znc' % date_str))

bands = ['w','ka','x']
spData = {}
for band in bands:
    fileNames[band].sort()
    print('loading %s-band data' % band)
    spData[band] = tripex.io.loadData(fileNames[band], precise_time=True)
spData['x'] = spData['x'].assign_coords({"range": spData['x'].range + range_shift_x })


for ii in range(4):
    spData['w'] = spData['w'].assign_coords({'C%dVel' % (ii+1,) : np.linspace(
            spData['w'].MaxVel.values[ii], -spData['w'].MaxVel.values[ii],   
            spData['w'].DoppLen.values[ii]+1)[:-1]})



# =============================================================================
# extracting radar parameters
# =============================================================================

dt = {band: np.median(np.diff(spData[band].time.values)) for band in bands}
dv, dh, height, dopp_bins = {}, {}, {}, {}
for band in ['x', 'ka']:  
    height[band] = spData[band].range.values
    dh[band] = np.diff(height[band][:2])[0]
    dopp_bins[band] = tripex.io.getSortedVel(spData[band])
    dv[band] = np.diff(dopp_bins[band][:2])[0]
    
band = 'w'
ind_h={}
count = 0
for ch in range(4):
    w_ch = '%s%d' % (band,ch)      
    dopp_bins[w_ch] = tripex.io.getSortedVel(spData[band], ch = ch)    
    dv[w_ch] = np.diff(dopp_bins[w_ch][:2])[0]
    height[w_ch] = spData[band]['C%dRange' % (ch+1)].values
    dh[w_ch] = np.diff(height[w_ch][:2])[0]
    ind_h[w_ch] = np.arange(count,count+height[w_ch].size)
    count += height[w_ch].size
    
    
# height['w'] = np.concatenate([height['w%d' % ch] for ch in range(4)])
dopp_bins['ka_x'] = np.copy(dopp_bins['ka'])
dopp_bins['w'] = dopp_bins['w0']
height['w'] = height['w0']
dv['w'] = dv['w0']

radar_spec = {'w':{}, 'ka':{}, 'x':{}}
radar_spec['w']= {'PRF': spData['w'].AvgNum.values/spData['w'].SeqIntTime.values,
                    'aver_cycles': spData['w'].AvgNum.values/spData['w'].DoppLen.values,
                    'DoppLen': spData['w'].DoppLen.values,
                    'wave_length': speed_of_light/spData['w'].Freq.values*1e-9}
for (band,Data) in zip(['ka','x'], [spData['ka'], spData['x']]):
    radar_spec[band]= {'PRF': np.expand_dims(Data.prf.values,0),
                    'aver_cycles': np.expand_dims(Data.nave.values,0),
                    'DoppLen': np.expand_dims(Data.nfft.values,0),
                    'wave_length': Data['lambda'].values.item(0)}



# =============================================================================
# load diff PIA estimate at the cloud top
# =============================================================================

diff_pia_path = os.path.join(retrievals_path, 'diffPIA', date_str)
h5_fn = os.path.join(diff_pia_path, 'diff_pia_%s.h5' % date_str)
diff_pia = xr.open_dataset(h5_fn, decode_times = True)
tmp_time = (diff_pia.time.values*np.timedelta64(1000,'ms') + 
                            np.datetime64(
                            "%04d-%02d-%02dT%02d:%02d:%02d" %
                            (*diff_pia.ref_time,)) )
diff_pia = diff_pia.rename({'phony_dim_0': 'time'})
diff_pia = diff_pia.assign_coords({"time": tmp_time })


   
# =============================================================================
# load the lowest detected velocity data
# =============================================================================
ds_vel = {}
for band in ['x','ka', 'w']:
    ds_vel[band] = xr.open_dataset(
        os.path.join(dataPath, 'minimum_detected_vel_%s_%s.nc' % 
                             (band, date_str)))
    
ds_vel['x'] = ds_vel['x'].assign_coords({"range": ds_vel['x'].range + 
                                         range_shift_x*1e-3 })
ds_vel['w'] = ds_vel['w'].assign_coords({"time": ds_vel['w'].time + 
                                         np.timedelta64(2,'s') })

# =============================================================================
# load the estimate of the BB altitude
# =============================================================================

dataPathLv1 = os.path.join('/data', 'doppler','TRIPEx','LV1', date_str)    
f_name_melt_data= os.path.join(dataPathLv1,
                               'melt_level_and_time_lag_20181124.nc')
ds_melt = xr.open_dataset(f_name_melt_data)


# =============================================================================
# setting up spectra data velocity range filters
# ============================================================================

spectra_limit = {}
lim_vels = ['min_vel','max_vel']

for band in bands:
    spectra_limit[band]={}
    for key in lim_vels:
        tmp_arr = np.copy(ds_vel[band][key].values)
        tmp_arr[~np.isfinite(tmp_arr)] = -99.
        spectra_limit[band][key] = tripex.interp.interp2( 
                    ds_vel[band].time.values[:], 
                    ds_vel[band].range.values[:], tmp_arr,
                    spData['ka'].time.values[:], 
                    spData['ka'].range.values[:]*1e-3) 
        flag_bad = (spectra_limit[band][key]<-15.)
        spectra_limit[band][key][flag_bad] = np.nan
        
max_vel_x = {}
for band in bands:
    key = 'max_vel'
    tmp_arr = np.copy(ds_vel['x'][key].values)
    tmp_arr[~np.isfinite(tmp_arr)] = -99.
    max_vel_x[band] = tripex.interp.interp2( 
                ds_vel['x'].time.values[:], 
                ds_vel['x'].range.values[:], tmp_arr,
                spData[band].time.values[:], 
                height[band]*1e-3) 
    flag_bad = (max_vel_x[band]<-15.)
    max_vel_x[band][flag_bad] = np.nan

# =============================================================================
# load the eastimate of the broadening 
# =============================================================================
broad = {band: xr.open_dataset (os.path.join(dataPathLv1,
                    'broadening_%s_20181124.nc' % (band,))) for band in bands}

# =============================================================================
# load environmental data
# =============================================================================
lidar_data = xr.open_dataset (os.path.join(dataPathLv1,
                                      '20181124_juelich_categorize.nc' ))

# # =============================================================================
# # Specific attenuation estimate due to the gasses
# # =============================================================================
# P = lidar_data.pressure.values*itur.u.Pascal
# T = lidar_data.temperature.values*itur.u.Kelvin
# SH = lidar_data.specific_humidity.values


# rho_air = tripex.thermodynamics.vapour_density(T=T, P=P,q = SH)

# pia_gas_est = xr.Dataset(coords= {'time': ("time",spData['ka'].time.values),
#                         'height': ("height",  height['ka'])} )

# # specific_att = xr.Dataset({band: (["time","height"], np.zeros(P.shape), 
# #                                   {'unit':'dB/km'}) for band in bands},
# #                           coords= {'time': ("time", lidar_data.time.values),
# #                         'height': ("height", lidar_data.model_height.values)} )
    
# print('specific attenuation computations')
# for band in bands:
#     print(band)
#     fs = tripex.utils.Ksq.freq[band.capitalize()]*itur.u.GHz 
#     one_wat_atts = itu676.gamma_exact(fs, P=P, rho=rho_air, T=T)
#     specific_att = one_wat_atts.value  

#     pia_gasses = 2*cumtrapz(specific_att,lidar_data.model_height.values*1e-3,
#                     initial = 0) + (
#         np.expand_dims((lidar_data.model_height.values[0]-111.)*
#             specific_att[:,0]*1e-3, axis = 1))
        
#     pia_gas_est[band] = (('time', 'height'), 
#                 tripex.interp.interp2(lidar_data.time.values, 
#                 lidar_data.model_height.values-111., pia_gasses,
#                 spData['ka'].time.values, height['ka']))
    



# =============================================================================
# spectra generators
# =============================================================================
spec_gen = doppler.simulate.spectrum_rain(bands = ['w','ka','x'])
spectra_sim = tripex.spectra_simulator.spectra_simulator(
                  spec_sim = spec_gen, vel_bins = dopp_bins['ka'],
                  turb_vect = np.arange(0,.51,0.01),min_vel_res = 0.25, k=1,
                  max_Deq_bin_size = 5e-4)

spectra_sim_coarse = tripex.spectra_simulator.spectra_simulator(
                  spec_sim = spec_gen, vel_bins = dopp_bins['ka'],
                  turb_vect = np.arange(0,.51,0.01),min_vel_res = 0.5, k=1 )

spectra_sim_very_coarse = tripex.spectra_simulator.spectra_simulator(
                  spec_sim = spec_gen, vel_bins = dopp_bins['ka'],
                  turb_vect = np.arange(0,.51,0.01),min_vel_res = 1., k=1 )



# plt.figure()
# plt.plot(spec_gen.d_sp['w']*1e3, spec_gen.vel_sp['w'],'-k', lw = 2)
# plt.grid()
# plt.xlim(0,6)
# plt.xlabel('D [mm]')
# plt.ylabel('V [m/s]')
# plt.title('Raindrop terminal velocity')
# png_name = os.path.join('/data', 'doppler', 'TRIPEx','LV1','20181124','psd_figs',
#                         'rain_terminal_vel.png')
# plt.tight_layout()
# plt.savefig(png_name)

# plt.figure()
# plt.plot(spec_gen.vel_sp['x'], dB(spec_gen.refl_sp['x']), lw = 2)
# plt.plot(spec_gen.vel_sp['x'], dB(spec_gen.refl_sp['ka']), lw = 2)
# plt.plot(spec_gen.vel_sp['x'], dB(spec_gen.refl_sp['w']), lw = 2)
# plt.grid()
# plt.xlim(0,9)
# plt.ylim(-50,50)
# plt.xlabel('V [m/s]')
# plt.ylabel('Z [dBZ]')
# plt.title('Raindrop radar reflectivity')
# png_name = os.path.join('/data', 'doppler', 'TRIPEx','LV1','20181124','psd_figs',
#                         'rain_refl.png')
# plt.tight_layout()
# plt.savefig(png_name)


DD = spectra_sim.psd_gen.D
VV = doppler.simulate.rain_vel(DD)
# plt.figure()
# for tt,turb in enumerate([0.01, 0.2, 0.4]):
#     line_style = ['-','--',':', '-.'][tt]
#     for dm in np.arange(0.5, 3, 2)*1e-3:
#         for mu  in [0,]:
#             unity_psd = doppler.psd.gamma_psd_met(DD,Dm = dm, mu = mu)            
#             dm_tmp = doppler.psd.Dm(unity_psd,DD)           
#             DSR = (spec_gen(dopp_bins['ka'], band = 'x', D=DD,psd = unity_psd,
#                         turb = turb,integ = 'prec')/
#                     spec_gen(dopp_bins['ka'], band = 'w', D=DD,psd = unity_psd, 
#                         turb = turb,integ = 'prec'))
#             plt.plot(dopp_bins['ka'],dB(DSR),  line_style, label = dm*1e3)
            
#             psd_c = spectra_sim.psd_gen.get_basis_rep(unity_psd)
#             DSR_2 = (spectra_sim.spec_sim(psd_c, band = 'x', turb_q = turb)/
#                      spectra_sim.spec_sim(psd_c, band = 'w', turb_q = turb))
#             plt.plot(dopp_bins['ka'],dB(DSR_2),  '-', lw = 1,)
            
# plt.grid()   
# plt.legend()         
# plt.show(block = False)




start_times = np.arange(np.datetime64('2018-11-24T06:00:00'),
                    np.datetime64('2018-11-25T00:00:00'), np.timedelta64(10,'m'))

##### initial parameters
make_plots = True
if test_flag:
    time_ind = 7  
    start_t = np.datetime64('2018-11-24T07:00:00')
    end_t = np.datetime64('2018-11-24T08:00:00')
    ind_comp = np.where((spData['ka'].time>start_t) &
                        (spData['ka'].time<=end_t))[0][::300]
else:
    cpu_ind = np.int(sys.argv[1])-1   
    time_ind = cpu_ind
    start_t = start_times[time_ind]
    end_t = start_times[time_ind+1]
    ind_comp = np.where((spData['ka'].time>start_t) & 
                        (spData['ka'].time<=end_t))[0]



##### retrieval grid
alt_indeces = np.where(spData['ka'].range<1200)[0]
ind_low_range = np.where(spData['ka']['range'].values<1500)[0]

Range = spData['ka']['range'].values[ind_low_range]
Vel = dopp_bins['ka']


rain_correction = {}
for band in bands:
    Ksq_band = refractive.refractive_index(freqs[band.capitalize()],
                                           temp_C = 5.)[1]    
    rain_correction[band] = -dB(Ksq_band/0.93)
    
# retrieved vars
aver_vars_list = ['Dm', 'Sm', 'WC', 'wind', 'turb', 'RR', 'N0_fg',
                    'Dm_fg', 'Sm_fg', 'WC_fg', 'wind_fg', 'turb_fg',
                'Zm_x', 'Zs_x', 'Zm_ka', 'Zs_ka', 'Zm_w', 'Zs_w', 
                'Vs_x', 'Vs_ka', 'Vs_w', 'Vm_x', 'Vm_ka','Vm_w',
                'pia_ka', 'pia_w',  'pia_ka_fg', 'pia_w_fg',
                'CF_meas','range', 'RR_std', 'Dm_std', 'Dm_fg_std',
                'N0_fg_std', 'Ka_saturated','retr_converged',
                'wind_std', 'turb_std', 'pia_ka_std', 'pia_w_std']

retr_units = {'Dm': 'm', 'Sm': 'm', 'WC': 'kg/m^3', 
                'wind': 'm/s', 'turb': 'm/s', 'RR': 'mm/h',
            'Dm_fg': 'm', 'Sm_fg': 'm', 'WC_fg': 'kg/m^3',
             'wind_fg': 'm/s', 'turb_fg': 'm/s',
             'Zm_x': 'dBZ', 'Zs_x': 'dBZ', 'Zm_ka':'dBZ', 
             'Zs_ka': 'dBZ', 'Zm_w': 'dBZ', 'Zs_w': 'dBZ',
             'Vs_x': 'm/s', 'Vs_ka': 'm/s', 'Vs_w': 'm/s', 
             'Vm_x': 'm/s', 'Vm_ka': 'm/s','Vm_w': 'm/s',
             'pia_ka': 'dB', 'pia_w': 'dB',  
            'pia_ka_fg': 'dB', 'pia_w_fg': 'dB',
            'N0_fg': 'dB', 'N0_fg_std': 'dB', 'Dm_fg_std':'m',
             'CF_meas': 'dB','range': 'm', 'psd': '10log_10(m^-4)',
             'psd_std': '10log_10(m^-4)','RR_std':'mm/h', 'Dm_std':'m',
             'Ka_saturated' : 'Boolean','retr_converged': 'Boolean',
             'wind_std': 'm/s', 'turb_std': 'm/s', 
             'pia_ka_std': 'dB', 'pia_w_std': 'dB'}

f_name= os.path.join(retrPath,'psd_retr_lev_%s.nc' % np.datetime_as_string(start_t))
attrs_t = {}
attrs_r = {'units': 'm'}
retr = xr.Dataset(coords = {'time': ('time', spData['ka'].time.values[ind_comp],attrs_t),
                    'diameter': ('diameter', DD, attrs_r)})

for var in aver_vars_list:
    retr[var] = (('time'), np.full((ind_comp.size,), np.nan), 
                 {'units': retr_units[var]})
retr['PSD'] = (('time','diameter'), 
               np.full((ind_comp.size,  DD.size, ), np.nan), 
               {'units': 'dB(m^-4)'})
retr['PSD_std'] = (('time','diameter'), 
                   np.full((ind_comp.size,  DD.size, ), np.nan), 
                   {'units': 'dB(m^-4)'})
  
psd_bs_size = spectra_sim.psd_gen.psd_basis_size


f_name2= os.path.join(retrPath,'psd_retr_basis_%s.nc' % np.datetime_as_string(start_t))
psd_basis_dset = xr.Dataset(coords = {'time': ('time', spData['ka'].time.values[ind_comp],attrs_t),
                    'diameter': ('diameter', DD, attrs_r),
                    'psd_nodes_x': ('psd_nodes_x', spectra_sim.psd_gen.psd_peaks, attrs_r),
                    'psd_nodes_y': ('psd_nodes_y', spectra_sim.psd_gen.psd_peaks, attrs_r),})
psd_basis_dset['PSD_basis'] = (('diameter','psd_nodes_x'), 
               spectra_sim.psd_gen.psd_basis, 
               {'units': 'm^-4'})
psd_basis_dset['PSD_expansion_coef'] = (('time','psd_nodes_x',), 
               np.full((ind_comp.size, psd_bs_size,), np.nan), 
               {'units': 'dB'})
psd_basis_dset['CF_Hessian_inv'] = (('time','psd_nodes_x','psd_nodes_y'), 
               np.full((ind_comp.size, psd_bs_size,psd_bs_size), np.nan), 
               {'units': ''})


# np.full((ind_comp.size, psd_bs_size,psd_bs_size), np.nan)
# retr['PSD_basis'] = spectra_sim.psd_gen.psd_basis  

comp = dict(zlib=False, complevel=0)
encoding = {var: comp for var in retr.data_vars}
encoding['time'] ={'units':'seconds since 1970-01-01'}

encoding2 = {var: comp for var in psd_basis_dset.data_vars}
encoding2['time'] ={'units':'seconds since 1970-01-01'}


##### plot dir
test_dir = os.path.join(dataPathLv1, 'psd_figs')
if not os.path.exists(test_dir): os.mkdir(test_dir)


time_plot = np.datetime64('2018-11-24T08:58:19')
# time_plot = np.datetime64('2018-11-24T07:01:34')

CF_prev = 100.
x_sp_oe_prev = np.array([])


for it, ti in enumerate(spData['ka'].time.values[ind_comp]):
    
    
    # if np.abs(ti-time_plot)/np.timedelta64(1,'s')>1:
    #     print(np.abs(ti-time_plot)/np.timedelta64(1,'s'))
    #     continue
    
    
    time_string = np.datetime_as_string(ti,'s')
    print(time_string)
    hour_str = time_string[-8:-6]
    figPathHour = os.path.join(figPath,hour_str)
    
    
    if not os.path.exists(figPathHour):
        os.makedirs(figPathHour)
        
    
    
    try:
        
        #restricting to one time only
        meltDataSingT_rain = ds_melt.sel({'time': ti}, 'nearest')
        
        alt = meltDataSingT_rain.melt_bot.values -50
        retr['range'][it,] = alt
        
        
        broadSingT = broad['ka'].sel({'time': ti}, 'nearest')
        broad_lev = broadSingT.total_broadening.interp(
                    {'range': alt}).values.item()
        
        #reading pressure data
        pressureSingT = lidar_data['pressure'].sel({'time': ti}, 'nearest')
        press_lev = pressureSingT.interp(
                    model_height=alt+111.,).values.item()*1e-2
        
        ind_alt ={}
        for band in bands:
            ind_alt[band] =  np.argmin(np.abs(height[band]  - alt))
        ind_alt['ref'] =  np.argmin(np.abs(Range  - alt))
        
        
        # =====================================================================
        #         reading the spectral data (full column)
        # =====================================================================
        snr, measSpec,ind_sel_time, sel_time = {}, {}, {}, {}
        for band, ch in zip(['x','ka', 'w'], ['o','o',0]): 
            ind_sel_time[band] = np.argmin(np.abs(spData[band].time.values-ti)/
                                           np.timedelta64(1,'s'))
            radar_spec[band]['spectra_num'] = ind_sel_time[band].size
            sel_time[band] = spData[band].time.values[ind_sel_time[band]]        
            measSpec[band],snr[band], _  = tripex.io.getSelectedSpec(
                spData[band], ch = ch, sel_time=sel_time[band],)
            
                           
        measSpec['ka_x'], snr['ka_x'], _ = tripex.io.getSelectedSpec(
                spData['ka'], ch = 'x', sel_time=sel_time['ka'],)
        
        measSpec['ka_x'] *= invdB(3.5)
        measSpec['ka'] *= invdB(3.5)
        measSpec['w'] *= invdB(-2.2)
        
        
        
        
        # plt.figure()
        # for band,col in zip(bands,['C0','C1','C2']):
        #     plt.plot(dopp_bins[band],dB(measSpec[band][0,ind_alt[band],:]))
        #     plt.vlines(
        #         spectra_limit[band]['max_vel'][ind_sel_time['ka'], ind_alt['ka']],
        #             -30,30,color= col)
        #     plt.vlines(
        #         spectra_limit[band]['min_vel'][ind_sel_time['ka'], ind_alt['ka']],
        #             -30,30, color = col)
        # plt.plot(dopp_bins['ka'],dB(measSpec['ka_x'][0,ind_alt['ka'],:]), '--')
        # plt.show()
        
        
        
        # =====================================================================
        #         computing moments of the spectra
        # =====================================================================
        Z_lin, mdv, spec_width = {}, {}, {}
        
        for band in bands:
            xx = np.copy(dopp_bins[band])
            dopp_spec = np.copy(measSpec[band])
            dopp_spec[dopp_spec<0] = 0
            tmp_z = doppler.psd.moment(dopp_spec,xx, moment =0, 
                    center =0, integration_axis = 2)
            tmp_mdv= doppler.psd.moment(dopp_spec,xx, moment =1, 
                    center =0, integration_axis = 2)/tmp_z
            tmp_sw = doppler.psd.moment(dopp_spec,xx, moment =2, 
                    center = tmp_mdv, integration_axis = 2)/tmp_z
            
            Z_lin[band] = np.squeeze(tmp_z)
            mdv[band] = np.squeeze(tmp_mdv)
            spec_width[band] = np.squeeze(tmp_sw)
                    
        # plt.figure()
        # for band in bands:
        #     plt.plot(dB(Z_lin[band]),height[band]*1e-3, label = band)
            
        #     plt.plot(dB(Z_lin[band][ind_alt[band]]), 
        #         height[band][ind_alt[band]]*1e-3, 'x')
        # plt.grid()
        # # plt.plot(dB(spDataWindow[band].Zg.mean('time')),
        # #         height[band]*1e-3,  '--', label = 'Zx integ')
        # plt.legend()
        # plt.show(block = False)
            
        
        # =====================================================================
        #         restricting the spectral data to one level
        # =====================================================================
            
        snr_lev, spec_lev, snr_lev_dB, spec_lev_dB = {}, {}, {}, {}
        # spec_lev_var, spec_lev_dB_sm =  {}, {}
        spec_lev_var =  {}
        min_vel, max_vel = {}, {}
        vel_width = 0.25
        for band in bands: 
            
            spec_lev[band] = measSpec[band][0,ind_alt[band],:]*1.
            tmp_v = dB(spec_lev[band])
            tmp_v[~np.isfinite(tmp_v)] = -99.
            tmp_v[tmp_v<-99.] = -99.
            spec_lev_dB[band] = tmp_v
            N_vel = (np.round(vel_width/dv[band])*2).astype(int)+1            
            sp_var = runningVar(spec_lev_dB[band], size = N_vel)
            
            snr_lev[band] = snr[band][0,ind_alt[band],:]*1.
            snr_lev[band][snr_lev[band]<0.] = 0
            snr_tmp = snr_lev[band]*1.
            snr_tmp[(snr_tmp<1e-12) | ~np.isfinite(snr_tmp)] = 1e-12
            snr_lev_dB[band] = dB(snr_lev[band])
            sw = spec_width[band][ind_alt[band]]
            if sw<0.1 or ~np.isfinite(sw): sw = 0.1
           
            std_z_integ = doppler.simulate.SpectStdDb(
                    SNR =  snr_tmp, spectral_width = sw,
                    M =(radar_spec[band]['aver_cycles'][0]*
                    radar_spec[band]['spectra_num']), 
                    tau_s = (radar_spec[band]['DoppLen'][0]/
                    radar_spec[band]['PRF'][0]), 
                    wave_length= radar_spec[band]['wave_length'])                       
            
            spec_lev_var[band] = std_z_integ**2 + sp_var 
            
            # tmp_weigh = 1/spec_lev_var[band]
            # tmp_weigh[~np.isfinite(tmp_weigh)] = 0.
            
            # w_tmp = uniform_filter(tmp_weigh, size = N_vel)           
            # spec_lev_dB_sm[band] =  uniform_filter(spec_lev_dB[band]*tmp_weigh, 
            #                               size =N_vel)/w_tmp            
            
            min_vel[band], max_vel[band] = (
                doppler.retrieve.particle_vel_SNR(snr_lev_dB[band], 
                dopp_bins[band], axis = 0, thres = 3, size = N_vel ))

        
        plt.figure()
        for (ic,band) in enumerate(bands): 
            plt.plot(dopp_bins[band], spec_lev_dB[band], '--',lw = 2, 
                      color = 'C%d' % ic, label = band)
        plt.xlim(-2, 10)
        plt.ylim(-45,30)
        plt.grid()
        plt.title(time_string)       
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Spectral Power Density [dBZ/m/s]')        
        plt.tight_layout(0.5)
        # =====================================================================
        #         bad data filtering
        # =====================================================================
        band = 'x'
        sm_sp_z = savgol_filter(spec_lev_dB[band], window_length=5,
                     polyorder=2, deriv= 0, delta = dv[band])
        d_sp = savgol_filter(spec_lev_dB[band], window_length=5,
                     polyorder=2, deriv= 1, delta = dv[band])
        sp_flat = np.where(np.abs(d_sp)<1e-12)[0]        
        sp_below_out_w = np.where((dopp_bins[band]<min_vel['w']-0.25) | (
                         dopp_bins[band]>max_vel['w']+0.25))[0]          
        ind_bad = np.intersect1d(sp_flat, sp_below_out_w)
        
        spec_lev_var[band][ind_bad] = 99.**2
        spec_lev_dB[band][ind_bad] = -99.
        snr_lev_dB[band][ind_bad] = -99.
        N_vel = (np.round(vel_width/dv[band])*2).astype(int)+1 
        min_vel[band], max_vel[band] = (
                doppler.retrieve.particle_vel_SNR(snr_lev_dB[band], 
                dopp_bins[band], axis = 0, thres = 3, size = N_vel ))
                      
        
# =============================================================================
#         data overflowing
# =============================================================================
        
        Z_band_min = np.min(spec_lev_dB['x'][spec_lev_dB['x']>-35])
        Z_band_min_right = np.min(spec_lev_dB['x'][(spec_lev_dB['x']>-35) & 
                                            (dopp_bins['x']>max_vel['x']-0.5)])
        Z_band_min_left = np.min(spec_lev_dB['x'][(spec_lev_dB['x']>-35) & 
                                            (dopp_bins['x']<min_vel['x']+0.5)])
        Z_band_max = np.max(spec_lev_dB['x'])
        
             
        for band in ['ka','w']:
                      
            flag_bad = ((dopp_bins[band]<=min_vel['x']-0.2) & 
                        (spec_lev_dB[band]>Z_band_min_left)) | ( 
                            (dopp_bins[band]<=min_vel['x']) & 
                        (spec_lev_dB[band]>Z_band_min_left+6.))
            flag_bad = flag_bad | ((dopp_bins[band]>=max_vel['x']+0.2) & 
                                  (spec_lev_dB[band]>Z_band_min_right)) | (
                                (dopp_bins[band]>=max_vel['x']) & 
                                (spec_lev_dB[band]>Z_band_min_right+6.))
            if (band =='w') & (Z_band_max>15):
                vel_limit = np.nanmax([7., max_vel['x']-1.5])
                flag_bad = flag_bad | (
                    (dopp_bins[band]>vel_limit) & (snr_lev_dB[band]<9))  
            if (band =='ka'):
                rad = 0.1
                ind_test_vel = np.where(
                    (np.abs(dopp_bins[band]-(min_vel['x']-rad))<rad) |
                   (np.abs(dopp_bins[band]- (max_vel['x']+rad))<rad))[0]
                Z_noise_db = dB(np.mean(invdB(spec_lev_dB[band][ind_test_vel])))
                
                ka_overflows = (Z_noise_db > Z_band_min+3) & (Z_band_max>20)
                if ka_overflows:
                    flag_bad = flag_bad | (spec_lev_dB[band]<Z_noise_db+6) 
                    if (Z_noise_db > Z_band_min+9):
                        spec_lev_var[band] *= 10 
            
            N_vel = (np.round(vel_width/dv[band])*2).astype(int)+1   
            d_sp = savgol_filter(spec_lev_dB[band], window_length=N_vel,
                     polyorder=2, deriv= 1, delta = dv[band])
            
            spec_lev_var[band][flag_bad] = 99.**2
            spec_lev_dB[band][flag_bad] = -99.
            snr_lev_dB[band][flag_bad] = -99.    
            
            min_vel[band], max_vel[band] = (
                doppler.retrieve.particle_vel_SNR(snr_lev_dB[band], 
                dopp_bins[band], axis = 0, thres = 3, size = N_vel ))
            
            min_vel[band] = np.nanmax([ min_vel[band],  min_vel['x']-0.5])  
            max_vel[band] = np.nanmin([ max_vel[band],  max_vel['x']+0.5])  
    
            iv = np.argmin(np.abs(dopp_bins[band]-min_vel[band]))
            cond = (iv>0)
            while cond:
                cond = ((d_sp[iv-1]>7.5) & (snr_lev_dB[band][iv-1]>0.) &
                    (dopp_bins[band][iv-1]>min_vel[band]-0.5)) & (iv>0)
                iv += -1
            min_vel[band] =  dopp_bins[band][iv]
            
            iv = np.argmin(np.abs(dopp_bins[band]-max_vel[band]))
            cond = (iv<dopp_bins[band].size-1)
            while cond:
                cond = ((d_sp[iv+1]<-7.5) & (snr_lev_dB[band][iv+1]>0.) &
                    (dopp_bins[band][iv+1]>max_vel[band]-0.5) &
                    (iv<dopp_bins[band].size-1))
                iv += 1
            max_vel[band] =  dopp_bins[band][iv]
                               
                
            # plt.plot(dopp_bins[band][flag_bad],spec_lev_dB[band][flag_bad],'o')
                  
            
        for band in bands:
            ind_bad = np.where((dopp_bins[band]<min_vel[band]) | (
                          dopp_bins[band]>max_vel[band]))[0]       
            
            spec_lev_var[band][ind_bad] = 99.**2
            spec_lev_dB[band][ind_bad] = -99.
            snr_lev_dB[band][ind_bad] = -99.
            
           
        
        for (ic,band) in enumerate(bands):                      
                        
            plt.plot(dopp_bins[band], spec_lev_dB[band], '-',lw = 2, 
                      color = 'C%d' % ic, label = band)
            plt.fill_between(dopp_bins[band], 
                        spec_lev_dB[band] - np.sqrt(spec_lev_var[band]),
                        spec_lev_dB[band] + np.sqrt(spec_lev_var[band]),
                                color = 'C%d' % ic, alpha = 0.3)       
            plt.vlines(min_vel[band], -45,30, color = 'C%d' % ic, lw =3)
            plt.vlines(max_vel[band], -45,30, color = 'C%d' % ic, lw =3)
            
        
        
        # =====================================================================
        #         data interpolation to the Ka band grid  (+pressure correction)
        # =====================================================================
        
        press_corr = np.power(press_lev*1e-3,0.4)
        sp_i, var_i, weight_i, sp_sm, weight_sm, not_noise = {}, {}, {}, {}, {}, {}
        
        # plt.figure()
        
        for (ic,band) in enumerate(bands):       
            
            var1 = invdB(spec_lev_dB[band])
            var2 = cumtrapz(var1, x = dopp_bins[band], initial = 0)
            plt.plot(dopp_bins[band],spec_lev_dB[band],'--', 
                     color = 'C%d' % ic)
            var3 = tripex.interp.interp1(dopp_bins[band]*press_corr,
                        var2/press_corr, dopp_bins['ka'])                        
            sp_i[band] = dB(np.diff(var3, append = 0 )/dv['ka'])
            flag_bad = ((sp_i[band]<-50) | ~np.isfinite(sp_i[band]))
            sp_i[band][flag_bad] = -99.            
            
            
            var1 = spec_lev_var[band]*1
            var_i[band] = tripex.interp.interp1(dopp_bins[band]*press_corr,
                        var1-2*dB(press_corr), dopp_bins['ka'])
            
            flag_bad = ((var_i[band]>55**2) | ~np.isfinite(var_i[band]))
            var_i[band][flag_bad] = 99.**2
           
            weight_i[band] = 1/var_i[band] 
            weight_i[band][flag_bad] = 0.
            
            weight_sm[band] = uniform_filter(weight_i[band], size = 7)
            not_noise[band] = uniform_filter((weight_i[band]>1e-4).astype(float),
                                             size = 7)
            sp_sm[band] = uniform_filter(sp_i[band]*weight_i[band], 
                                          size = 7)/weight_sm[band]
                           
            sp_sm[band][weight_sm[band]<1e-3] = -99.
            
        
        # =====================================================================
        #         bad data at ka (usually looks like stretched spectrum 
        #         compared to the x-band)
        # =====================================================================
               
        for band in ['ka','w']:
            flag_in = ((dopp_bins['ka'] >= min_vel['x']) | 
                       (dopp_bins['ka'] <= max_vel['x'])) 
            flag_vel =  ((dopp_bins['ka']<4) | (dopp_bins['ka']>7.)) 
            
            dfr_tmp = sp_sm['x']-sp_sm[band] 
            
            ind_bad  = ((dfr_tmp<-2) & (not_noise['x']>.51) & 
                        flag_vel & flag_in & (not_noise[band]>.51)) 
            
            ka_overflows = (np.sum(ind_bad)>1)
            plt.plot(dopp_bins['ka'][ind_bad],sp_i[band][ind_bad],'o')
            weight_i[band][ind_bad] = weight_i[band][ind_bad]*0.
            var_i[band][ind_bad] = np.ones_like(var_i[band][ind_bad])*1e4   
             
          

            
        # plt.figure()
        # for ic,band in enumerate(bands): 
        #     plt.plot(dopp_bins['ka'], sp_i[band], '-',lw = 2, 
        #               color = 'C%d' % ic, label = band)
        #     plt.fill_between(dopp_bins['ka'], 
        #                 sp_i[band] - np.sqrt(var_i[band]),
        #                 sp_i[band] + np.sqrt(var_i[band]),
        #                         color = 'C%d' % ic, alpha = 0.3)  
        # plt.xlim(-1, 10)
        # plt.ylim(-45,30)
        
        dfr_i, var_dfr_i, weight_dfr_i = {},{}, {}
        ic = 0
        # plt.figure()
        for band1 in bands[1:]:
            for band2 in bands[:-1]:
                if band1==band2: continue
                tmp_str = '%s_%s' % (band1,band2)
                # print(tmp_str)
                var_dfr_i[tmp_str] = var_i[band1] +  var_i[band2]
                flag_bad = ((var_dfr_i[tmp_str]>55**2) | 
                            ~np.isfinite(var_dfr_i[tmp_str]))
                var_dfr_i[tmp_str][flag_bad] = 99.**2
                weight_dfr_i[tmp_str] = 1/var_dfr_i[tmp_str]
                weight_dfr_i[tmp_str][flag_bad] = 0.
                
                dfr_i [tmp_str] = sp_i[band1] - sp_i[band2]              
                
                # plt.plot(dopp_bins['ka'], dfr_i[tmp_str], '-',lw = 2, 
                #      color = 'C%d' % ic, label = tmp_str)
                # plt.fill_between(dopp_bins['ka'], 
                #         dfr_i[tmp_str] - np.sqrt(var_dfr_i[tmp_str]),
                #         dfr_i[tmp_str] + np.sqrt(var_dfr_i[tmp_str]),
                #                 color = 'C%d' % ic, alpha = 0.3)  
                # ic +=1
        # plt.grid()
        # plt.legend()
        # plt.xlim(-1, 10)
        # plt.ylim(-5,35)
        # plt.show(block = False)        
         
        
        
        # =====================================================================
        #         expected psd parametars usefull for detecting bada data
        # =====================================================================
        
        mdv_vec = np.array([mdv[band][ind_alt[band]]
                    for band in bands])
        
        fg_Dm =  retrievals.dm_leo['2DVD']((mdv_vec[1]-mdv_vec[0], 
                                mdv_vec[2]-mdv_vec[1] )).item()
        if np.isnan(fg_Dm): fg_Dm = .3
        
        fg_Dm_std =  0.25*fg_Dm               
        
        if fg_Dm<0.8:
            dfr_x_ka_m =dB(Z_lin['x'][ind_alt['x']])- dB(Z_lin['ka'][ind_alt['ka']])
            dfr_ka_w_m =dB(Z_lin['ka'][ind_alt['ka']])-dB(Z_lin['w'][ind_alt['w']])
        
         
            dfr_x_ka_best_guess = np.nanmax([dfr_x_ka_m,0])
            dfr_ka_w_best_guess = np.nanmax([dfr_ka_w_m,0])
        
            ind_best = np.argmin(
                (dfr_x_ka_best_guess-spectra_sim.DFR_val['x_ka'][:,0])**2 +
                (dfr_ka_w_best_guess-spectra_sim.DFR_val['ka_w'][:,0])**2)
            fg_Dm = spectra_sim.Dm_vect[ind_best]*1e3
            fg_Dm_std = 0.5
        
        fg_Dm = np.nanmax([np.nanmin([fg_Dm,2.]),0.1])
        ind_best = np.argmin(np.abs(fg_Dm-spectra_sim.Dm_vect*1e3))
        N0 = Z_lin['x'][ind_alt['x']]*spectra_sim.N0_0dBZ['x'][ind_best,0] 
               
        psd_lin_c = spectra_sim.get_exp_psd(Dm = fg_Dm*1e-3, N= N0)       
      
        pr_fg = spectra_sim.get_PR(psd_lin_c)        
        lowest_det_vel = spectra_sim.get_lowest_detectable_vel(Dm = fg_Dm*1e-3, 
                             PR=pr_fg, band ='x', turb_q = broad_lev, 
                             thres = Z_band_min+3)
        
        
        fg_wind = min_vel['x'] - lowest_det_vel   
        fg_turb = broad_lev*1.
        
        Z_int = {band: np.interp(height['ka'],height[band],dB(Z_lin[band]))
                 for band in bands}           
            
        ind_ice = np.where(np.abs(height['ka'] -(meltDataSingT_rain.melt_top.values +
                            50 + 250 ))<500)
        dfr_tmp_height =Z_int['x'][ind_ice]-Z_int['ka'][ind_ice]
        dfr_x_ka = np.nanmean(dfr_tmp_height[Z_int['x'][ind_ice]>-30]) 
        dfr_x_ka_std = np.nanstd(dfr_tmp_height[Z_int['x'][ind_ice]>-30]) 
        
        dfr_tmp_height = Z_int['x'][ind_ice]-Z_int['w'][ind_ice]
        dfr_x_w = np.nanmean(dfr_tmp_height[Z_int['x'][ind_ice]>-30]) 
        dfr_x_w_std = np.nanstd(dfr_tmp_height[Z_int['x'][ind_ice]>-30]) 
        
        ind_Rayl = (np.abs(dopp_bins['ka']-(min_vel['x']+1.))<1.)
        dfr_x_ka_Rayl = np.sum(dfr_i['x_ka'][ind_Rayl]*
                        weight_dfr_i['x_ka'][ind_Rayl])/np.sum(
                            weight_dfr_i['x_ka'][ind_Rayl])
                            
        ext_ka = np.nanmax([0,np.nanmin([dfr_x_ka_Rayl,dfr_x_ka])])
       
        
        dfr_x_w_Rayl = np.sum(dfr_i['x_w'][ind_Rayl]*
                        weight_dfr_i['x_w'][ind_Rayl])/np.sum(
                            weight_dfr_i['x_w'][ind_Rayl])
                            
        ext_w = np.nanmax([0,np.nanmin([dfr_x_w_Rayl,dfr_x_w])])
        
        
        ext_ka = np.minimum(ext_ka,ext_w)
        
        
        ind_close = np.where(np.abs((diff_pia.time-ti)/unit_t)<10)[0]
        sel_diff_pia = diff_pia.sel(
            {'time': diff_pia.time[ind_close]}, 'nearest')                           
        
        
        ct_h = np.max(height['ka'][(Z_lin['ka']>1e-3)])
        
        pia_ka_ct = np.nanmean(sel_diff_pia.diff_pia_x_ka_est_fred.values)
        # ext_ka = pia_ka_ct - (pia_gas_column['ka']-pia_gass_ice_level['ka'])
             
        pia_w_ct = np.nanmean(sel_diff_pia.diff_pia_ka_w_est_fred.values)
        # ext_w =  pia_w_ct + pia_ka_ct - ( pia_gas_column['w'] - 
        #                                  pia_gass_ice_level['w'])
        std_ext_ka = np.sqrt(1/np.nansum(1/sel_diff_pia.diff_pia_x_ka_std_fred.values**2))
        std_ext_w = np.sqrt(1/np.nansum(1/sel_diff_pia.diff_pia_ka_w_std_fred.values**2))
        
        
        ub_pia_w = np.nanmin([pia_ka_ct+pia_w_ct+(std_ext_w+std_ext_ka)*3.,
                              dfr_x_w+dfr_x_w_std*3.])        
        ub_pia_ka =  np.nanmin([pia_ka_ct+std_ext_ka*3.,
                                ub_pia_w, dfr_x_ka+dfr_x_ka_std*3.])
        if (ub_pia_w<=1e-2) | np.isnan(ub_pia_w):
            ub_pia_w = 6.
        if (ub_pia_ka<=1e-2) | np.isnan(ub_pia_ka):
            ub_pia_ka = 6.
        
        # ext_ka = np.nanmax([0.,np.nanmin([ext_ka,pia_ka_ct])])
        # ext_w = np.nanmax(0.,np.nanmin([ext_w,pia_ka_ct+pia_w_ct]))
            
        xb_gamma = np.array([dB(N0),fg_Dm, ext_ka, ext_w, fg_turb, fg_wind])
        lb_gamma = np.array([-15,0.1, 0., 0., 0.01, -2.])
        ub_gamma = np.array([90,3, ub_pia_ka, ub_pia_w, .5, 2.])
        lb_gamma[lb_gamma>ub_gamma] = 0.
        # ub_gamma = np.maximum(ub_gamma,lb_gamma*1.1)
        
        
        bounds = Bounds(lb=lb_gamma, ub=ub_gamma)       
        
        xb_gamma = map_between_bounds(xb_gamma,lb_gamma,ub_gamma,
                                      min_dist = 0.01)
        x0_gamma = xb_gamma*1.
        
        var_inv_b_gamma = 1/np.array([15,fg_Dm_std,  6, 6, 
                                0.1+fg_turb*2,0.2+fg_turb*2])**2

        dfr_oe = np.concatenate([dfr_i[dfr_str] for dfr_str in 
                               ['x_ka','x_w','ka_w']])       
        var_inv_dfr = np.concatenate([weight_dfr_i[dfr_str] 
                                     for dfr_str in ['x_ka','x_w','ka_w']])       
             
        refl_obs = np.array([dB(Z_lin[band][ind_alt[band]]) for band in bands]) 
        refl_var_inv = np.array([np.max(weight_i[band]) for band in bands])*10
        
             
        # CF_gamma(x, spectra_sim, x_b , var_inv_b , y_obs , var_inv_obs, 
        #     dv = 0.041282654, return_CFo = False, plot_dfr = False)
        
        args_gamma =  (spectra_sim, xb_gamma , var_inv_b_gamma,
                        dfr_oe, var_inv_dfr, dv['ka'], refl_obs, refl_var_inv,
                        False,  False)
        min_val_gamma = minimize(CF_gamma_refl, x0 = x0_gamma, args =args_gamma,
                method = 'SLSQP', options = {'disp': False, 'ftol': 0.001, 
                'eps': dv['ka'] }, bounds = bounds, )
                     
        
        hess_diag = nd.Hessdiag(CF_gamma_refl,step = dv['ka'] )
        hess_diag_vect = hess_diag(min_val_gamma.x, *args_gamma, )        
        std_gamma = 1/np.sqrt(hess_diag_vect)
        std_gamma[np.isnan(std_gamma)] = 1/np.sqrt(var_inv_b_gamma[np.isnan(std_gamma)])
        
        gamma_parameters = ['N0','Dm', 'pia_ka', 'pia_w', 'turb', 'wind']
        gamma_dict = {key: min_val_gamma.x[ind_key] 
                          for (ind_key, key) in enumerate(gamma_parameters)}
        gamma_std_dict = {key: std_gamma[ind_key] 
                          for (ind_key, key) in enumerate(gamma_parameters)}
        

         
        fg_dm_up=  gamma_dict['Dm']*1e-3
        fg_N0_up = invdB(gamma_dict['N0'])
        psd_lin_c = spectra_sim.get_exp_psd(Dm = fg_dm_up, N= fg_N0_up)
        psd_lin_x0 = spectra_sim.psd_gen(psd_lin_c)
        fg_WC_up = np.dot(spectra_sim.psd_gen.WC_basis, psd_lin_c)
        
        
        
        var_inv_oe_or = np.concatenate([weight_i[band] for band in bands])
        ind_large = np.where(dopp_bins['ka']>min_vel['x'])[0]
        
        weight_i['x'][ind_large] = np.maximum(weight_i['x'][ind_large],
                                                  1e-2)
       
      
        add_par_vect = min_val_gamma.x[-4:]
        lb_add_par = lb_gamma[-4:]
        ub_add_par = ub_gamma[-4:]
        var_inv_oe = np.concatenate([weight_i[band] for band in bands])        
        y_oe = np.concatenate([sp_i[band] for band in bands])
        y_oe_lin = invdB(y_oe)
        
        
        
        if CF_prev<1.5:
            psd_lin_x0 = x_sp_oe_prev*1.
            
         
        for spectra_simulator in [spectra_sim_very_coarse, spectra_sim_coarse,spectra_sim]:
            
            
               
            psd_lin_x0_c = spectra_simulator.psd_gen.get_basis_rep(psd_lin_x0)
            x0_sp = np.concatenate([dB(psd_lin_x0_c), add_par_vect])
            
            psd_lin_c = spectra_simulator.get_exp_psd(Dm = fg_dm_up,
                                                      N= fg_N0_up)

            var_inv_b_spline_psd = np.ones(psd_lin_c.size)/10**2
            var_inv_b_spline_psd[
                spectra_simulator.psd_gen.Dm_basis>fg_dm_up*2.5] = 1/15**2
            
            xb_sp = np.concatenate([dB(psd_lin_c), add_par_vect])
            var_inv_b_spline = np.concatenate([
                np.ones(psd_lin_c.size)*np.maximum(1/36, 
                    1/(gamma_std_dict['N0']*2+3)**2), 
                np.maximum(hess_diag_vect[-4:]/2, var_inv_b_gamma[-4:])])
            
            lb_spline = -50*np.ones_like(x0_sp)       
            ub_spline = 80*np.ones_like(x0_sp)  
            lb_spline[-4:] = lb_add_par
            ub_spline[-4:] = ub_add_par
            bounds = Bounds(lb = lb_spline, ub = ub_spline) 
            
            d_lu_spline = ub_spline-lb_spline        
            x0_sp = np.minimum(np.maximum(x0_sp,lb_spline+0.01*d_lu_spline),
                              ub_spline-0.01*d_lu_spline)
            
            
            args_spline = (spectra_simulator, y_oe, var_inv_oe, xb_sp, var_inv_b_spline, 
              gamma_dict['pia_ka'], gamma_dict['pia_w'], 
              gamma_dict['turb'], gamma_dict['wind'],refl_obs, refl_var_inv,dv['ka'],False)
            args_spline_2 = (spectra_simulator, y_oe, var_inv_oe, xb_sp, var_inv_b_spline, 
              gamma_dict['pia_ka'], gamma_dict['pia_w'], 
              gamma_dict['turb'], gamma_dict['wind'],refl_obs, refl_var_inv,dv['ka'],True)                    
                
            print('CF 0th iteration: %.1f' % CF_spline(x0_sp,*args_spline))
            min_val_spl = minimize(CF_spline, jac = CF_spline_grad,
                x0 = x0_sp,   method = method, bounds= bounds,                
                options = {'disp': False },
                args = args_spline)
            
            x1_sp = min_val_spl.x
            retr_converged = min_val_spl.success
            CF_o_val,_,_,_ = CF_spline(x1_sp,*args_spline_2)
            
                        
            print('CF 1st iteration: %.1f' % CF_spline(x1_sp,*args_spline))            
            args_spline = (spectra_simulator, y_oe, var_inv_oe, xb_sp, var_inv_b_spline, 
               None, None, None, None,refl_obs, refl_var_inv,dv['ka'],False)   
            args_spline_2 = (spectra_simulator, y_oe, var_inv_oe, xb_sp, var_inv_b_spline, 
               None, None, None, None,refl_obs, refl_var_inv,dv['ka'],True) 
            
            min_val_spl = minimize(CF_spline,  jac = CF_spline_grad,
                x0 = x1_sp, method = method, bounds= bounds,  options = {'disp': False },
                args = args_spline)
            
            x2_sp = min_val_spl.x
            
            print('CF 2nd iteration: %.1f' % CF_spline(x2_sp,*args_spline))
            CF_o_val2,_,_,_ = CF_spline(x1_sp,*args_spline_2)
            
            
            retr_converged = (retr_converged & (CF_o_val2<CF_o_val)) | min_val_spl.success           
            
            print('CF_o: %.1f; CF_b: %.1f; CF_c: %.1f; CF_z: %.1f' % 
                CF_spline(x2_sp,*args_spline_2))         
            
            
                
            psd_lin_x0 = spectra_simulator.psd_gen(invdB(min_val_spl.x[:-4]))
                    
        
            
        
        psd_gamma_lin =  spectra_simulator.psd_gen(psd_lin_c)
                   
        x_final = min_val_spl.x
        psd_spl_final = x_final[:-4]
        
        args_spline = (spectra_simulator, y_oe, var_inv_oe, xb_sp, var_inv_b_spline, 
               None, None, None, None,refl_obs, refl_var_inv,dv['ka'],False) 
        
        # grad_spline = nd.Gradient(CF_spline, step = dv['ka'] )
        # grad_spline_vect = grad_spline(xb_sp, *args_spline)  
        # grad_vect = CF_spline_grad(xb_sp, *args_spline)
        
        # plt.figure()
        # plt.plot(grad_vect)
        # plt.plot(grad_spline_vect,'--')
        # plt.grid()
        # plt.ylim(-20,20)
            
        
        hess_spline = nd.Gradient(CF_spline_grad, step = dv['ka'])
        hess_spline_arr = hess_spline(x_final,  *args_spline )
     
        std_spline = 1/np.sqrt(np.diag(hess_spline_arr))
        
        spline_std_dict = {'pia_ka': std_spline[-4], 'pia_w': std_spline[-3],
                    'turb': std_spline[-2], 'wind': std_spline[-1] }
        spline_std_dict =  gamma_std_dict
        
        
        spline_dict = {'pia_ka': x_final[-4], 'pia_w': x_final[-3],
                    'turb': x_final[-2], 'wind': x_final[-1] }   
        
        
       
        retr_pert = {'default': psd_spl_final*1.}
        for perturb in spline_dict.keys():
            tmp_dict = {key: spline_dict[key]*1. for key in spline_dict.keys()}
            tmp_dict[perturb] += spline_std_dict[perturb]*1. 
            pert_par_vect = np.array([tmp_dict[key] for key in tmp_dict.keys()])
            
            x0_sp = np.concatenate([psd_spl_final, pert_par_vect])            
            x0_sp = np.minimum(np.maximum(x0_sp,lb_spline+0.01*d_lu_spline),
                              ub_spline-0.01*d_lu_spline)           
            
            args_spline = (spectra_simulator, y_oe, var_inv_oe, xb_sp, var_inv_b_spline, 
              tmp_dict['pia_ka'], tmp_dict['pia_w'], 
              tmp_dict['turb'], tmp_dict['wind'],refl_obs, refl_var_inv,dv['ka'],False)
            
            tmp =minimize(CF_spline, jac = CF_spline_grad,
                x0 = x0_sp,   method = method, bounds= bounds,                
                options = {'disp': False },args = args_spline)
            
            
            retr_pert[perturb] = tmp.x[:-4]*1.            
            print('CF pert %s: %.1f' % (perturb,CF_spline(tmp.x,*args_spline)))            
            print()
                
             
        retr_moments = {}
        # plt.figure()
        for key in retr_pert.keys():
            retr_moments[key] = {}
            psd_sp_lin_tmp = invdB(retr_pert[key]*1.)
            retr_moments[key]['wc'] = np.dot(psd_sp_lin_tmp, 
                                             spectra_sim.psd_gen.WC_basis)
            retr_moments[key]['pr'] = np.dot(psd_sp_lin_tmp, 
                                             spectra_sim.psd_gen.PR_basis)
            retr_moments[key]['dm'] = np.dot(psd_sp_lin_tmp, 
                                             spectra_sim.psd_gen.Dm_basis*
                            spectra_sim.psd_gen.WC_basis)/retr_moments[key]['wc']
           
            # plt.plot(spectra_sim.psd_gen.psd_peaks*1e3,
            #          dB(np.abs(retr_pert['default']- retr_pert[key])),
            #          '--', label = key)
        # plt.ylim(-20,20)
        # plt.legend()
        # plt.grid()
        
        delta_pr_sq, delta_dm_sq = 0., 0.
        delta_psd_sq = np.zeros(x_final.size-4)
        for perturb in spline_dict.keys():
            # print('delta PR due to %s is %.2f mm/h' % (perturb,
            #  np.abs(retr_moments['default']['pr']-retr_moments[perturb]['pr'])))
            delta_pr_sq += (retr_moments['default']['pr']-retr_moments[perturb]['pr'])**2
            
            # print('delta Dm due to %s is %.2f mm' % (perturb,
            #  np.abs(retr_moments['default']['dm']-retr_moments[perturb]['dm'])*1e3))
            delta_dm_sq += ((retr_moments['default']['dm']-retr_moments[perturb]['dm']))**2
            
            delta_psd_sq += (retr_pert[perturb]-retr_pert['default'])**2
        
                 
        
        args_spline = (spectra_simulator, y_oe, var_inv_oe, xb_sp, var_inv_b_spline, 
               x_final[-4], x_final[-3],
               x_final[-2], x_final[-1], refl_obs, refl_var_inv,dv['ka'],False) 
        
        hess_full_spline = nd.Gradient(CF_spline_grad)
        hess_spline_full_arr = hess_full_spline(x_final,  *args_spline )[:-4,:-4]
        
           
        H_inv = np.linalg.inv(hess_spline_full_arr)
        
        # plt.figure()
        # plt.pcolormesh(spectra_sim.psd_gen.psd_peaks*1e3,
        # spectra_sim.psd_gen.psd_peaks*1e3, H_inv, 
        #     cmap = 'jet',vmax = 2., vmin =-0.2)
        # plt.title('covariance matrix')
        # plt.xlabel('D [mm]')
        # plt.ylabel('D [mm]')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(os.path.join(figPathHour, 'cov_at_%s.png' % time_string), dpi =300)
        # plt.show()
        
        
        psd_sp_lin_final = invdB(psd_spl_final)
        
        
        wc_final = np.dot(psd_sp_lin_final, spectra_sim.psd_gen.WC_basis)
        rr_final = np.dot(psd_sp_lin_final, spectra_sim.psd_gen.PR_basis)
        dm_final = np.dot(psd_sp_lin_final, spectra_sim.psd_gen.Dm_basis*
                        spectra_sim.psd_gen.WC_basis)/wc_final
        
       
        sigma_rr =  np.sqrt(np.dot( np.matmul(H_inv+np.diag(delta_psd_sq),
                            psd_sp_lin_final*spectra_sim.psd_gen.PR_basis),
                            psd_sp_lin_final*spectra_sim.psd_gen.PR_basis)
                            *(np.log(10)/10)**2)
        
                            
        sigma_wc = np.sqrt(np.dot( np.matmul(H_inv+np.diag(delta_psd_sq),
                            psd_sp_lin_final*spectra_sim.psd_gen.WC_basis),
                            psd_sp_lin_final*spectra_sim.psd_gen.WC_basis)
                            *(np.log(10)/10)**2)
                            
        tmp = (spectra_sim.psd_gen.Dm_x_WC_basis -
                    spectra_sim.psd_gen.WC_basis*dm_final)/wc_final
       
        
        sigma_dm = np.sqrt(np.dot( np.matmul(H_inv+np.diag(delta_psd_sq),
                            psd_sp_lin_final*tmp), psd_sp_lin_final*tmp)
                            *(np.log(10)/10)**2)
        
            
        psd_lin_std = np.sqrt(np.sum( np.matmul(
            psd_sp_lin_final*spectra_sim.psd_gen.psd_basis,
            H_inv+np.diag(delta_psd_sq)
            )*(psd_sp_lin_final*spectra_sim.psd_gen.psd_basis), 
            axis = 1))*np.log(10)/10
        
        
        psd_final_lin = spectra_sim.psd_gen(psd_sp_lin_final)                          
        psd_final = dB(psd_final_lin)           
        
        psd_dB_std = 10/np.log(10)*psd_lin_std/psd_final_lin
        
                                                   
               
        
    
        # wc1 = doppler.psd.WC(psd_final_lin,DD)
        # dm1 = doppler.psd.Dm(psd_final_lin,DD)
        sm1 = doppler.psd.Sm(psd_final_lin,DD)
        # rr1 = doppler.psd.RR(psd_final_lin,DD)
        
      
        sim_spec,_ = simulate_spectra(psd_sp_lin_final,spectra_sim, 
                 pia_ka = spline_dict['pia_ka'], pia_w =spline_dict['pia_w'], 
                turb=spline_dict['turb'], wind=spline_dict['wind'], 
                dv = dv['ka'])
            
        
        Z_s, V_s = {}, {}
        for  band in bands:
            sim_spec[band][~np.isfinite(sim_spec[band])] = -99.
            sp_lin = invdB(sim_spec[band])
            Z_s[band] = np.trapz(sp_lin, Vel)
            V_s[band] = np.trapz(sp_lin*Vel,Vel, )/Z_s[band]
            
        args_spline = (spectra_simulator, y_oe, var_inv_oe_or, xb_sp, var_inv_b_spline, 
               None, None, None, None,refl_obs, refl_var_inv,dv['ka'],True)    
        
        CF_obs_final,_,_,_ = CF_spline(x_final,*args_spline)
        CF_obs_final *= 1/np.sum(var_inv_oe)
        
        
        CF_prev = CF_obs_final
        x_sp_oe_prev = psd_final_lin*1.
        
        # plt.figure()
        # for iii in range(3):
        #     plt.plot(Vel,dy[iii*Vel.size:(iii+1)*Vel.size])
        # plt.show(block = False)
        
                
        retr['PSD'][it,:] = psd_final
        retr['PSD_std'][it,:] = psd_dB_std
        retr['WC'][it] = wc_final
        retr['Dm'][it] = dm_final
        retr['Sm'][it] = sm1
        retr['RR'][it] = rr_final
        
        retr['wind'][it] = spline_dict['wind']
        retr['turb'][it] = spline_dict['turb']
        retr['pia_ka'][it] = spline_dict['pia_ka']
        retr['pia_w'][it] = spline_dict['pia_w']
        
        retr['wind_std'][it] = spline_std_dict['wind']
        retr['turb_std'][it] = spline_std_dict['turb']
        retr['pia_ka_std'][it] = spline_std_dict['pia_ka']
        retr['pia_w_std'][it] = spline_std_dict['pia_w']
        

        retr['Dm_std'][it] = sigma_dm
        retr['Ka_saturated'][it] = float(ka_overflows)
        retr['retr_converged'][it] = retr_converged.astype(float)
        retr['RR_std'][it] = sigma_rr
        
        
        psd_basis_dset['PSD_expansion_coef'][it] = psd_spl_final
        psd_basis_dset['CF_Hessian_inv'][it] = H_inv+np.diag(delta_psd_sq)
        
    
        for band in bands:
            retr['Zm_' + band ][it]= dB(Z_lin[band][ind_alt[band]])
            retr['Zs_' + band ][it]= dB(Z_s[band])
            retr['Vm_' + band ][it]= mdv[band][ind_alt[band]]
            retr['Vs_' + band ][it]= V_s[band]
            
        retr['CF_meas'][it] = CF_obs_final
        retr['WC_fg'][it] = fg_WC_up
        retr['N0_fg'][it] = gamma_dict['N0']
        retr['Dm_fg'][it] = gamma_dict['Dm']*1e-3
        retr['Sm_fg'][it] = gamma_dict['Dm']/2*1e-3
        retr['N0_fg_std'][it] = gamma_std_dict['N0']
        retr['Dm_fg_std'][it] = gamma_std_dict['Dm']*1e-3
        retr['wind_fg'][it] = fg_wind
        retr['turb_fg'][it] = fg_turb
        retr['pia_ka_fg'][it] = ext_ka
        retr['pia_w_fg'][it] = ext_w
        
        
        plt.figure(figsize = (9., 4.5))
        ax = plt.subplot(1,2,1)
        for (ii_band,band) in enumerate(bands):
            color = 'C%d' % (ii_band,)
            # plt.plot(Vel, sp_lev[band], label = '%s meas, %.1f [dBZ]' % (band, retr['Zm_' + band ][it]),
            #             color = color,lw =2)
            plt.plot(Vel/press_corr, invdB(sim_spec[band])*press_corr, '--',color=color, lw =2,)
                     # label = '%s sim [dBZ], %.1f' % (band, retr['Zs_' + band ][it]),)
            plt.plot(Vel, invdB(sp_i[band]), lw = 2, color = color, label = band)
            tmp_std = np.sqrt(var_i[band])
            tmp_std[tmp_std>25] = 25. 
            plt.fill_between(Vel, invdB(sp_i[band] - tmp_std),
                        invdB(sp_i[band] + tmp_std),
                                color = color, alpha = 0.3)
        plt.grid()
        # plt.title(time_string)
       
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Spectra [mm$^6$m$^{-3}$ per m s$^{-1}$]')
        plt.xlim(-1,10)
        plt.ylim(10**(-3),1e3)
        plt.yscale('log')
        plt.text(0, 2e2, '(a)', fontsize = 18)
        plt.legend(loc = 'upper right')
        plt.text(0,1.2e-3, 'turb = %.2f $\pm$ %.2f m/s' % (
            spline_dict['turb'],spline_std_dict['turb'] ,), fontsize = 16)
        plt.text(0,4e-3, 'wind = %.2f $\pm$ %.2f m/s' % (
            spline_dict['wind'] ,spline_std_dict['wind']), fontsize = 16)
        plt.text(0,1.2e-2, 'PIA Ka = %.1f $\pm$ %.1f dB' % (
            spline_dict['pia_ka'],spline_std_dict['pia_ka'] ,), fontsize = 16)
        plt.text(0,4e-2, 'PIA W = %.1f $\pm$ %.1f dB' % (
            spline_dict['pia_w'] ,spline_std_dict['pia_w']), fontsize = 16)
        
       
        
        ax = plt.subplot(1,2,2)
        plt.plot(DD*1e3,psd_final_lin, lw =2, label = 'binned PSD')
        plt.fill_between(DD*1e3,invdB(psd_final)-psd_lin_std, invdB(psd_final)+psd_lin_std, 
                lw =2,  color = 'C0', alpha = 0.5)
        plt.plot(DD*1e3,(psd_gamma_lin), lw =2, label = '$\Gamma$ PSD')
        # plt.plot(DD*1e3,(psd_final_lin*DD**3*np.pi/6*1e9), lw =2, label = 'mass [mg/m]')
        plt.grid()
        # plt.title(time_string)
        
        plt.xlabel('$D_{eq}$ [mm]')
        plt.ylabel('particle concentration [m$^{-4}$]')
        plt.text(0.1,4e0, 'RR = %.2f $\pm$ %.2f mm/h ' % (
            rr_final, sigma_rr), fontsize = 16)        
        plt.text(0.1,1.2e0, 'Dm = %.2f $\pm$ %.2f mm' % (
            dm_final*1e3,sigma_dm*1e3),  fontsize = 16)
        # plt.text(0.5,3.5e1, '$\sigma_m$ = %.2f [mm]' % (retr['Sm'][it] * 1e3,),  fontsize = 16)
        plt.legend(loc = 'upper right')
        plt.xlim(0,3.75)
        plt.ylim(1e0,1e8)
        plt.yscale('log')
        plt.text(0.3, 2e7, '(b)', fontsize = 18)
        plt.tight_layout(0.3)
        plt.subplots_adjust(bottom=0.13, right=0.99, left =0.1, top=0.98, hspace = 0.02)
        plt.savefig(os.path.join(figPathHour, 'psd_at_%s.png' % time_string), dpi =300)
        # plt.show(block = False)
        
        
        
    except Exception as e:
        print(e)
        
    # plt.close('all')
    
    
    
    
    if (np.mod(it,60)==0) and not test_flag:
    #     attrs_t = {}
    #     attrs_r = {'units': 'm'}
    #     ds = xr.Dataset(coords = {'time': ('time', spData['ka'].time.values[ind_comp],attrs_t),
    #                         'diameter': ('diameter', DD, attrs_r)})
        
    #     for var in aver_vars_list:
    #         ds[var] = (('time'), retr[var], {'units': retr_units[var]})
    #     ds['PSD'] = (('time','diameter'), retr['psd'], {'units': 'dB(m^-4)'})
    #     ds['PSD_std'] = (('time','diameter'), retr['psd_std'], {'units': 'dB(m^-4)'})
                
    #     comp = dict(zlib=True, complevel=5)
    #     encoding = {var: comp for var in ds.data_vars}
    #     encoding['time'] ={'units':'seconds since 1970-01-01'}
        
        # if os.path.exists(f_name):
        #     os.remove(f_name)
        retr.to_netcdf(f_name,     encoding = encoding)
        psd_basis_dset.to_netcdf(f_name2,     encoding = encoding2)
        
# retr.Dm.plot()
if not test_flag:
    retr.to_netcdf(f_name,     encoding = encoding)
    psd_basis_dset.to_netcdf(f_name2,     encoding = encoding2)
    #     plt.close('all')
        
        