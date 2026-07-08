from scipy import signal
from scipy.ndimage import filters
import os
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# def sens(r,a,b):
#     return a + b*10*np.log10(r)
# 
# def get_sens(Z, Range, plot_flag = False):
#     
#     bins = (np.arange(-90,100,0.25), np.arange(1.5,15,0.25))
#     z_r_hist = np.histogramdd((Z.ravel(), Range.ravel()), 
#                 bins=bins)[0]
#     ind_bott = np.zeros(z_r_hist.shape[1], dtype = int)
#     for ii in range(z_r_hist.shape[1]):
#         min_c = 10
#         ind = np.where(z_r_hist[:,ii]>min_c)[0]
#         while np.size(ind)==0:
#             min_c += -1
#             ind = np.where(z_r_hist[:,ii]>min_c)[0]
#         ind_bott[ii] = np.min(ind)
#     
#     ind = [ind_bott>0]
#     ran_c = (bins[1][1:]+bins[1][:-1])/2
#     popt, pcov = curve_fit(sens, 
#             ran_c[ind], bins[0][ind_bott[ind]])
#     if plot_flag:
#         print(popt)
#         plt.figure()
#         plt.pcolormesh(bins[0],bins[1],np.log(z_r_hist.T))
#         plt.plot(sens(ran_c,*popt), ran_c)
#         plt.plot(bins[0][ind_bott],ran_c)
#         plt.show()
#     return lambda r: sens(r,*popt)
    


import auxiliary_tools as at
loc_deriv = at.calc.loc_deriv

home = os.path.expanduser('~')
data_path = os.path.join(home, 'Data', 'StefanData2','20181124')


file = '20181124_140005.nc'
nc_file = Dataset(os.path.join(data_path,file))
variab = ['Zg','range','time','VELg','LDRg','RMSg', 'NPKg']
            
file = '20181124_1400.mmclx'
nc_file = Dataset(os.path.join(data_path,file))
variab = ['Zg','range','time','VELg','LDRg','RMSg', 'NPKg','TEMP','MeltHei',
            'MeltHeiDet','MeltHeiDB']
            
data = {}
names = {}
units ={}
for var in nc_file.variables.keys():
    data[var] = nc_file.variables[var][:].data
    names[var] = nc_file.variables[var].long_name
    # units[var] = nc_file.variables[var].units
    # print('%s [%s]' % (names[var],units[var]))
    print(var)
    print(names[var])
    print('')
aaa
nc_file.close()

plt.figure(99)
for var in variab:
    plt.figure()
    if len(data[var].shape)==1:
        plt.plot(data[var])
    else:
        plt.pcolormesh(data[var].T)
        plt.colorbar()
    plt.grid()
    plt.title(names[var])
    ii= 1800
    plt.figure(99)
    if len(data[var].shape)==2:
         plt.plot(data[var][ii,:],'--', label = var)
         
         
plt.figure(99)
plt.grid()
plt.legend()
plt.ylim(-50,50)
plt.show() 


p_melt_temp = gaussian(data['TEMP'], 0, 2.5)
LDR_db = 10*np.log10(data['LDRg'])

siz = 13
siz_der = 7
d2LDR_dh2 = signal.savgol_filter(LDR_db, window_length=siz_der, polyorder=3, deriv=2,
                    axis= 1, delta = -delta)
dV_dh = signal.savgol_filter(data['VELg'], window_length=siz_der, polyorder=5, deriv=1,
                    axis = 1, delta = -delta)
                    
min_LDR = filters.minimum_filter1d(LDR_db,siz, axis = 1)
max_LDR = filters.maximum_filter1d(LDR_db,siz, axis = 1)
ldr_local_ratio = (LDR_db-min_LDR)

plt.figure()
plt.pcolormesh(data['time'],data['range']*1e-3, LDR_db.T)
plt.colorbar()
plt.title('LDR [dB]')
plt.ylim(0,4)
plt.plot(data['time'],data['MeltHei']*1e-3,'-r')
plt.plot(data['time'],data['MeltHeiDB']*1e-3,'-m')
plt.show()



plt.figure()
plt.pcolormesh(d2LDR_dh2.T)
plt.clim(-2000,2000)
plt.colorbar()
plt.ylim(0,200)
plt.title('$d^2 LDR/dh^2$ [dB/km^2]')
plt.show()

plt.figure()
plt.pcolormesh(ldr_local_ratio.T)
plt.colorbar()
plt.ylim(0,200)
plt.title('LDR peak height [dB]')
plt.show()

plt.figure()
plt.pcolormesh(dV_dh.T)
plt.clim(-10,10)
plt.colorbar()
plt.ylim(0,200)
plt.title('dV/dh [m/s per km]')
plt.show()


plt.figure()
plt.pcolormesh((((ldr_local_ratio>5)&(d2LDR_dh2<0)).astype(float)).T)
plt.colorbar()
plt.show()









ldr_local_ratio = (LDR_db-min_LDR)
ldr_local_ratio_resc = ldr_local_ratio/np.nanmax(
                        ldr_local_ratio[:,5:],axis=1)[:,np.newaxis]

plt.figure()
plt.pcolormesh((ldr_local_ratio.T>5).astype(float))
plt.colorbar()
# plt.clim(0,1)
plt.show()



plt.figure()
plt.pcolormesh(p_melt_temp.T)
plt.show()

ldr_rescaled = data['LDRg']/np.nanmax(data['LDRg'][:,5:],axis=1)[:,np.newaxis]
plt.figure()
plt.pcolormesh(ldr_rescaled.T)
plt.colorbar()
plt.clim(0,1)
plt.show()


plt.figure()
plt.plot(data['range'],np.nansum(ldr_rescaled,axis =0))
plt.show()

Zdb = 10*np.log10(Z)
Range,_ = np.meshgrid(R,T)
sens_fun = get_sens(Zdb, Range, plot_flag = True)



flag = ~np.isfinite(Zdb)
V[flag] = np.nan
# Zdb[flag] = sens_fun(Range[flag])
# V[sens_fun(Range)+5>Zdb] = 0
# ii =1800
sig = Zdb[ii,:]


siz = 7
sig_savgol = signal.savgol_filter(sig, window_length=siz, polyorder=3, deriv=0)

#Create an order 3 lowpass butterworth filter
b, a = signal.butter(4, 0.1, 'lowpass', analog=False)
sig_lowpass = signal.filtfilt(b, a, sig)

sig_wiener = signal.wiener(sig,mysize=siz, noise = 0.5)


plt.figure()
plt.plot(sig,R,'o', label = 'data')
plt.plot(sig_savgol,R,'-', label = 'savgol')

plt.plot(sig_lowpass,R,'-', label = 'low pass')
plt.plot(sig_wiener,R,'-', label = 'wiener')
plt.grid()
plt.legend()
plt.show()


siz = 13
ker = np.exp(-(np.linspace(-2,2,5)/1)**2)
Zdb_sm = signal.convolve(Zdb,ker[np.newaxis,:],mode = 'same')

delta = np.nanmean(np.diff(R))
dZ_dh = signal.savgol_filter(Zdb, window_length=siz, polyorder=5, deriv=1,
                    axis = 1, delta = -delta)
d2Z_dh2 = signal.savgol_filter(Zdb, window_length=siz, polyorder=5, deriv=2,
                    axis= 1, delta = -delta)



plt.figure()
plt.pcolormesh(Zdb_sm.T, vmin = -30, vmax = 40)
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(dV_dh.T, vmin = -10, vmax = 10)
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(d2Z_dh2.T/100, vmin = -10, vmax = 10)
plt.colorbar()
plt.show()


fl_h = 1
p_prior = np.exp(-((Range-(fl_h-0.25))/0.25)**2)

def trans_func(x,x1,x2):
    xx = np.copy(x)
    x_ref = np.tan(np.pi/2*0.9)
    xx += -(x1+x2)/2
    xx *= 2*x_ref/(x2-x1)
    return np.arctan(xx)/np.pi+0.5

p_sec_der = 1-trans_func(d2Z_dh2[:],-700,-200)
p_vel_in = 1-trans_func(dV_dh[:],-7.5,-1)

plt.figure()
plt.pcolormesh(p_vel_in.T,)
plt.colorbar()
plt.show()

plt.figure()
plt.pcolormesh(p_sec_der.T,)
plt.colorbar()
plt.show()


plt.figure()
plt.pcolormesh((p_prior*(p_vel_in+p_sec_der)).T, vmin = 0, vmax = 1)
plt.colorbar()
plt.show()




