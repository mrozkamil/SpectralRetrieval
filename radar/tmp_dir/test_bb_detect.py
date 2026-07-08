#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:41:55 2019

@author: km357
"""

import matplotlib.pyplot as plt
import numpy as np

import os, sys
home = os.path.expanduser("~")
module_dir = os.path.join(home, 'Documents','Python3',
    'my_modules', 'tools' )
if os.path.isdir(module_dir):
    sys.path.append(module_dir)

import tools.io
import tools.calc
import tools.radar



hiwrap_folder = os.path.join( home, 'Data', 'iphex','HIWRAP','data', 'nc')
all_files = os.listdir(hiwrap_folder) 

file = [f for f in all_files if '20140503' in f and 'HKu' in f 
        and '.h5' not in f and '._' not in f]

file_ku = os.path.join(hiwrap_folder, file[0])
file = [f for f in all_files if '20140503' in f and 'HKa' in f 
        and '.h5' not in f and '._' not in f]
file_ka = os.path.join(hiwrap_folder, file[1])

ku, ku_attr, _, ku_file_attr  = tools.io.ncutils.read_file(file_ku)
ka, ka_attr, _, ka_file_attr  = tools.io.ncutils.read_file(file_ka)


idx = 6000
refl = ku['zku'][idx,:]
altitude = ku['altitude'][idx]-ku['range']
freezing_height = 4000
alt_bb, ind_bb, z_bb = tools.radar.detectMelt1d(refl, altitude, freezing_height)
alt2_bb, ind2_bb, z2_bb = tools.radar.detectMelt1d(refl, altitude, freezing_height, der_cal = 'poly')

plt.figure()
plt.plot(ku['zku'][idx,:], ku['altitude'][idx]-ku['range'])
plt.grid()
plt.plot(z_bb, alt_bb,'x')
plt.plot(z2_bb, alt2_bb,'o')

ind_BB = np.zeros((ku['altitude'].size,5))
alt_BB = np.zeros((ku['altitude'].size,5))
alt2_BB = np.zeros((ku['altitude'].size,5))
z_BB = np.zeros((ku['altitude'].size,5))

for idx in range(ku['altitude'].size):
    refl = ku['zku'][idx,:]
    ang_corr=np.cos(np.deg2rad(ku['roll'][idx])) * np.cos(np.deg2rad(ku['pitch'][idx]))
    altitude = (ku['altitude'][idx]-ku['range']*ang_corr)*1e-3
    freezing_height = 4
    alt_bb, ind_bb, z_bb = tools.radar.detectMelt1d(refl, altitude, freezing_height)
    
    ind_BB[idx,:] = ind_bb
    alt_BB[idx,:] = alt_bb
    z_BB[idx,:] = z_bb
    alt_bb, ind_bb, z_bb = tools.radar.detectMelt1d(refl, altitude, freezing_height, der_cal = 'poly')
    alt2_BB[idx,:] = alt_bb
    print(idx)
    
#plt.figure()
#plt.plot(z_BB[:,0])
ang_corr=np.cos(np.deg2rad(ku['roll'][:])) * np.cos(np.deg2rad(ku['pitch'][:]))
alt = (ku['altitude'][:,np.newaxis]-ku['range'][np.newaxis,:]*ang_corr[:,np.newaxis])*1e-3
_,TimeUTC = np.meshgrid(ku['range'],ku['timed'],)
plt.figure()
plt.pcolormesh(TimeUTC,alt,ku['zku'], cmap = 'inferno')
plt.clim(0,45)
plt.colorbar()
plt.tight_layout()
plt.plot(ku['timed'],alt2_BB)



tayl_dop = tools.calc.loc_deriv(ka['dopcorr'],dx = 0.037, axis = 1,order =1,gates =13)[0]
ang_corr_ka=np.cos(np.deg2rad(ka['roll'][:])) * np.cos(np.deg2rad(ka['pitch'][:]))
alt_ka = (ka['altitude'][:,np.newaxis]-ka['range'][np.newaxis,:]*ang_corr_ka[:,np.newaxis])*1e-3
_,TimeUTC_ka = np.meshgrid(ka['range'],ka['timed'],)

tayl_dop[1][ang_corr_ka<0.99,:]= np.nan

plt.figure()
plt.pcolormesh(TimeUTC_ka,alt_ka, tools.misc.nanconv(tayl_dop[1], np.ones((125,1))/125), cmap = 'bwr')
plt.clim(-1.5,1.5)
plt.colorbar()
plt.tight_layout()
plt.plot(ku['timed'],alt2_BB)



