#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:22:14 2019

@author: km357
"""

from radar import simulate
import matplotlib.pyplot as plt
import numpy as np

D_test = np.linspace(0.05,3)
o = np.ones_like(D_test)
mu = 0
models = simulate.rain.keys()

for mu in [-2,0,3]:
    z={}
    dopp={}
    ext ={}
    assym = {}
    
    for model in models:
        z[model], ext[model], dopp[model], assym[model] = {}, {}, {}, {}
        
        for band in ['x','ka','w']:
            if band in simulate.rain[model].keys():
                z[model][band] = simulate.rain[model][band]['refl']((D_test,o*mu,o*280))
                ext[model][band]= simulate.rain[model][band]['ext']((D_test,o*mu,o*280))
                dopp[model][band] = simulate.rain[model][band]['dopp']((D_test,o*mu,o*280))
                assym[model][band] = simulate.rain[model][band]['g']((D_test,o*mu,o*280))
                
    plt.figure()
    for band in ['x','ka','w']:
        if band in z['mie'].keys():
            plt.plot(D_test,z['mie'][band],'-',label = 'mie at %s-band' % ( band,))
        if band in z['tmat'].keys():
            plt.plot(D_test,z['tmat'][band],'--',label = 'tmat at %s-band' % ( band,))
        if band in z['tmat_cant'].keys():
            plt.plot(D_test,z['tmat_cant'][band],':',label = 'tmat_cant at %s-band' % ( band,))
            
    plt.legend()
    plt.grid()
    plt.title('reflectivity [dBZ] for $\mu$= %d' % mu)
    plt.savefig('refl_mu_%d.png' % mu)
    
    plt.figure()
    for band in ['x','ka','w']:
        if band in ext['mie'].keys():
            plt.plot(D_test,ext['mie'][band],'-',label = 'mie at %s-band' % ( band,))
        if band in ext['tmat'].keys():
            plt.plot(D_test,ext['tmat'][band],'--',label = 'tmat at %s-band' % ( band,))
        if band in ext['tmat_cant'].keys():
            plt.plot(D_test,ext['tmat_cant'][band],':',label = 'tmat_cant at %s-band' % ( band,))
            
    plt.legend()
    plt.grid()
    plt.title('ext [dB/km] for $\mu$= %d' % mu)
    plt.savefig('ext_mu_%d.png' % mu)
    
    plt.figure()
    for band in ['x','ka','w']:
        if band in dopp['mie'].keys():
            plt.plot(D_test,dopp['mie'][band],'-',label = 'mie at %s-band' % ( band,))
        if band in dopp['tmat'].keys():
            plt.plot(D_test,dopp['tmat'][band],'--',label = 'tmat at %s-band' % ( band,))
        if band in dopp['tmat_cant'].keys():
            plt.plot(D_test,dopp['tmat_cant'][band],':',label = 'tmat_cant at %s-band' % ( band,))
            
    plt.legend()
    plt.grid()
    plt.title('dopp [m/s] for $\mu$= %d' % mu)
    plt.savefig('dopp_mu_%d.png' % mu)
    
    plt.figure()
    for band in ['x','ka','w']:
        if band in assym['mie'].keys():
            plt.plot(D_test,assym['mie'][band],'-',label = 'mie at %s-band' % ( band,))
        if band in assym['tmat'].keys():
            plt.plot(D_test,assym['tmat'][band],'--',label = 'tmat at %s-band' % ( band,))
        if band in assym['tmat_cant'].keys():
            plt.plot(D_test,assym['tmat_cant'][band],':',label = 'tmat_cant at %s-band' % ( band,))
            
    plt.legend()
    plt.grid()
    plt.title('g for $\mu$= %d' % mu)
    plt.savefig('assym_mu_%d.png' % mu)

