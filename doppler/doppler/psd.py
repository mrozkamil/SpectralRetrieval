#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:44:14 2019

@author: km357

"""
import numpy as np
from scipy.special import gamma
from . import simulate

def moment(dist,x, moment = 0, center = 0, integration_axis = 2):   
    if isinstance(center, np.ndarray):
        C = np.expand_dims(center, axis=integration_axis)
    else:
        C = center
    return np.trapz(dist*np.power(x-C,moment),x, axis = integration_axis)

def WC(psd,D,mass = None):
    mass = np.pi/6*1e3*(D**3) if mass is None else mass
    return np.trapz(psd*mass,D)

def vel_atlas(x, alpha= 1.173, beta = 1.204, gamma = 1676.106):
    return alpha-beta*np.exp(-gamma*x)

def RR(psd,D,vel = None, mass = None):
    mass = np.pi/6*1e3*(D**3) if mass is None else mass
    vel = simulate.rain_vel(D) if vel is None else vel
    return 3.6*1e3*np.trapz(psd*mass*vel,D)

def Dm(psd,D,mass = None, wc = None):
    mass = np.pi/6*1e3*D**3 if mass is None else mass
    wc = WC(psd,D,mass =mass) if wc is None else wc
    return np.trapz(D*mass*psd,D)/wc

def Sm(psd,D,mass = None, wc = None,dm= None):
    mass = np.pi/6*1e3*D**3 if mass is None else mass
    wc = WC(psd,D,mass =mass) if wc is None else wc
    dm = Dm(psd,D,mass =mass,wc=wc) if dm is None else dm
    return np.sqrt(np.trapz(mass*psd*(D-dm)**2,D)/wc)

def gamma_psd(D, Nt = 1e8, beta = 4e3, alpha= 1., ):
    pdf = np.power(beta, alpha)*np.power(D,alpha-1)*np.exp(-D*beta)/gamma(alpha)
    return Nt*pdf

def gamma_psd_met(D, WC = 1e-5, Dm = 1e-3, mu= 0., ):
    L = (4+mu)/Dm
    pdf = np.power(D,mu)*np.exp(-D*L)
    WC_t = 1e3*np.pi/6*gamma(mu+4.)*np.power(L, -(mu+4.))
    return WC/WC_t*pdf

def gen_gamma_psd(D, Nt = 1e8, beta = 4e3, alpha= 1., delta =1. ):
    pdf = (delta* np.power(beta, alpha)*np.power(D,alpha-1)*
           np.exp(-np.power(D*beta, delta))/gamma(alpha/delta))
    return Nt*pdf
