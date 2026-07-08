# import os, re
# from scipy.ndimage.filters import convolve
import numpy as np
from scipy.interpolate import splev, splrep #,sproot, make_lsq_spline, LSQUnivariateSpline
from scipy.optimize import minimize
import auxiliary_tools as at
loc_deriv = at.calc.loc_deriv
nanconv = at.misc.nanconv
gaussian1d = at.misc.gaussian1d
gauss_3db_vect = np.vectorize(gaussian1d)

def get_alt(alt, ran, pitch = 0, roll = 0):
    """
    Calculates the altitude of the data gates based on the altitude of a plane and the range gate
    
    Parameters
    ----------
    alt : :class:`numpy:numpy.ndarray` 
        One dimensional array of the altitude [m], 
    ran : :class:`numpy:numpy.ndarray`
        One dimensional array reprezenting the sampling range from 
        the airplane [m], 
    pitch : :class:`numpy:numpy.ndarray`
        One dimensional array reprezenting the pitch angle of the airplane [deg], 
    roll : :class:`numpy:numpy.ndarray`
        One dimensional array reprezenting the roll angle of the airplane [deg], 
    Returns
    -------
    alt_array : :class:`numpy:numpy.ndarray`
        Array of the altitudes of the data bins [m]
    """
    cos_roll = np.cos(np.deg2rad(roll))
    cos_pitch = np.cos(np.deg2rad(pitch))
    alt_array = alt[:,np.newaxis] - (
        (cos_roll*cos_pitch)[:,np.newaxis]*ran[np.newaxis,:])
    return  alt_array

def detect_melt1d(refl, altitude, bb_lims ,  
                  der_cal = 'spl', z_thres = -50., 
                  prec = None, w_x = None,  ):
    
    '''
    Function detects the top and the bottom of the melting zone from the 
    reflectivityprofile, based on the bright band signatures.
    
    Parameters
    ----------
    refl : :class:`numpy:numpy.ndarray` 
        The reflectivity profile, 
    altitude : :class:`numpy:numpy.ndarray` 
        The altitude profile in the same units as the freezing level height
    bb_lims: :class:'tuple'
        The range of altitudes (extent) of the bright band.
    der_cal : : string
        The method for derivatives' calculations, 
        default value 'spl' - derivatives calculated with spline expansion
        other options 'poly' - the local polynomial expantion 
    
    Returns
    -------
    alt_bb: :class:`numpy:numpy.ndarray`
        The melting level breaking points:
            top1: d^3z/dh^3 ==0,
            top2: d^2z/dh^2 ==0
            bb peak: dz/dh ==0
            bot2: d^2z/dh^2 ==0
            bot1: d^3z/dh^3 ==0
    ind_bb: :class:`numpy:numpy.ndarray`
        The index of the BB breaking points 
    dz_bb: :class:`numpy:numpy.ndarray`
        The reflectivity at the BB breaking points
    '''
    
    # determine the altitude units (km or m)
    max_alt, min_alt = np.nanmax(altitude), np.nanmin(altitude)
    total_alt = max_alt - min_alt
    
    #the default output
    ind_bb = np.full(5,-999, dtype = int)
    alt_bb = np.full(5,np.nan)
    dz_bb  = np.full(5,np.nan)
    
    # new variables created
    h = np.copy(altitude)
    ind_order = np.argsort(h) 
    h = h[ind_order]
    
    # h_fl = np.copy(freezing_height)
    z = np.copy(refl[ind_order])
    if w_x is not None:
        w_x = w_x[ind_order]
    
   
            
    
    # rescaling to km
    if total_alt>1e2:
        h = h*1e-3
        bb_lims[0] *= 1e-3
        bb_lims[1] *= 1e-3
        max_alt *= 1e-3
        min_alt *= 1e-3
        if prec is not None:
            prec *= 1e-3 #km
    if prec is None:
        prec = 0.25 #km
    
    #data in proximity to the FL and valid data considered only
    ind_valid = (z>z_thres)
    z = np.maximum(z,z_thres)
    near_peak = (h<=bb_lims[1]) & (h>=bb_lims[0]) 

    

    d_alt = h[1]-h[0]
    n_gates =  int((prec/2//d_alt)*2+1)
    
    if np.sum(near_peak & ind_valid)<n_gates:
        return alt_bb, ind_bb, dz_bb
        
    if w_x is None:
        w_x = (~ind_valid).astype(float)+1e-16
    order = 3  # order of the spline approximation

    if der_cal == 'spl':
        
        # calculations of the derivatives
        z_taylor = []
        
        #nodes
        t_alt = np.arange(np.ceil(min_alt/prec)*prec, 
                          np.floor(max_alt/prec)*prec+1e-6, prec) # spline nodes
        tck = splrep(h, z, w = w_x,k = order, t = t_alt, quiet = True)
        for ii in range(4):
            z_taylor.append(splev(h,tck,der = ii))

    elif der_cal == 'poly':
        z_taylor = loc_deriv(data = z, dx = d_alt, order = order, 
                             gates = n_gates, axis = 0, reg_term = 0.)[0]
    
    near_peak_valid = (near_peak & ~(np.isnan(z_taylor[0])))
   
    
    #detecting local maxima around the FL
    der_sign = np.sign(np.diff(z_taylor[0]))
    sign_change = np.ones_like(z)
    sign_change[1:-1] = der_sign[1:]*der_sign[:-1]
   
    ind_max_expec = (near_peak_valid & (sign_change<0) )
    if all(~ind_max_expec): 
        return alt_bb, ind_bb, dz_bb
        
    #ind where peak 
    i_max = np.argmax(np.ma.masked_where(~ind_max_expec,z))
    h_max_z = h[i_max]
    
    ind_bb[2] = i_max
    alt_bb[2] = h_max_z
    dz_bb[2] = z_taylor[0][i_max]
        
    
    sec_der_pos = (np.ma.masked_where(~ind_valid,z_taylor[2])>0).data
    third_der_pos = (np.ma.masked_invalid(z_taylor[3])>0).data    
    above_peak = (h>h_max_z)
    
    #top of the BB - the lowest point above the peak where d^2Z/dr^2 > 0
    ind_top_dr2 = np.argmin( np.ma.masked_where( 
                ~(sec_der_pos & near_peak_valid & above_peak ),h))
    
                
    #bot of the BB - the highest point below the peak where d^2Z/dr^2 > 0
    ind_bot_dr2 = np.argmax( np.ma.masked_where( 
                ~(sec_der_pos & near_peak_valid & ~above_peak ),h)) 
    
                
    if ind_top_dr2!=0:        
        ind_bb[1] = ind_top_dr2
        alt_bb[1] = h[ind_top_dr2]
        dz_bb[1] = z_taylor[0][ind_top_dr2] 
        # top of the BB based on the third derivative:
        # the lowest point above the inflection point above the BB 
        # where d^3Z/dr^3 > 0
        ind_top_dr3 = np.argmin( np.ma.masked_where( 
                    ~(~third_der_pos & near_peak_valid & 
                      (h>h[ind_top_dr2]) ),h))
        if ind_top_dr3!=0:        
            ind_bb[0] = ind_top_dr3
            alt_bb[0] = h[ind_top_dr3]
            dz_bb[0] = z_taylor[0][ind_top_dr3] 
        
              
    if  ind_bot_dr2!=0:
        ind_bb[3] = ind_bot_dr2
        alt_bb[3] = h[ind_bot_dr2]
        dz_bb[3] = z_taylor[0][ind_bot_dr2]
        # bot of the BB based on the third derivative:
        # the highest point below the inflection point below the BB 
        # where d^3Z/dr^3 < 0
        ind_bot_dr3 = np.argmax( np.ma.masked_where( 
                    ~(third_der_pos & near_peak_valid & 
                      (h<h[ind_bot_dr2])),h))          
              
        if  ind_bot_dr3!=0:
            ind_bb[4] = ind_bot_dr3
            alt_bb[4] = h[ind_bot_dr3]
            dz_bb[4] = z_taylor[0][ind_bot_dr3]
   # plt.figure()
   # plt.plot(z, h, lw =3)
   # plt.plot(dz_bb,alt_bb,'d')
   # # plt.plot(z_taylor[0],h)
   # # plt.plot(z_taylor[1]*1e-1,h)
   # plt.plot(z_taylor[2]*1e-2,h)   
   # # plt.plot(z_taylor[3]*1e-3,h) 
   # plt.plot(d_taylor[1]*1e-1,h,'--')
   # plt.xlim(-20,50)
   # plt.grid()
   # plt.show()  
    if total_alt>1e3:
        alt_bb = alt_bb*1e3
    ind_bb[ind_bb>=0] = ind_order[ind_bb[ind_bb>=0]]
    
    
    return alt_bb, ind_bb, dz_bb

'''
from auxiliary_tools.io import ncutils
import numpy as np
fname = '/Users/km357/Data/iphex/HIWRAP/data/nc/IPHEX_HIWRAP_L1B_20140503-185702-20140503-192402_HKu_dist_v01.nc'
refl = ncutils.ncread(fname, 'zku')
plane_alt = ncutils.ncread(fname, 'altitude') 
data_range = ncutils.ncread(fname, 'range')
plane_roll = np.deg2rad(ncutils.ncread(fname, 'roll'))
plane_pitch = np.deg2rad(ncutils.ncread(fname, 'pitch'))
altitude = (plane_alt[:,np.newaxis] 
    - data_range[np.newaxis,:]*(np.cos(plane_roll)*np.cos(plane_pitch))[:,np.newaxis])*1e-3
freezing_height = 4.2
der_cal = 'poly' 
alt_axis = 1

'''

def detect_melt_2d(refl, altitude, freezing_height, der_cal = 'poly', alt_axis = 1):
    
    '''
    Function detects the top and the bottom of the melting zone from the 
    reflectivity profiles given in a form of a matrix,
    based on the bright band signatures.
    
    Parameters
    ----------
    refl : :class:`numpy:numpy.ndarray` 
        The reflectivity profiles [dBZ]
    altitude : :class:`numpy:numpy.ndarray` 
        The altitude profiles in [km]
    freezing_height: :class:`numpy:numpy.ndarray`
        The altitude of the freezing level [km].
    der_cal : : string
        The method for derivatives' calculations, 
        default value 'spl' - derivatives calculated with spline expansion
        other options 'poly' - the local polynomial expantion 
    
    Returns
    -------
    alt_bb: :class:`numpy:numpy.ndarray`
        The melting level breaking points:
            top1: d^3z/dh^3 ==0,
            top2: d^2z/dh^2 ==0
            bb peak: dz/dh ==0
            bot2: d^2z/dh^2 ==0
            bot1: d^3z/dh^3 ==0
    ind_bb: :class:`numpy:numpy.ndarray`
        The index of the BB breaking points 
    dz_bb: :class:`numpy:numpy.ndarray`
        The reflectivity at the BB breaking points
    '''
    #refl, altitude, freezing_height = data['ku']['zku'], data['ku']['alt_array']*1e-3, 4.
    
    
    # new variables created
    #h = np.copy(altitude)
    z_shape= refl.shape   
    if alt_axis==1:
        h = altitude.T
        z = refl.T  

    ind_order = np.argsort(h[:,0])        
    d_alt = h[ind_order[1],0] - h[ind_order[0],0] 
    h = h[ind_order,:]
    z = z[ind_order,:]
    get_ind = lambda h_0: np.argmin((h-h_0[np.newaxis,:])**2, axis = alt_axis)
    grater_flag = lambda h_0: (h>h_0[np.newaxis,:])       
    hor_shape = z.shape[1]

    ind_valid = np.isfinite(z) & (z>-50)        
    prec = 4
    order = 3  # order of the spline approximation
    spl_supp = 1/prec*(order+1)/2
                 
    if der_cal == 'poly':
        z[~ind_valid] = np.NaN
        n_gates =  (spl_supp//d_alt).astype(int)
        z_taylor = loc_deriv(data = z, dx = 1., order = order, 
                             gates = n_gates, axis = 0, reg_term = 0.)[0]
    
    if np.isscalar(freezing_height):
        ap_fl = freezing_height*np.ones(hor_shape)
    elif  freezing_height.size == hor_shape:
        ap_fl = freezing_height[:]
    
    hor_var = np.arange(hor_shape,dtype = float) 
    
    x_0 = splrep(hor_var, ap_fl, t = hor_var[10::50])
    spl_rep = lambda xx: splev(hor_var, tck = [x_0[0], xx,x_0[2]])
    
    CF_ap = lambda xx: np.sum((xx-ap_fl)**2)
    # between the peak and the top of the BB Z is decreasing
    # between the bottom and the peak of the BB Z is increasing
    
    CF_first_der = lambda bb_bot_high, bb_peak_hight, bb_top_high : (np.sum(
            z_taylor[1][grater_flag(bb_peak_hight) & 
                    ~grater_flag(bb_top_high)]>0) + np.sum(
            z_taylor[1][~grater_flag(bb_peak_hight) & grater_flag(bb_bot_high)]<0)
            - 0.25*(np.sum(
            z_taylor[1][grater_flag(bb_peak_hight) & 
                    ~grater_flag(bb_top_high)]<0) + np.sum(
            z_taylor[1][~grater_flag(bb_peak_hight) & grater_flag(bb_bot_high)]>0)))
            
    CF_second_der = lambda bb_bot_high, bb_top_high : np.sum(
            z_taylor[2][grater_flag(bb_bot_high) & 
                    ~grater_flag(bb_top_high)]>0)- 0.25*np.sum(
            z_taylor[2][grater_flag(bb_bot_high) & 
                    ~grater_flag(bb_top_high)]<0)
    CF_third_der = lambda bb_bot_high, bb_peak_hight, bb_top_high : (np.sum(
            z_taylor[3][grater_flag(bb_peak_hight) & 
                    ~grater_flag(bb_top_high)]<0) + np.sum(
            z_taylor[3][~grater_flag(bb_peak_hight) & 
                    grater_flag(bb_bot_high)]>0)) - 0.25*(np.sum(
            z_taylor[3][grater_flag(bb_peak_hight) & 
                    ~grater_flag(bb_top_high)]>0) + np.sum(
            z_taylor[3][~grater_flag(bb_peak_hight) & 
                    grater_flag(bb_bot_high)]<0))
    CF_order = lambda bb_bot_high, bb_top_high: np.sum(np.maximum(bb_bot_high - bb_top_high,0))
    
    CF_total = lambda xx_long: ( CF_ap(spl_rep(xx_long[4::5])) +
            CF_first_der(spl_rep(xx_long[0::5]), spl_rep(xx_long[2::5]),spl_rep(xx_long[4::5]))+
            CF_second_der(spl_rep(xx_long[1::5]), spl_rep(xx_long[3::5]))+
            CF_third_der(spl_rep(xx_long[0::5]), spl_rep(xx_long[2::5]),spl_rep(xx_long[4::5]))+
            1e3*(CF_order(spl_rep(xx_long[0::5]), spl_rep(xx_long[1::5])) + 
            CF_order(spl_rep(xx_long[1::5]), spl_rep(xx_long[2::5])) +
            CF_order(spl_rep(xx_long[2::5]), spl_rep(xx_long[3::5])) + 
            CF_order(spl_rep(xx_long[3::5]), spl_rep(xx_long[4::5]))))
    
    xx_long_init = np.stack((x_0[1]-0.5, x_0[1]-0.4,x_0[1]-0.25, x_0[1]-0.1, x_0[1]),axis =1)
    
    out = minimize(CF_total, xx_long_init.reshape(xx_long_init.size), 
                   options = {'eps': d_alt*1.5})
    alt_bb = np.array([spl_rep(out.x[ii::5]) for ii in range(5)])
    ind_bb = np.array([np.argmin((h-alt_bb[ii,:][np.newaxis,:])**2, axis = 0) for ii in range(5)])
    dz_bb = np.array([z_taylor[0][ind_bb[ii,:], hor_var.astype(int)] for ii in range(5)])
    ind_bb = ind_order[ind_bb]
    return alt_bb, ind_bb, dz_bb

def increase_beam_width_fast(data,data_range, ground_dist, 
    beam_width_original, beam_width_new, weight=None):
    """
    Reduces the resolution of the data by increasing the beam width of the instrument

    Parameters
    ----------
    data : :class:`numpy:numpy.ndarray` 
        Two dimensional array of data, 
    data_range : :class:`numpy:numpy.ndarray` 
        One dimensional array of ranges from radar [m],
    ground_dist: :class:`numpy:numpy.ndarray` 
        One dimensional array of the distances from a given point during the flight [m],
    beam_width_original : scalar [deg]
        A 3-dB beam width of the antena
    beam_width_new : scalar [deg]
        A new 3-dB beam width of the antena, must be greater than the original
        
    
    Returns
    -------
    resampled_data : :class:`numpy:numpy.ndarray`
        Data smoothed out withrespect by increasing the beam-width
    """
    if beam_width_original>= beam_width_new:
        return data
        
    M = data.shape[0]
    N = data.shape[1]
    resampled_data = np.full((M, N), 10**(-9.99)) 
    sigma_new_square = beam_width_new**2 - beam_width_original**2
    sigma_deg = np.deg2rad(np.sqrt(sigma_new_square))
#    max_dist = 2*np.nanmax(data_range)*np.tan(sigma_deg/2)
    mean_delta_x = np.nanmean(np.diff(ground_dist))
    
    if weight is not None:
        ww = np.copy(weight)
        mask = ~np.isfinite(weight)
        ww[mask] = 0.
        
    for idx,ran in enumerate(data_range):
        width = np.round(2*ran*np.tan(sigma_deg/2)/mean_delta_x).astype(int)
        num_elem = 2*width+1
        if num_elem == 1:
            resampled_data[:,idx] = data[:,idx]
        else:
            x = ground_dist[range(num_elem)]
            x += -x[width]
            ang = np.arctan2(x,ran)
            w = gauss_3db_vect(ang,sigma_deg)
            w = w/np.sum(w)
            if weight is None:
                resampled_data[:,idx] = nanconv(data[:,idx],w)
            else:
                ker2 = np.ones(w.shape)/w.shape
                total_mass = nanconv(ww[:,idx],ker2)
                valid = (total_mass>0)
                nomin = nanconv(data[:,idx]*ww[:,idx],w)
                resampled_data[valid,idx] = (nomin[valid]/total_mass[valid])

    return resampled_data
