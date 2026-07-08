from scipy.interpolate import splrep, splev, RectBivariateSpline
from scipy.integrate import cumtrapz
import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter


def dB(val):
    return 10*np.log10(val)
def invdB(val):
    return np.power(10, val/10)

# import matplotlib.pyplot as plt

class spline_gen():
    def __init__(self, out_nodes, nodes, deg = 2):
        self.deg = deg
        self.internal_nodes = nodes
        self.out_nodes = out_nodes

        tmp_f = out_nodes**3

        self.tck = splrep(out_nodes,
                        tmp_f, task = -1,  t= nodes, k=deg)
        self.all_nodes = self.tck[0]

        self.all_node_cent = np.zeros(self.all_nodes.size)
        self.node_weigh = np.zeros(self.all_nodes.size)

        # plt.figure()
        for ii in range(self.all_nodes.size):
            node_i = np.zeros_like(self.all_nodes)
            node_i[ii] =1.
            tck = (self.tck[0], node_i, self.tck[2])
            tmp_v = splev(out_nodes,tck = tck, ext = 3)
            self.node_weigh[ii] = np.trapz(tmp_v,out_nodes)
            self.all_node_cent[ii] = np.trapz(out_nodes*tmp_v,out_nodes)/self.node_weigh[ii]
            # if ~np.isclose(self.node_weigh[ii],0.):
            #     plt.plot(out_nodes,tmp_v, color = 'C%d' % np.mod(ii,8))
            #     plt.plot(self.all_node_cent[ii],0., 'o',color = 'C%d' % np.mod(ii,8))
            # plt.grid()
            # plt.show()

        self.node_signif = ~np.isclose(self.node_weigh,0.)
        self.ind_node_signif = np.where(self.node_signif)[0]
        self.node_cent = self.all_node_cent[self.ind_node_signif]

        print('vector with %d required' % (self.ind_node_signif.size))

    def __call__(self,node_values, der = 0):
        node_values_ext = np.zeros_like(self.all_nodes)
        node_values_ext[self.ind_node_signif] = node_values
        tck = (self.tck[0], node_values_ext, self.tck[2])
        return splev(self.out_nodes,tck = tck, ext = 3, der = der)

    def get_rep(self, x,y):
        tmp =  splrep(x, y, task = -1,  t= self.internal_nodes, k=self.deg)[1]
        return tmp[self.node_signif]



def binned_psd_2_smooth(bin_edges, bin_values_dB, x_out, lowest_val = -99.):

    bin_values_dB[~np.isfinite(bin_values_dB)] = lowest_val
    sp = spline_gen(x_out,x_out[1:-1:3])
    bin_c = 0.5*(bin_edges[1:]+bin_edges[:-1])

    tmp_fun = np.interp(x_out, bin_c, bin_values_dB,
                left = lowest_val, right = lowest_val)

    fg = sp.get_rep(x_out,tmp_fun)

    ref_integral = invdB(bin_values_dB)*np.diff(bin_edges)

    def CF_spline(spl_x, ret_cf_obs = False):
        tmp_v = invdB(sp(spl_x))

        tmp_int = cumtrapz(tmp_v,x_out, initial = 0.)
        tmp_int_nod = np.interp(bin_edges,x_out,tmp_int)
        bin_integral = np.diff(tmp_int_nod)
        cf_obs = np.sum((ref_integral-bin_integral)**2)
        cf_cont = np.sum(np.diff(spl_x,n=2)**2)*1e-2

        # plt.figure()
        # plt.plot(bin_c,dB(ref_integral),'x')
        # plt.plot(bin_c,dB(bin_integral),'o')
        # plt.plot(x_out,dB(tmp_v))
        # plt.show()

        if ret_cf_obs:
            difference = np.sum(np.abs(ref_integral-bin_integral))
            print('%.1f drops per m^3 in the wrong bin' % difference)
        else:
            return cf_obs+cf_cont

    min_val_spl = minimize(CF_spline, x0 = fg,
            method = 'Powell', options = {'disp': False, 'ftol': 0.001 })

    CF_spline(spl_x, ret_cf_obs = True)
    psd_final = sp(min_val_spl.x)

    return psd_final

def reduce_vertical_gradient(x,y,z, dx_step = 300, fill_val = -30. ):
    """
    ind_t = np.where(np.abs((spData['x']['time'].values-np.datetime64('2018-11-24T08:00:00'))/
            np.timedelta64(1,'h'))<1.)[0]
    x = ((spData['x']['time'].values[ind_t]-np.datetime64('2018-11-24'))/
            np.timedelta64(1,'s'))
    ind_h = np.where((spData['x']['range']<6000) & (spData['x']['range']>1200) )[0]
    y = spData['x']['range'].values[ind_h]
    z = dB(spData['x']['Zg'].values[ind_t][:,ind_h])
    """

    sp = spline_gen(y,y[1:-1:5])
    dy = np.diff(y[0:2])[0]*1e-3

    z[~np.isfinite(z)] = fill_val
    z_rep = RectBivariateSpline(x,y,z, kx =1, ky=1)

    YY, XX = np.meshgrid(y,x)

    dz_dy = savgol_filter(z, window_length = 7, polyorder = 3, deriv=1,
            delta=1.0, axis=1)

    plt.figure()
    plt.pcolormesh(x,y,z.T)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.pcolormesh(x,y,dz_dy.T)
    plt.colorbar()
    plt.show()



    def CF_grad(z_vect, XX = XX, YY = YY, ret_z_mod= False,):

        d_XX = sp(z_vect)
        d_XX_cum = np.cumsum(d_XX)
        XX_sh = XX+d_XX_cum[np.newaxis,:]

        ZZ_sh = z_rep(XX_sh, YY, grid=False)
        ind_val = np.where((ZZ_sh>fill_val+20) & (XX_sh>x[0]) & (XX_sh<x[-1]))
        dz_sh_dy = savgol_filter(ZZ_sh, window_length = 7, polyorder = 3, deriv=2,
            delta=dy, axis=1)

        CF_g = np.sum(dz_sh_dy[ind_val]**2)/ind_val[0].size

        if ret_z_mod:
            return  XX_sh,ZZ_sh, dz_sh_dy
        else:
            return CF_g

    def callBack(z_vect):
        print(CF_grad(z_vect))


    ind_bot = 0

    while ind_bot<x.size:
        ind_top = np.minimum( ind_bot+dx_step, x.size)
        XX_gr = XX[ind_bot:ind_top]
        YY_gr = YY[ind_bot:ind_top]


        best_shift = minimize(CF_grad, x0 = -np.ones(sp.node_cent.size)*10.,
                args = (XX_gr,YY_gr,False), callback = callBack,
            method = 'Powell', options = {'disp': True, 'ftol': 0.001,
                'maxiter': 10 })


        ind_bot = ind_top

    XX_sh,ZZ_sh, dz_sh_dy = CF_grad(z_vect= best_shift.x,ret_z_mod= True,
                            XX=XX_gr, YY=YY_gr)



    plt.figure()
    plt.pcolormesh(x[ind_bot:ind_top],y,ZZ_sh.T)
    plt.plot(sp(best_shift.x)+x[ind_bot], y)
    # plt.plot(XX[ind_val],YY[ind_val],'.')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.pcolormesh(x[ind_bot:ind_top],y,dz_sh_dy.T)
    plt.plot(sp(best_shift.x)+x[ind_bot], y)
    # plt.plot(XX[ind_val],YY[ind_val],'.')
    plt.colorbar()
    plt.show()

















