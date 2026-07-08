import numpy as np
import scipy.integrate as integrate
# from scipy.interpolate import splrep, splev
from scipy.special import gamma

chi = lambda x, x1,x2: np.heaviside(x-x1,0.5)-np.heaviside(x-x2,0.5)
def stairs(x,bin_edg, bin_val):
    v = np.zeros_like(x)
    ind_not_nan = np.where(np.isfinite(bin_val) & (bin_val!=0))[0]
    for ii in ind_not_nan:
        v +=  chi(x,bin_edg[ii],bin_edg[ii+1])*bin_val[ii]
    return v 
    

class PSD():
    """ Constructor of a particle size distribution (PSD)
        D is assumed to be in [m]
        we assume the following mass-diameter relation:
            m [kg] = a D^b
            where a = pi/6*1e3, b = 3 (i.e. water, D is and equivalent volume diameter)
        please modify the atributes mD_a and mD_b for any changes, 
        alternatively a handel to the mass-diamter function can be implicitely provided
    Attributes:
        D_min: the minimum diameter where the PSD>0 
        D_max: the maximum diameter where the PSD>0 
        mD_a: a prefactor of the mass-diameter relation
        mD_b: an exponent of the mass-diameter relation
        mass_function: the mass-diameter relation function
    """
    def __init__(self, D_min = 0, D_max = 8*1e-3, mD_a= None, mD_b = None, mass_function = None
    ):
        self.func = lambda D: np.zeros_like(D)
        self.D_min = D_min
        self.D_max = D_max
        mD_a = np.pi/6*1e3 if mD_a is None else mD_a
        mD_b = 3 if mD_b is None else mD_b
        self.mass_function = lambda D: mD_a* np.power(D, mD_b)
        
        
    def __call__(self, D):
        v = self.func(D)
        v[(D>self.D_max) & (D<self.D_min)] = 0
        return v

    def _get_weighted_moment(self, moment = 1, center=None, 
                weight_function = None, normalize = True):
        if weight_function is None:
            weight_function = lambda D: np.ones_like(D) 
        # print(weight_function)
        # print(weight_function(np.linspace(1,5)))
        
        center = 0. if center is None else center
        tmp_func = lambda D: self.func(D) * np.power(D-center,moment) * weight_function(D)
        tmp_v = integrate.quad(tmp_func, self.D_min, self.D_max)[0]
        if normalize:
            tmp_func = lambda D: self.func(D) *  weight_function(D)
            norm_const = integrate.quad(tmp_func, self.D_min, self.D_max)[0]
            tmp_v /= norm_const
        return tmp_v
    def Nt(self):
        return self._get_weighted_moment(moment = 0, normalize = False)
    def D_m(self):
        return self._get_weighted_moment(moment = 1, 
            weight_function=self.mass_function )
    def Sigma_m(self):
        return np.sqrt(self._get_weighted_moment( moment = 2, 
        center=self.D_m(), weight_function=self.mass_function ))
    def Skewness_m(self):
        return (self._get_weighted_moment(moment = 3, center=self.D_m(),
        weight_function=self.mass_function )/self.Sigma_m()**3)
    def Kurtosis_m(self):
        return (self._get_weighted_moment(moment = 4, center=self.D_m(), 
        weight_function=self.mass_function )/self.Sigma_m()**4)
    def WaterContent(self):
        return self._get_weighted_moment(moment = 0, center=None, 
                weight_function = self.mass_function, normalize = False)
    def show(self):
        plt.figure()
        tmp_d = np.linspace(0,self.D_max,101)[1:]
        tmp_n = self.__call__(tmp_d)
        plt.plot(tmp_d*1e3,tmp_n)
        plt.grid()
        plt.yscale('log')
        plt.ylabel('N [m$^{-4}$]')
        plt.xlabel('D [mm]')
        plt.tight_layout()
        plt.show(block = False)
 

class GammaPSD(PSD):
    """ Constructor of the Gamma particle size distribution (PSD) of the form:
    N(D) = N0 * D**mu * exp(-Lambda*D)
    D is assumed to be in [m]

    Attributes:
        N0: the intercept parameter, defaults to 8e6
        Lambda: the slope parameter, defaults to 4e3
        mu: the shape parameter, defaults to 0. 
        D_max: the maximum diameter where the PSD>0 (defaults to 11/Lambda, 
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """
    
    def __init__(self, N0=8e6, Lambda=4e3, mu=0.0, D_min = 0, D_max=None):
        D_max = 11.0/Lambda if D_max is None else D_max
        super().__init__(D_min, D_max)
        self.mu = mu
        self.N0 = N0
        self.Lambda = Lambda
        self.func = lambda D: self.N0 * np.power(D, self.mu)* np.exp(-self.Lambda*D)
        
class GeneralizedGammaPSD(PSD):
    """ Constructor of the Gamma particle size distribution (PSD) of the form:
    N(D) = M0 * c*Lambda/(gamma(mu))* (Lambda*D)**(c*mu-1) * exp(-(Lambda*D)**c)
    D is assumed to be in [m]

    Attributes:
        M0: the zero moment of the PSD, defaults to 8e6
        Lambda: the slope parameter, defaults to 4e3
        mu: the shape parameter, defaults to 0. 
        c: the second shape parameter, defaults to 1. 
        D_max: the maximum diameter where the PSD>0 (defaults to 11/Lambda, 
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """
    
    def __init__(self, M0=8e6, Lambda=4e3, mu=0.0, c = 1., D_min = 0, D_max=None):
        D_max = 11.0/Lambda if D_max is None else D_max
        super().__init__(D_min, D_max)
        self.mu = mu
        self.M0 = M0
        self.Lambda = Lambda
        self.c = c
        
        self.func = lambda D: self.M0*self.c*self.Lambda/gamma(self.mu)* (self.Lambda*D)**(self.c*self.mu-1) * np.exp(-(self.Lambda*D)**self.c)

class NormalizedGammaPSD(PSD):
    """Constructor of the normalized gamma particle size distribution (PSD)
    of the form:
    N(D) = N0 * f(mu) * (D/D0)**mu * exp(-(3.67+mu)*D/D0)
    f(mu) = 6/(4**4) * (4+mu)**(mu+4)/Gamma(mu+4)

    Attributes:
        Dm: the mean volume diameter, defaults to 0.001 m
        N0: the intercept parameter, defaults to 8e6 m^-4
        mu: the shape parameter, defaults to 0
        D_max: the maximum diameter to consider (defaults to 3*D0 when
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, Dm=1.0*1e-3, N0=8e6, mu=0.0, D_min = 0, D_max=None):
        D_max = Dm*3 if D_max is None else D_max
        super().__init__(D_min, D_max)
        self.Dm = Dm
        self.mu = mu
        self.N0 = N0
        fmu =  6.0/(4**4) * (4+mu)**(mu+4)/gamma(mu+4)
        h = lambda x: fmu * x**mu * np.exp(-(mu+4)*x)
        self.func = lambda D: N0*h(D/Dm)
class NormalizedGeneralizedGammaPSD(PSD):
    """ Constructor of the Gamma particle size distribution (PSD) of the form:
    N(D) = N0 * h_gg(mu,c,x)
    where N0 = M_i^((j+1)/(j-1))M_j^((i+1)/(i-j))
    Dm = (M_j/M_i)^(1/(j-i))
    h(x) = c*Gamma_i^((j+c*mu)/(i-j))*Gamma_j^((-i-c*mu)/(i-j))*x^(c*mu-1)*exp(-(Gamma_i/Gamma_j)^(c/(i-j))*x^c
    x = D/D_m
    Gamma_i = Gamma(mu+i/c)
    D is assumed to be in [m]
    i =3, j= 4
    

    Attributes:
        N0: the median volume diameter, defaults to 0.001 m
        Dm: the intercept parameter, defaults to 8e6 m^-4
        mu: the shape parameter, defaults to 1
        c: the other shape parameter, defaults to 1
        D_max: the maximum diameter to consider (defaults to 3*D0 when
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, Dm=1.0*1e-3, N0=8e6, mu=1.0, c=1., D_min = 0, D_max=None):
        D_max = Dm*3 if D_max is None else D_max
        super().__init__(D_min, D_max)
        self.Dm = Dm
        self.mu = mu
        self.N0 = N0
        self.c = c
        i, j = 3, 4
        G_i = gamma(mu+i/c)
        G_j = gamma(mu+j/c)
        h = lambda x: c * G_i**((j+c*mu)/(i-j)) * G_j**((-i-c*mu)/(i-j)) * x**(c*mu-1) * np.exp(-(G_i/G_j)**(c/(i-j)) * x**c)
        self.func = lambda D: N0*h(D/Dm)



class BinnedPSD(PSD):
    """Binned gamma particle size distribution (PSD).
    
    Callable class to provide a binned PSD with the given bin edges and PSD
    values.

    Args (constructor):
        The first argument to the constructor should specify n+1 bin edges, 
        and the second should specify n bin_psd values.        
        
    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters outside the bins.
    """
    def __init__(self, D_bin_edges, N):
        D_max = np.nanmax(D_bin_edges)
        D_min = np.nanmin(D_bin_edges)
        super().__init__(D_min, D_max)
        self.func = lambda D: stairs(D,D_bin_edges, N)