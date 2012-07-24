import numpy as np
import matplotlib.pylab as plt

##################################
# fitting functions
#
def two_expo(x,p) : 
    return p[0]*np.exp(-x/p[1]) + p[2]*np.exp(-x/p[3])

def two_sech2(x,p) : 
    return p[0]*sech(-x/p[1])**2 + p[2]*sech(-x/p[3])**2

def sech(x) : 
    return 1/np.cosh(x)

def sech2(x,p) : 
    return p[0]*sech(-x/p[1])**2

def expo(x,p) : 
    return p[0]*np.exp(-x/p[1])

###################################

def overplot_fit(p,func) : 
    x = np.linspace(0,p[1]*10,100)
    plt.plot(x,func(x,p), '--')

def fit_profile(prof,func,p0,units,xmin=0,xmax=10) : 
    from scipy import optimize 

    fitfunc = lambda p, x : func(x,p)
    errfunc = lambda p, x, y, err : (y-fitfunc(p,x))/err

    ind = np.where((prof['rbins'] > xmin)&(prof['rbins'] <= xmax))[0]

    r = np.array(prof['rbins'])[ind]
    den = np.array(prof['density'].in_units(units)[ind])
    err = den/np.sqrt(prof['n'][ind])

    p1, res = optimize.leastsq(errfunc, p0, args = (r,den,err))

    red_chisq = sum((den - func(r,p1))**2/err**2)/(len(r)-len(p1)-1)

    return p1, red_chisq
