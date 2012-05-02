"""

isolated.py
===========


a set of routines for analysing an isolated run


Rok Roskar
University of Zurich

"""


import pynbody
from pynbody.analysis.angmom import faceon
from pynbody.analysis.profile import Profile, VerticalProfile
import spiral_structure as ss
import numpy as np
import warnings
import scipy as sp
import parallel_util
from parallel_util import interruptible, run_parallel
import matplotlib.pylab as plt 

#try: 
#    from IPython.parallel import Client
#    lv = Client().load_balanced_view()
#except : 
#    warnings.warn("Parallel interface not able to start -- using serial functions", RuntimeWarning)


def get_rform(sim) : 
    """ 

    Determine the formation radius of star particles after reading 
    the starlog file and the center of mass information

    """


    import os.path
    
    base = sim

    while hasattr(base, 'base') :
        base = base.base

    path = os.path.abspath(os.path.dirname(base.filename))
    path += '/../'
    data = np.load(path + 'cofm.npz')

    if hasattr(sim,'base') : 
        up = sim.base
        while hasattr(up,'base') : up = up.base

    
    try: 
        for arr in ['x','y','z'] : 
            del(up[arr+'form'])
            
    except KeyError: 
        pass

    try : 
        sim['posform']
        starlog = True
    except KeyError:
        starlog = False
        sim['posform'] = sim['rform']
        sim['posform'].units = sim['x'].units
        del(sim['rform'])
            
    for i, arr in enumerate(['x','y','z']) : 
        #spl = sp.interpolate.interp1d(data['times'], data['cofm'][:,i],kind='linear',bounds_error=False)
        #        spl = sp.interpolate.UnivariateSpline(data['times'], data['cofm'][:,i]
        if starlog: pass
        else : sim[arr+'form'] = sim['posform'][:,i]

        sim[arr+'form'] -= sp.interp(sim['tform'],data['times'],data['cofm'][:,i])
        
    sim['posform'] = np.array([sim['xform'],sim['yform'],sim['zform']]).T
    sim['rform'] = np.sqrt(np.sum(sim['posform'][:,0:2]**2,axis=1))


def get_cofm(dir='./', filepattern='*/*.0????') : 
    """

    Generate a center of mass data file from all the outputs in a run

    **Optional Keywords**: 
    
       *dir*: base directory

       *filepattern*: the file pattern to search for

       *filelist*: list of filenames to process -- if specified, *dir* and 
          *filepattern* are ignored
          
    """


    filelist = glob.glob(filepattern)
    filelist.sort()
    
    times = pynbody.array.SimArray(np.empty(len(filelist)))
    #times.units = 'Gyr'
    cofms = pynbody.array.SimArray(np.empty((len(filelist),3)))
    #cofms.units = 'kpc'
    
    
    for i,name in enumerate(filelist) : 
        times[i], cofms[i] = get_cofm_single_file(name)
        
    np.savez('cofm.npz', cofm = cofms, times = times)
    

def get_cofm_parallel(dir='./', filepattern='*/*.0????', filelist = None, block = True, procs = pynbody.config['number_of_threads']) : 
    """

    A parallel version of get_cofm() -- uses the IPython load balanced view
    to farm out the work. 

    Generate a center of mass data file from all the outputs in a run

    **Optional Keywords**: 
    
       *dir*: base directory

       *filepattern*: the file pattern to search for

       *filelist*: list of filenames to process -- if specified, *dir* and 
          *filepattern* are ignored
          
    """

    import glob 
    
    if filelist is None: 
        filelist = glob.glob(dir+filepattern)

    if len(filelist) == 0 : 
        raise RuntimeError, "No files found matching " + dir + filepattern

    filelist.sort()

    times = np.empty(len(filelist))
    cofms = np.empty((len(filelist),3))

    res = run_parallel(get_cofm_single_file, filelist, [],processes=procs)
    
    if block : 
        res = sorted(res)

        for i, x in enumerate(res) : 
            times[i] = res[i][0]
            cofms[i] = res[i][1]

        np.savez(dir+'cofm.npz', cofm = cofms, times = times)
        
    else : 
        return res, filelist

@interruptible
def get_cofm_single_file(args) : 
    """

    Return the center of mass of a single file 

    **Input**:

       *name*: filename

    """

    name = args[0]

    s = pynbody.load(name)

    time = s.properties['a']
    cofm = pynbody.analysis.halo.center(s, retcen=True)

    return time, np.array(cofm)


def plot_dist_mean(x, y, mass, **kwargs) : 
    """
    
    Plots the KDE (or 2D histogram) of the data including points
    showing the mean trend. 

    """
    import matplotlib.pylab as plt

    g, xs, ys = pynbody.plot.generic.gauss_kde(x,y,mass=mass, **kwargs)
    
    
    range = kwargs.get('x_range', None)

    h1, bins = np.histogram(x, weights=y*mass,range=range)
    h1_mass, bins = np.histogram(x,weights=mass,range=range)

    h1 /= h1_mass
    bins = .5*(bins[:-1]+bins[1:])

    plt.plot(bins, h1, 'oy')


def plot_means(x, y, mass, range, *args, **kwargs) : 
    import matplotlib.pylab as plt

    h1, bins = np.histogram(x, weights=y*mass,range=range)
    h1_mass, bins = np.histogram(x,weights=mass,range=range)

    h1 /= h1_mass
    bins = .5*(bins[:-1]+bins[1:])

    plt.plot(bins, h1, *args, **kwargs)


def one_d_kde(x, weights=None, range=None, gridsize=100):

    """
    
    generate a 1D weighted KDE 

    """


    import scipy as sp
    import numpy as np
    import scipy.sparse

    nx = gridsize

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(x.size)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size'
                    ' as input x & y arrays!')


    # Default extents are the extent of the data
    if range is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = map(float, range)
        
    dx = (xmax - xmin) / (nx - 1)

    xi = np.vstack(x).T
    xi -= xmin
    xi /= dx
    xi = np.floor(xi).T

    grid = sp.sparse.coo_matrix((weights, xi), shape=nx).toarray()

    cov = np.cov(xi)

    scotts_factor = np.power(x.size, -1.0/4) 

    std_devs = np.diag(np.sqrt(cov))

    # this next line sets the size of the kernel in pixels

    kern_nx = np.round(scotts_factor * 2 * np.pi * std_devs)

    inv_cov = np.linalg.inv(cov*scotts_factor**2)

    xx = np.arange(kern_nx, dtype=np.float) - kern_nx/2.0\

    return kern_nx, inv_cov, xx

##################################
# fitting functions
#
def two_expo(x,p) : 
    return p[0]*np.exp(-x/p[1]) + p[2]*np.exp(-x/p[3])

def two_sech2(x,p) : 
    return p[0]*sech(-x/p[1])**2 + p[2]*sech(-x/p[3])**2

def sech(x) : 
    return 1/np.cosh(x)

def expo(x,p) : 
    return p[0]*np.exp(-x/p[1])

###################################

def overplot_fit(p,func) : 
    x = np.linspace(0,p[1]*5,100)
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

@interruptible
def single_profile_fits(x) : 
    from pynbody.analysis.profile import Profile, VerticalProfile

    filename, merger = x

    s = pynbody.load(filename)
    pynbody.analysis.angmom.faceon(s)
    
    if merger : s = s.s[np.where(s.s['mass']>.1)[0]]

    p  = Profile(s.s,min=0,max=15,nbins=30)
    fit_r, chsq = fit_profile(p,expo,[1e9,3],'Msol kpc^-2',3,6)

    # make vertical profiles at 1, 2, 3 scalelengths
    pv1 = VerticalProfile(s.s,fit_r[1]*.8,fit_r[1]*1.2,3.0,nbins=30)
    pv2 = VerticalProfile(s.s,fit_r[1]*1.8,fit_r[1]*2.2,3.0,nbins=30)
    pv3 = VerticalProfile(s.s,fit_r[1]*2.8,fit_r[1]*3.2,3.0,nbins=30)

    fit_v1, chsq1 = fit_profile(pv1,two_sech2,[0.1,.2,0.01,.5],'Msol pc^-3',0,3)
    fit_v2, chsq2 = fit_profile(pv2,two_sech2,[0.1,.2,0.01,.5],'Msol pc^-3',0,3)
    fit_v3, chsq3 = fit_profile(pv3,two_sech2,[0.1,.2,0.01,.5],'Msol pc^-3',0,3)
    
    return s.properties['time'].in_units('Gyr'),fit_r, fit_v1, fit_v2, fit_v3

def disk_structure_evolution(flist, merger=False) : 
    
    times = np.zeros(len(flist))
    rfits = np.zeros((len(flist),2))
    v1fits = np.zeros((len(flist),4))
    v2fits = np.zeros((len(flist),4))
    v3fits = np.zeros((len(flist),4))

    
    res = run_parallel(single_profile_fits, flist, [merger], processes=10)

    for i in xrange(len(flist)) : 
        times[i]  = res[i][0]
        rfits[i]  = res[i][1]
        v1fits[i] = res[i][2]
        v2fits[i] = res[i][3]
        v3fits[i] = res[i][4]

    return times, rfits, v1fits, v2fits, v3fits

    
def plot_profile_fit(filename, merger, rmin=3, rmax=6) : 
    
    s = pynbody.load(filename)
    faceon(s)
    if merger : s = s.s[np.where(s.s['mass']>.1)[0]]
    p = Profile(s.s,min=0,max=15,nbins=30)
    fit, chsq = fit_profile(p,expo,[1e9,3],'Msol kpc^-2',rmin,rmax)

    plt.figure()
    plt.plot(p['rbins'],p['density'].in_units('Msol kpc^-2'))
    overplot_fit(fit,expo)
    print fit, chsq
    plt.semilogy()
