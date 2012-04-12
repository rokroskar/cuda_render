"""

isolated.py
===========


a set of routines for analysing an isolated run


Rok Roskar
University of Zurich

"""


import pynbody
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

    for i, arr in enumerate(['x','y','z']) : 
        #spl = sp.interpolate.interp1d(data['times'], data['cofm'][:,i],kind='linear',bounds_error=False)
 #        spl = sp.interpolate.UnivariateSpline(data['times'], data['cofm'][:,i])
        sim[arr+'form'] -= sp.interp(sim['tform'],data['times'],data['cofm'][:,i])
                   
    sim['rform'] = np.sqrt(np.sum(sim['posform']**2,axis=1))


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


def two_expo(x,p) : 
    return p[0]*np.exp(x/p[1]) + p[2]*np.exp(x/p[3])


def two_sech2(x,p) : 
    return p[0]*np.sech(x/p[1])**2 + p[2]*np.sech(x/p[3])**2

def fit_vertical_profile(prof,zmin=0,zmax=3,func=two_expo) : 
    from scipy import optimize 

    fitfunc = lambda p, x : func(x,p)
    errfunc = lambda p, x, y : fitfunc(p,x) - y

    # initial guesses 
    p0 = [1.0,0.3,.1,1.0]

    p1, done = optimize.leastsq(errfunc, p0, 
                                args = (prof['rbins'], 
                                        prof['density'].in_units('Msol pc^-3')))


    
    
