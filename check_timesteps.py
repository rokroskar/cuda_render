import pynbody
import pylab as pl
import numpy as np

if __name__ == '__main__' : 
    import sys

    pl.figure()

    s = pynbody.load(sys.argv[1])

    pynbody.analysis.halo.center(s)

    inner = pynbody.filt.Sphere("0.5 kpc")

    s['timestep'] = 0.175*np.sqrt(s['eps']/np.sqrt((s['accg']**2).sum(axis=1)))

    meantstep = s.g[inner]['timestep'].mean()

    pl.hist(s.g[inner]['timestep']/meantstep, bins=20, label='gas',alpha=.3,normed=True)
    pl.hist(s.g[inner]['dt']/meantstep, bins=20, label='sph',alpha=.3,normed=True)
    pl.hist(s.d[inner]['timestep']/meantstep, bins=20, label='dm',alpha=.3,normed=True)
    pl.hist(s.s[inner]['timestep']/meantstep, bins=20, label='star',alpha=.3,normed=True)
    
    pl.legend()
