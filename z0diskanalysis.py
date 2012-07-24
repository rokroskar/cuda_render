import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pynbody, sys, math
from pynbody.analysis import profile, angmom, halo
from pynbody import units, config
import pynbody.filt as f
import diskfitting

from pynbody.analysis import luminosity as lum
import os, glob, pickle

disk = pynbody.filt.Disc('30 kpc','3 kpc')
bins = 50

s = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr_diff_coeff0.05/10/12M_hr_diff_coeff0.05.01000')
#h = s.halos()
pynbody.analysis.angmom.faceon(s)

diskstars = s.star[disk]
n_left = len(diskstars)
starsperbin = n_left / bins
print "stars per bin: "+str(starsperbin)

hzs = []
rexps = []
hzerrs = []
hrerrs = []
ages = []
ofes = []
fehs = []
sigmavzs = []

tfile = s._filename
n_done=0
i=0
while n_left>0:
    print i
    n_block = min(n_left,starsperbin)
    n_left -= n_block
    thesestars = diskstars[n_done:n_done+n_block]

    rmin = '4 kpc'
    rmax = '10 kpc'
    zmin = '0 kpc'
    zmax = '4 kpc'

    hr,hz, fitnum = diskfitting.two_exp_fit(thesestars,rmin=rmin, rmax=rmax, 
                                            zmin=zmin,zmax=zmax)
    if np.isfinite(hr):
        print "hr: %g, hz: %g"%(hr,hz)
        hzs.append(hz)
        rexps.append(hr)
        
        print "Using emcee to find errors"
        hrerr, hzerr = diskfitting.mcerrors(thesestars,[hr,hz],rmin=rmin,
                                            rmax=rmax,zmin=zmin,zmax=zmax)

        print "hrerr: %g, hzerr: %g"%(hrerr,hzerr)
        hzerrs.append(hzerr)
        hrerrs.append(hrerr)

        vertstarprof = profile.VerticalProfile(thesestars, 4, 10, 4,ndim=2,
                                               nbins=20)
        radstarprof = profile.Profile(thesestars,min=rmin,max=rmax,nbins=20)

        sigmavzs.append(diskfitting.velocity_dispersion(thesestars))
        
        age = np.mean(thesestars['age'].in_units('Gyr'))
        ages.append({'mean':age,
                    'dispersion':np.std(thesestars['age'].in_units('Gyr'))})
        ofe = np.mean(thesestars['ofe'])
        ofes.append({'mean':ofe,'dispersion':np.std(thesestars['ofe'])})
        feh = np.mean(thesestars['feh'])
        fehs.append({'mean':feh,'dispersion':np.std(thesestars['feh'])})

        diskfitting.plot_two_profiles(vertstarprof,radstarprof,
                                      hz=hz,hr=hr,units='Msol pc^-2',
                                      outfig=tfile+'.profs%02d'%i,
          title="OFe: %2.2f, FeH: %2.2f, age: %2.1f Gyr, N:%d"%(ofe,feh,age,fitnum))
        i+=1

    n_done+=n_block
    
    
pickle.dump({'hz':np.array(hzs), 'rexp': np.array(rexps), 
             'hzerr':np.array(hzerrs), 'rexperr': np.array(hrerrs),
             'sigmavz': np.array(sigmavzs), 'age':ages, 'ofe':ofes,'feh':fehs},
            open(tfile+'.z0agedecomp.dat','w'))
