import numpy as np
import pynbody
import smbh
import matplotlib.pylab as plt

def make_eo_fo_clumps_figure(s,h) : 

    fig = plt.figure(figsize=(16,11))
    cmap = plt.cm.Blues_r

    # face-on far 
    ax = fig.add_subplot(2,2,1)
    pynbody.plot.image(s.g,width=5,units='Msol kpc^-2',cmap=cmap,subplot=ax)

    # face-on close

    ax = fig.add_subplot(2,2,2)
    pynbody.plot.image(s.g,width=.25,units='Msol kpc^-2',cmap=cmap,subplot=ax)
    plt.ylabel('$z/\mathrm{kpc}$')
    smbh.overplot_clump_centers(s,h,.125,2)
    smbh.overplot_bh(s.d)

    # edge-on far
    
    s.rotate_x(-90)
    ax = fig.add_subplot(2,2,3)
    pynbody.plot.image(s.g,width=5,units='Msol kpc^-2', cmap=cmap,subplot=ax)

    # edge-on close

    ax = fig.add_subplot(2,2,4)
    pynbody.plot.image(s.g,width=.25,units='Msol kpc^-2', cmap=cmap,subplot=ax)
    plt.ylabel('$z/\mathrm{kpc}$')
    smbh.overplot_clump_centers(s,h,.125,2)
    smbh.overplot_bh(s.d)

    s.rotate_x(90)

def make_r_z_figure(path): 

    orbit = np.load(path+'/bh_orbit.npz')

    fig = plt.figure(figsize=(10,15))

    ax = fig.add_subplot(2,1,1)

    plt.plot(orbit['t'], orbit['r'])

    plt.xlabel('$t$ [Myr]')
    plt.ylabel('$R_{sep}$ [kpc]')
    
    plt.semilogy()

    ax = fig.add_subplot(2,1,2)
    
    plt.plot(orbit['t'], orbit['pos'][:,0,2])
    plt.plot(orbit['t'], orbit['pos'][:,1,2])

    plt.xlabel('$t$ [Myr]')
    plt.ylabel('$z$ [kpc]')

    
    
