import numpy as np
import pynbody as pyn
import pylab as plt
import pynbody.units as units
import multiprocessing

def plot_central(sim, clear = True, qty='rho', units = 'm_p cm^-3', width = None, **kwargs) :
    
    if width is None : 
        width = np.round(sim['r'][bh_index(sim)[0]]*5.0)

        if width == 0 :
            width = sim['r'][bh_index(sim)[0]]*5.0
        
    pyn.plot.image(sim.g,width=width,qty=qty,units=units, clear = clear, **kwargs)
    #overplot_bh(sim)
    plt.ylim((-width/2., width/2.))
    plt.xlim((-width/2., width/2.))   

def bh_index(sim) :
    return np.array(np.where(sim['mass'] == sim['mass'].max())).flatten()


def overplot_bh(sim):
    plt.plot(sim[bh_index(sim)]['x'],sim[bh_index(sim)]['y'], 'ro')
    

def plot_jeansratio(sim):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.hist(np.log10(sim.g['mjeans'].in_units('Msol')/sim.g['mass'].in_units('Msol')),
             histtype='step', bins=100, color = 'blue', log = True, label = 'thermal')
    plt.hist(np.log10(sim.g['mjeans_turb'].in_units('Msol')/sim.g['mass'].in_units('Msol')),
             histtype='step', bins=100, color = 'green', label = 'turbulent')
    plt.xlabel(r'$log(M_{jeans}/M_{particle})$')
    plt.legend()

    plt.subplot(1,2,2)
            
    plt.hist(np.log10(sim.g['ljeans'].in_units('kpc')/sim.g['smooth'].in_units('kpc')),
             histtype='step', bins=100, color = 'blue', log = True, label = 'thermal')
    plt.hist(np.log10(sim.g['ljeans_turb'].in_units('kpc')/sim.g['smooth'].in_units('kpc')),
             histtype='step', bins=100, color = 'green', label = 'turbulent')
    plt.xlabel(r'$log(\lambda_{jeans}/h_s)$')

def quick_plots(sim):
    plt.figure(figsize=(12,12))
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
    sim.g['rho'].convert_units('m_p cm^-3')
    
    pyn.analysis.halo.center(sim,ind=bh_index(sim),mode='ind')
    
    plt.subplot(2,2,1)

    plot_central(sim, clear = False, units='m_p cm^-3')
    plt.subplot(2,2,2)
    plot_central(sim, clear = False, units='m_p cm^-2')
    plt.subplot(2,2,3)
    sim.rotate_x(90)
    plot_central(sim, clear = False, units='m_p cm^-2')    
    plt.subplot(2,2,4)
    pyn.plot.rho_T(sim, clear = False, t_range = [1,7], rho_range = [-7,10])


def smbh_orbits(output=False, processes = multiprocessing.cpu_count()/4, test=False):
    import glob, orbits

    flist = glob.glob('?/*.00???')
    flist.sort()
    print flist[0:10]

    s = pyn.load(flist[0])
    inds = bh_index(s.d)

    assert(len(inds) == 2)

    pos, vel, mass, t = orbits.trace_orbits_parallel(flist, inds, processes, family='dark', test=test)

    dpos = np.diff(pos,axis=1).squeeze()

    r = pyn.array.SimArray(np.sqrt(np.sum(dpos**2,axis=1)),'kpc')
    t = pyn.array.SimArray(t, 's kpc km^-1').in_units('Myr')


    plt.plot(t - t[0], r)
    plt.plot(t - t[0], r, 'or')
    plt.xlabel('time [Myr]')
    plt.ylabel('r [kpc]')

    if output:
        np.savez('bh_orbit',t=t,r=r, pos = pos)

    return t, r, pos

def filelist() : 
    import glob
    import sys

    flist = glob.glob('?/*.00???')
    flist.sort()

    for f in flist : 
        stemp = pyn.load(f,only_header=True)
        print>>sys.stderr, f, stemp.properties['time'].in_units('Myr')


def central_gas(path, radius=0.5, fig = None) :
    import glob

    if path[-1] is not "/" : path = path+"/"

    flist = glob.glob(path+"*.00???")
    flist.sort()

    cen_mass = pyn.array.SimArray(np.empty(len(flist),dtype='float'),'Msol')
    time = pyn.array.SimArray(np.empty(len(flist),dtype='float'),'Gyr')

    for i, filename in enumerate(flist) : 
        s = pyn.load(filename)

        bhind = bh_index(s)
        sph1 = pyn.filt.Sphere(radius,s[bhind[0]]['pos'].flatten())
        sph2 = pyn.filt.Sphere(radius,s[bhind[1]]['pos'].flatten())
        
        cen_mass[i] = s[sph1 or sph2].g['mass'].sum().in_units('Msol')
        time[i] = s.properties['time'].in_units('Gyr')
        
    if fig is None : 
        fig = plt.figure()
        plt.xlabel('time [Gyr]')
        plt.ylabel('mass $M_{\odot}$')

    plt.plot(time,cen_mass.in_units('Msol'))
    
        
    return time,cen_mass


def plot_sequence() : 
    
    from matplotlib.backends.backend_pdf import PdfPages

    pp = PdfPages('smbh_plots.pdf')

    flist = ['551','592','601','612','634','676']

    basename = 'gas_merger0.1_thr10_Rx8_highSFthresh.00'

    for i in range(len(flist)) : 
        name = "{0}{1}".format(basename,flist[i])
        sim = pyn.load(name)

        plt.figure(figsize=(12,12))
        plt.subplots_adjust(wspace=0.2,hspace=0.3)
        sim.g['rho'].convert_units('m_p cm^-3')
        
        pyn.analysis.halo.center(sim,ind=bh_index(sim),mode='ind')
    
        plt.subplot(2,2,1)

        plot_central(sim, clear = False, units='m_p cm^-3',vmin=-1,vmax=5)
        plt.subplot(2,2,2)

        plot_central(sim, clear = False, units='m_p cm^-2',vmin=20,vmax=25)
        plt.subplot(2,2,3)

        sim.rotate_x(90)
        plot_central(sim, clear = False, units='m_p cm^-2',vmin=20,vmax=25)    
        plt.subplot(2,2,4)

        pyn.plot.rho_T(sim, clear = False, t_range = [1,7], rho_range = [-7,7])

        pp.savefig()

    pp.close()


def sfh_orbit(path):
    import scipy.interpolate as interpolate
    if path[-1] is not "/" : path = path+"/"
    filename = path + path[:-1]+".00650"

    orbit = np.load(path+'bh_orbit.npz')
    s = pyn.load(filename)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    sfh, bins = pyn.plot.sfh(s,trange=(4.9,5.1),nbins=100,subplot=ax1,clear=False)
    ax2 = ax1.twinx()
    ax2.plot(orbit['t']/1000., orbit['r'],'r--')
    ax1.set_ylabel('SFR [M$_{\odot}$~yr$^{-1}$]')
    ax2.set_ylabel('separation [kpc]')
    f = interpolate.interp1d(orbit['t']/1000.0,orbit['r'])
    #points = (4.938, 4.979, 4.988, 4.997, 5.02, 5.061)
    #ax2.plot(points, f(points), 'og')
    ax2.semilogy()
    ax1.set_xlabel('t [Gyr]')
    ax1.set_xlim(4.9, 5.062)
    fig.savefig("sfh_orbit.pdf", format='pdf')

def overplot_clump_centers(s,clumps,rmax,massmin) : 


    # get snapshot center

    cent = pyn.analysis.halo.center(s.g,mode='hyb',retcen=True)
    print cent
    # recenter 

    if any(abs(cent) > 1e-10) : 
        s['pos'] -= cent
        clumps['pos'] -= cent
        pyn.plot.image(s.g,width=rmax*2,av_z=True)
        

    # else the snapshot was already centered
    
    center_clumps = np.where((clumps['r'] < rmax) & (clumps['mass'] > massmin))[0]
    
    print 'Number of clumps in the center = %d'%len(center_clumps)

    # iterate through the clumps of interest and overplot their positions

    for clump_ind in center_clumps : 
        
        clump = s[np.where(s['grp'] == clump_ind+1)[0]]

        clump_center = pyn.analysis.halo.center(clump,mode='ssc',retcen=True)

        cir = plt.Circle((clump_center[0],clump_center[1]), radius=clumps[clump_ind]['eps'],  edgecolor='y', fill = False)
        plt.gca().add_patch(cir)

#        plt.plot(clump_center[0], clump_center[1], 'ro')

    plt.xlim(-rmax,rmax)
    plt.ylim(-rmax,rmax)

    
