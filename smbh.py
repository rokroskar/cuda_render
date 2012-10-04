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
    
    ax = plt.subplot(2,2,1)

    plot_central(sim, clear = False, units='m_p cm^-3', subplot=ax)
    ax = plt.subplot(2,2,2)
    plot_central(sim, clear = False, units='m_p cm^-2', subplot=ax)
    ax = plt.subplot(2,2,3)
    sim.rotate_x(90)
    plot_central(sim, clear = False, units='m_p cm^-2', subplot=ax)    
    ax = plt.subplot(2,2,4)
    pyn.plot.rho_T(sim.g, clear = False, t_range = [1,7], rho_range = [-7,10])


def smbh_orbits(dir = './', output=False, processes = 5, test=False):
    import glob, orbits

    if dir[-1] != '/' : dir += '/'

    flist = glob.glob(dir+'*/*.0????')
    flist.sort(key=lambda x: x[-5:])
    print flist[0:10]

    s = pyn.load(flist[0])
    inds = bh_index(s.d)

    assert(len(inds) == 2)

    pos, vel, mass, phi, t = orbits.trace_orbits_parallel(flist, inds, processes, family='dark', test=test)

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

def filelist(path='./', pattern = '?/*.00???') : 
    import glob
    import sys
    
    if path[-1] != '/' : path += '/'

    flist = glob.glob(path+pattern)
    flist.sort()

#    outfile = open('filelist', 'w')
    
    names = []
    times = []
    
    for f in flist : 
        stemp = pyn.load(f,only_header=True)
        print>>sys.stderr, f, stemp.properties['time'].in_units('Myr')
        names.append(f)
        times.append(stemp.properties['time'].in_units('Myr'))

    np.savez(path+'filelist', names = names, times = times)

def central_mass(path, radius=0.5, fig = None) :
    import glob

    if path[-1] is not "/" : path = path+"/"

    flist = glob.glob(path+"*.0????")
    flist.sort(key=lambda x: x[-5:])
    print path
    print flist

    cen_mass_g = pyn.array.SimArray(np.empty(len(flist),dtype='float'),'Msol')
    cen_mass_s = pyn.array.SimArray(np.empty(len(flist),dtype='float'),'Msol')
    time = pyn.array.SimArray(np.empty(len(flist),dtype='float'),'Gyr')

    for i, filename in enumerate(flist) : 
        s = pyn.load(filename)

        bhind = bh_index(s)
        sph1 = pyn.filt.Sphere(radius,s[bhind[0]]['pos'].flatten())
        sph2 = pyn.filt.Sphere(radius,s[bhind[1]]['pos'].flatten())
        
        cen_mass_g[i] = s[sph1 or sph2].g['mass'].sum().in_units('Msol')
        cen_mass_s[i] = s[sph1 or sph2].s['mass'].sum().in_units('Msol')
        time[i] = s.properties['time'].in_units('Myr')
        
    if fig is None : 
        fig = plt.figure()
        plt.xlabel('time [Myr]')
        plt.ylabel('mass $M_{\odot}$')

    plt.plot(time,cen_mass_g.in_units('Msol'),label='gas')
    plt.plot(time,cen_mass_s.in_units('Msol'),label='stars')
    
        
    return time,cen_mass_g, cen_mass_s


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

    
def nearest_output(time, dir = './') : 
    if dir[-1] != '/' : dir += '/'

    try : 
        flist = np.load(dir+'filelist.npz')
    except IOError: 
        filelist()
        flist = np.load(dir+'filelist.npz')

        
    time = np.array(time)


    if len(time.shape) == 0 : 
        return flist['names'][np.abs(flist['times'] - time).argmin()]
    else : 
        res = []
        for t in time : 
            res.append(flist['names'][np.abs(flist['times'] - t).argmin()])
        return res


    
def plot_orbit(dir='./', ax = None): 
    if dir[-1] != '/' : dir += '/'


    try : 
        data = np.load(dir+'bh_orbit.npz')
    except IOError : 
        smbh_orbits(dir)
        data = np.load(dir+'bh_orbit.npz')

    if ax is None : 
        fig, ax = plt.subplots()

    ax.plot(data['t'], data['r']*1e3)
    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('separation [pc]')
    ax.semilogy()
