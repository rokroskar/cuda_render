import numpy as np
import pynbody
import smbh
import matplotlib.pylab as plt
import utils

def make_eo_fo_clumps_figure(s,h) : 

    fig = plt.figure(figsize=(16,11))
    cmap = plt.cm.Blues_r

    # face-on far 
    ax = fig.add_subplot(2,2,1)
    pynbody.plot.image(s.g,width=5,units='Msol pc^-2',cmap=cmap,subplot=ax)

    # face-on close

    ax = fig.add_subplot(2,2,2)
    pynbody.plot.image(s.g,width=.25,units='Msol pc^-2',cmap=cmap,subplot=ax)
    plt.ylabel('$z/\mathrm{kpc}$')
    smbh.overplot_clump_centers(s,h,.125,2)
    smbh.overplot_bh(s.d)

    # edge-on far
    
    s.rotate_x(-90)
    ax = fig.add_subplot(2,2,3)
    pynbody.plot.image(s.g,width=5,units='Msol pc^-2', cmap=cmap,subplot=ax)

    # edge-on close

    ax = fig.add_subplot(2,2,4)
    pynbody.plot.image(s.g,width=.25,units='Msol pc^-2', cmap=cmap,subplot=ax)
    plt.ylabel('$z/\mathrm{kpc}$')
    smbh.overplot_clump_centers(s,h,.125,2)
    smbh.overplot_bh(s.d)

    s.rotate_x(90)

def make_r_z_figure(path): 

    orbit = np.load(path+'/bh_orbit.npz')

    fig = plt.figure(figsize=(10,15))

    ax = fig.add_subplot(2,1,1)

    plt.plot(orbit['t'], orbit['r']*1000.)

    plt.xlabel('$t$ [Myr]')
    plt.ylabel('$R_{sep}$ [pc]')
    
    plt.semilogy()

    ax = fig.add_subplot(2,1,2)
    
    plt.plot(orbit['t'], orbit['pos'][:,0,2]*1000.)
    plt.plot(orbit['t'], orbit['pos'][:,1,2]*1000.)

    plt.xlabel('$t$ [Myr]')
    plt.ylabel('$z$ [pc]')

    
    
def load_snapshots(): 
    slist= [pynbody.load('6/gas_merger0.1_thr10_Rx8_nometalcool_1pc.00633'), 
            pynbody.load('6/gas_merger0.1_thr10_Rx8_nometalcool_1pc.00660'),
            pynbody.load('7/gas_merger0.1_thr10_Rx8_nometalcool_1pc.00771'),
            pynbody.load('9/gas_merger0.1_thr10_Rx8_nometalcool_1pc.00921')]
            
    pynbody.analysis.halo.center(slist[0],mode='ind',ind=smbh.bh_index(slist[0]))
    for s in slist[1:] : 
        pynbody.analysis.halo.center(s.g,mode='hyb')

    return slist



def make_morph_evol_figure(slist) : 
    fig = plt.figure(figsize=(6.5,12))
    cmap = plt.cm.Blues_r

    for i,s in enumerate(slist) : 

        ax = fig.add_subplot(4,2,i*2+1)
        pynbody.plot.image(s.g,width=1,units='Msol kpc^-2',
                           cmap=cmap,subplot=ax,show_cbar=False)
        smbh.overplot_bh(s)
        plt.xlim(-.5,.5)
        plt.ylim(-.5,.5)
        if i == 0:
            plt.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                         arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
            plt.annotate("1 kpc", xy=(0.45,0.065), color ="white",fontsize='smaller', 
                         xycoords = 'axes fraction')

        plt.annotate('$t = %0.0f$ Myr'%s.properties['time'].in_units('Myr'), 
                     (0.1,0.85), color='white', fontweight='bold', 
                     xycoords = 'axes fraction')
        utils.clear_labels(ax)
        s.rotate_x(-90)
        ax = fig.add_subplot(4,2,i*2+2)
        pynbody.plot.image(s.g,width=1,units='Msol kpc^-2',
                           cmap=cmap,subplot=ax,show_cbar=False)
        s.rotate_x(90)
        utils.clear_labels(ax)
        smbh.overplot_bh(s)
        plt.xlim(-.5,.5)
        plt.ylim(-.5,.5)
        
    
    plt.subplots_adjust(hspace=0.1)
