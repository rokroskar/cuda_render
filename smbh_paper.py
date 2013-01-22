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

    
    
def load_snapshots(flist = None):
    if flist is None : 
        flist = ['6/gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc.00614', 
                 '6/gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc.00660',
                 '6/gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc.00691',
                 '22/gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc.02275']
    slist = []

    for f in flist : 
        slist.append(pynbody.load(f))
 
    return slist



def make_morph_evol_figure(slist,width=1.0,overplot_bh = True) : 
    fig = plt.figure(figsize=(12,12))
    cmap = plt.cm.Blues_r

    for i,s in enumerate(slist) : 

        ax = fig.add_subplot(4,4,i*4+1)
        pynbody.plot.image(s.g,width=width,units='Msol kpc^-2',
                           cmap=cmap,subplot=ax,show_cbar=False)
        smbh.overplot_bh(s)
        plt.xlim(-width/2.0,width/2.0)
        plt.ylim(-width/2.0,width/2.0)
        if i == 0:
            plt.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                         arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
            plt.annotate("%.0f kpc"%width, xy=(0.45,0.065), color ="white",fontsize='smaller', 
                         xycoords = 'axes fraction')

        plt.annotate('$t = %0.0f$ Myr'%(s.properties['time'].in_units('Myr')-slist[0].properties['time'].in_units('Myr')), 
                     (0.1,0.85), color='white', fontweight='bold', 
                     xycoords = 'axes fraction')
        utils.clear_labels(ax)
        s.rotate_x(-90)


        ax = fig.add_subplot(4,4,i*4+2)
        pynbody.plot.image(s.g,width=width,units='Msol kpc^-2',
                           cmap=cmap,subplot=ax,show_cbar=False)
        s.rotate_x(90)
        smbh.overplot_bh(s)
        plt.xlim(-width/2.0,width/2.0)
        plt.ylim(-width/2.0,width/2.0)
        utils.clear_labels(ax)

        ax = fig.add_subplot(4,4,i*4+3)
        pynbody.plot.image(s.g,width=width*10.0,units='Msol kpc^-2',
                           cmap=cmap,subplot=ax,show_cbar=False)
        s.rotate_x(-90)
        utils.clear_labels(ax)
        plt.xlim(-width*10/2.0,width*10/2.0)
        plt.ylim(-width*10/2.0,width*10/2.0)
        if i == 0:
            plt.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                         arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
            plt.annotate("%.0f kpc"%(float(width*10)), xy=(0.45,0.065), color ="white",fontsize='smaller', 
                         xycoords = 'axes fraction')

        ax = fig.add_subplot(4,4,i*4+4)
        pynbody.plot.image(s.g,width=width*10.0,units='Msol kpc^-2',
                           cmap=cmap,subplot=ax,show_cbar=False)
        s.rotate_x(90)
        utils.clear_labels(ax)
        plt.xlim(-width*10/2.0,width*10/2.0)
        plt.ylim(-width*10/2.0,width*10/2.0)
    
    plt.subplots_adjust(hspace=0.1)
    

def central_profile_figure(slist):

    d = pynbody.filt.Disc(1, .5)

    fig, axs = plt.subplots()
    fig2, axs2 = plt.subplots()

    labels = ['nomet','met',]
    colors = ['b','r']
    for i,s in enumerate([slist[0],slist[2]]) : 
        pg = pynbody.analysis.profile.Profile(s.g[d], nbins=50, type = 'log',min=.001,max=1)
        ps = pynbody.analysis.profile.Profile(s.s[d], nbins=50, type = 'log',min=.001,max=1)

        axs.plot(pg['rbins'].in_units('pc'),pg['density'].in_units('Msol kpc^-2'),colors[i],label=labels[i])
        axs.plot(ps['rbins'].in_units('pc'),ps['density'].in_units('Msol kpc^-2'),'%s--'%colors[i])

        

    
    axs.legend()
    axs2.legend()
    axs.set_xlabel('R [kpc]')
    axs2.set_xlabel('R [kpc]')
    axs.set_ylabel(r'$M_{\odot}$')
    axs2.set_ylabel(r'$v, v_{disp} \mathrm{km/s}$')
    axs.loglog()
    axs2.semilogx()


def central_vsigma(slist) : 
    d = pynbody.filt.Disc(1,.1)

    fig, axs = plt.subplots(2,1, figsize=(8,12))

    labels = ['nomet','met',]
    colors = ['b','r']

    for i, s in enumerate([slist[0], slist[2]]) : 
        pg = pynbody.analysis.profile.Profile(s.g[d], nbins=50, type = 'log',min=.001,max=1)
        ps = pynbody.analysis.profile.Profile(s.s[d], nbins=50, type = 'log',min=.001,max=1)
    

        axs[0].plot(pg['rbins'].in_units('pc'),pg['speed']/pg['speed_disp'], colors[i], label=labels[i])
        
        axs[1].plot(ps['rbins'].in_units('pc'),ps['speed']/ps['speed_disp'], colors[i])


    for ax in axs: 
        ax.semilogx()
        ax.set_xlabel(r'R [pc]')
        ax.set_ylabel(r'$v/\sigma$')
    axs[0].legend()
    axs[0].set_title('gas')
    axs[1].set_title('stars')
    
def central_velocity_profile(slist) : 
    d = pynbody.filt.Disc(1,.5)

    fig, axs = plt.subplots(2,1, figsize=(8,12))

    labels = ['nomet','met',]
    colors = ['b','r']

    for i, s in enumerate([slist[0], slist[2]]) : 
        pg = pynbody.analysis.profile.Profile(s.g[d], nbins=50, type = 'log',min=.001,max=1)
        ps = pynbody.analysis.profile.Profile(s.s[d], nbins=50, type = 'log',min=.001,max=1)
    

        axs[0].plot(pg['rbins'].in_units('pc'),pg['speed'], colors[i], label=labels[i])
        axs[0].plot(pg['rbins'].in_units('pc'),pg['speed_disp'], '%s--'%colors[i])


        axs[1].plot(ps['rbins'].in_units('pc'),ps['speed'], colors[i])
        axs[1].plot(ps['rbins'].in_units('pc'),ps['speed_disp'],'%s--'%colors[i])


    for ax in axs: 
        ax.semilogx()
        ax.set_ylim(0,1000)
        ax.set_xlabel(r'R [pc]')
        ax.set_ylabel(r'$v, v_{disp} \mathrm{~km/s}$')
    axs[0].legend()
    axs[0].set_title('gas')
    axs[1].set_title('stars')


def vsigma_vs_time(dir, times = [5003.1, 5004, 5005]) : 
    flist = smbh.nearest_output(times,dir)

    fig, ax = plt.subplots()

    for f,t in zip(flist,times) : 
        s = pynbody.load(f)
        pynbody.analysis.halo.center(s,mode='hyb',verbose=True,min_particles = 100000)
        s.g['speed'] = np.sqrt(s.g['v2'])
        ind = np.where(s.g['rho'].in_units('m_p cm^-3') < 1e6)[0]
        p = pynbody.analysis.profile.Profile(s.g[ind],max = 1, min = .002, type = 'log', nbins=20)
        ax.plot(p['rbins'].in_units('pc'), p['speed']/p['speed_disp'], label = 't=%.2f Myr'%t)

    ax.set_xlabel('$R$ [pc]')
    ax.set_ylabel(r'$v/\sigma$')
    ax.legend()
    ax.semilogx()

 

def pdf_vs_time(dir, times = [5003.1,5004,5005]) : 
    from scipy.stats import gaussian_kde as kde
    from utils import shrink_sphere

    flist = smbh.nearest_output(times,dir)

    fig, ax = plt.subplots()

    rho = np.linspace(-2,12,100)

    for f, t in zip(flist,times) : 
        s = pynbody.load(f)
        s['pos'] -= utils.shrink_sphere(s,r = 5,verbose=True)
        pynbody.analysis.halo.vel_center(s)
        k = kde(np.log10(s.g['rho'].in_units('m_p cm^-3')[np.where(s.g['r'] < 0.5)[0]]))
#        ax.hist(np.log10(s.g['rho'].in_units('m_p cm^-3')),histtype='step',normed=True)
        ax.plot(rho, k(rho), label = 't=%.2f Myr'%t)
    ax.set_xlabel(r'$\rho$ [amu/cc]')
    ax.legend(loc='upper right')
                
        

def savefig(fname): 

    plt.savefig('smbh_p1_figs/%s.pdf'%fname,format='pdf',bbox_inches='tight')
    plt.savefig('smbh_p1_figs/%s.eps'%fname,format='eps',bbox_inches='tight')
