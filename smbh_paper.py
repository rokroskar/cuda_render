import numpy as np
import pynbody
import smbh
import matplotlib.pylab as plt
import utils


paper_times = [5010, 5025, 5040, 5080]

after_merger = [5001,5003,5005,5009]

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

    rs = np.sqrt((orbit['pos'][:,:,0:2]**2).sum(axis=2))

    dr = np.sqrt(((orbit['pos'][:,0,:] - orbit['pos'][:,1,:])**2).sum(axis=1))

    fig, axs = plt.subplots(2,1,figsize=(10,15))

 #   axs[0].plot(orbit['t'], rs*1000.)
    axs[0].plot(orbit['t'], dr*1000.)

    axs[0].set_xlabel('$t$ [Myr]')
    axs[0].set_ylabel('separation [pc]')
    axs[0].semilogy()
    axs[0].set_xlim(orbit['t'].min(), orbit['t'].max())

    axs[1].plot(orbit['t'], orbit['pos'][:,0,2]*1000.)
    axs[1].plot(orbit['t'], orbit['pos'][:,1,2]*1000.)

    axs[1].set_xlabel('$t$ [Myr]')
    axs[1].set_ylabel('$z$ [pc]')
    axs[1].set_xlim(orbit['t'].min(), orbit['t'].max())

    # zrms of the stars at 5, 10, 50 pc
    zrms = [12, 20, 42]

    for zr in zrms :
        axs[1].plot([orbit['t'].min(),orbit['t'].max()],
                    [zr,zr],'k:')
    
def load_snapshots(flist = None, times = None):
    if flist is None : 
        if times is None: times = paper_times
        flist = map(smbh.nearest_output, times)
     
    return [pynbody.load(x) for x in flist]


def load_snapshot_sequence(dir='./', ntiles = 20, t1 = None, t2 = None):
    out1 = pynbody.load(smbh.nearest_output(0))
    out2 = pynbody.load(smbh.nearest_output(1e6))

    if t1 is None:
        t1 = out1.properties['time'].in_units('Myr')
    if t2 is None: 
        t2 = out2.properties['time'].in_units('Myr')

    times = np.linspace(t1,t2,ntiles)

    return [pynbody.load(x) for x in map(smbh.nearest_output, times)]
    

def center_snapshot(s, align=True, disk_size = '100 pc') : 
    if np.diff(s[smbh.bh_index(s)]['r'])[0] > 0.2 : 
            pynbody.analysis.halo.center(s,mode='ind',ind=smbh.bh_index(s),vel=False)
    else:
        pynbody.analysis.halo.center(s.g,mode='hyb',vel=False)

    pynbody.analysis.halo.vel_center(s.g,cen_size=.5)
    if align: pynbody.analysis.angmom.faceon(s.g,disk_size=disk_size, cen=[0,0,0])



def make_filmstrip_figure(slist): 

    f, axs = plt.subplots(len(slist)/2,4,figsize=(12,len(slist)/2*3.0))

    cmap = plt.cm.Greys_r
    
    width = .5

    for i,s in enumerate(slist) : 
        ax = axs.flatten()[i*2]

        pynbody.plot.image(s.g,width=width,units='Msol kpc^-2', subplot=ax, show_cbar=False, cmap=cmap, threaded=10, approximate_fast=False,vmin=8,vmax=12)
        ax.annotate('$t = %0.0f$ Myr'%(s.properties['time'].in_units('Myr')), 
                     (0.1,0.85), color='white', fontweight='bold', 
                     xycoords = 'axes fraction', fontsize=12)
        smbh.overplot_bh(s,ax)
        ax.set_xlim(-width/2.0,width/2.0)
        ax.set_ylim(-width/2.0,width/2.0)

        if i == 0: 
            ax.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
            ax.annotate("500 pc", xy=(0.38,0.065), color ="white",fontsize='smaller', 
                        xycoords = 'axes fraction')

        ax = axs.flatten()[i*2+1]
        s.rotate_x(90)
        pynbody.plot.image(s.g,width=width,units='Msol kpc^-2', subplot=ax, show_cbar=False, cmap=cmap, approximate_fast=False, threaded=10, vmin=8,vmax=12)
        smbh.overplot_bh(s,ax)
        ax.set_xlim(-width/2.0,width/2.0)
        ax.set_ylim(-width/2.0,width/2.0)
        s.rotate_x(-90)

    map(utils.clear_labels,axs.flatten())
    plt.subplots_adjust(hspace=.1,wspace=.1)

    # set the colorbar
    bb1 = axs.flatten()[3].get_position()
    bb2 = axs.flatten()[-1].get_position()
    cbax = f.add_axes([bb1.x1+.01,bb2.y0,0.02,bb1.y1-bb2.y0])
    cb1 = f.colorbar(axs.flatten()[-1].get_images()[0],cax=cbax)
    cb1.set_label(r'log($\Sigma$) [M$_{\odot}$ kpc$^{-2}$]',fontsize='smaller', fontweight='bold')


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

def make_ic_zoomin_figure(s): 
    from utils import clear_labels

    f, axs = plt.subplots(1,3,figsize=(9,3))
    widths = ['50 kpc','10 kpc','1 kpc']

    for width,ax in zip(widths,axs) : 
        if width==widths[-1] :
            s['pos'] -= s['pos'][smbh.bh_index(s)[0]]
        pynbody.plot.image(s.g,width=width,subplot=ax,show_cbar=False,av_z=True,cmap=plt.cm.Greys_r,resolution=1000)
        ax.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
        ax.annotate(width, xy=(0.38,0.065), color ="white",fontsize='smaller', 
                    xycoords = 'axes fraction')
        
        clear_labels(ax)

def make_zoomin_figure_single(s) : 
    from utils import clear_labels

    f, axs = plt.subplots(1,4,figsize=(12,3))
    widths = ['50 kpc','10 kpc','5 kpc','1 kpc']

    for width,ax in zip(widths,axs) : 
        pynbody.plot.image(s.g,width=width,subplot=ax,show_cbar=False,av_z=True,cmap=plt.cm.Greys_r)
        ax.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
        ax.annotate(width, xy=(0.38,0.065), color ="white",fontsize='smaller', 
                    xycoords = 'axes fraction')
        clear_labels(ax)
def make_zoomin_figure(slist) : 
    f, axs = plt.subplots(5,4,figsize=(8,10))
    cmap = plt.cm.Blues_r
    
    widths = 10.0/np.logspace(0,2,len(slist))

    for i,s in enumerate(slist) : 
        ax = axs[i]
        
        if np.diff(s[smbh.bh_index(s)]['r'])[0] > 0.2 : 
            pynbody.analysis.halo.center(s,mode='ind',ind=smbh.bh_index(s))
        else:
            pynbody.analysis.halo.center(s.g,mode='hyb')
                    
        for j in range(4) : 
            
            pynbody.plot.image(s.s,width=widths[j],units='Msol kpc^-2', subplot=ax[j], show_cbar=False, cmap=cmap, threaded=20)
        
        ax[0].annotate('$t = %0.0f$ Myr'%(s.properties['time'].in_units('Myr')-slist[0].properties['time'].in_units('Myr')), 
                       (0.1,0.85), color='white', fontweight='bold', 
                       xycoords = 'axes fraction', fontsize=12)
        
        
    map(utils.clear_labels,axs.flatten())

    for i,ax in enumerate(axs[-1]) : 
        ax.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='<->',color='white',linewidth=1.5))
        ax.annotate("%.2f kpc"%widths[i], xy=(0.3,0.065), color ="white",
                    fontsize=12, xycoords = 'axes fraction')

    plt.subplots_adjust(hspace=.1,wspace=.1)
        
    
    

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

    fig, axs = plt.subplots()#2,1, figsize=(8,12))

    for i, s in enumerate(slist) : 
        pg = pynbody.analysis.profile.Profile(s.g[d], nbins=10, type = 'log',min=.01,max=1)
        #ps = pynbody.analysis.profile.Profile(s.s[d], nbins=50, type = 'log',min=.01,max=1)
    

        axs.plot(pg['rbins'].in_units('pc'),pg['vt']/pg['vt_disp'], label="t = %.0f"%s.properties['time'].in_units('Myr'))
        #axs[1].plot(ps['rbins'].in_units('pc'),ps['vt']/ps['vt_disp'])


    axs.semilogx()
    axs.set_xlabel(r'$R$ [pc]')
    axs.set_ylabel(r'$v/\sigma$')
    axs.legend()
    #axs[0].set_title('gas')
    #axs[1].set_title('stars')
    
def central_vsigma_inclined(s,ax=None) : 

    d = pynbody.filt.Sphere(1)
    if ax is None:
        f,ax = plt.subplots()

    for angle in [0,20,45,60,89.9]  : 
        s.rotate_x(angle)
        s['absvz'] = np.abs(s['vz'])
        p = pynbody.analysis.profile.InclinedProfile(s.g,angle,nbins=10,max=1,min=.01,type='log')
        ax.plot(p['rbins'].in_units('pc'),p['absvz']/p['absvz_disp'],label='%.0f'%angle)
        s.rotate_x(-angle)
        del(s['absvz'])

    ax.semilogx()
    ax.set_xlabel('$R$ [pc]')
    ax.set_ylabel('$v_{\mathrm{los}}/\sigma$')
    ax.legend(prop=dict(size=12))


def central_properties_figure(times = [5010,5020,5050]):

    f,axs = plt.subplots(len(times),3,figsize=(3*len(times),8))

    for i,time in enumerate(times):
        s = pynbody.load(smbh.nearest_output(time))
        center_snapshot(s)
        s.physical_units()
        central_vsigma_inclined(s,ax=axs[i,0])
        s.rotate_x(45)
        pynbody.plot.image(s.g,width='1000 pc', av_z='rho', qty='temp',subplot=axs[i,1],show_cbar=False,vmin=4,vmax=7)
        pynbody.plot.image(s.g,width='1000 pc', av_z='rho', qty='vz',log=False,vmin=-350,vmax=350,subplot=axs[i,2],show_cbar=False)
        s.rotate_x(-45)
        
        
    
        

def central_velocity_profile(slist) : 
    d = pynbody.filt.Disc(1,.1)

    fig, axs = plt.subplots(2,1, figsize=(8,12))

    labels = ['nomet','met',]
    colors = ['b','r']

    for i, s in enumerate(slist) : 
        pg = pynbody.analysis.profile.Profile(s.g[d], nbins=50, type = 'log',min=.01,max=1)
        ps = pynbody.analysis.profile.Profile(s.s[d], nbins=50, type = 'log',min=.01,max=1)
    

        axs[0].plot(pg['rbins'].in_units('pc'),pg['vt'], label="t = %.0f"%s.properties['time'].in_units('Myr'))
        axs[0].plot(pg['rbins'].in_units('pc'),pg['vt_disp'], '--')


        axs[1].plot(ps['rbins'].in_units('pc'),ps['vt'])
        axs[1].plot(ps['rbins'].in_units('pc'),ps['vt_disp'],'--')


    for ax in axs: 
        ax.semilogx()
        ax.set_ylim(0,400)
        ax.set_xlabel(r'R [pc]')
        ax.set_ylabel(r'$v, v_{disp} \mathrm{~km/s}$')
    axs[0].legend()
    axs[0].set_title('gas')
    axs[1].set_title('stars')

def temperature_maps(sl) : 
    f,axs = plt.subplots(2,3,figsize=(10.575, 7.9625))
    
    for i, s in enumerate(sl) : 
        pynbody.plot.image(s.g,width='2 kpc',av_z='rho',qty='temp',vmin=3.5,vmax=7.,subplot=axs[0,i],show_cbar=False)
        axs[0,i].set_title('t = %.0f'%s.properties['time'].in_units('Myr'))
        s.rotate_x(90)
        pynbody.plot.image(s.g,width='2 kpc',av_z='rho',qty='temp',vmin=3.5,vmax=7.,subplot=axs[1,i],show_cbar=False)
        s.rotate_x(-90)
        
    map(utils.clear_labels,axs.flatten()[1:])
    utils.make_spanned_colorbar(f,axs,'$T$ [K]')

    
def vsigma_vs_time(sl) : 
    fig, ax = plt.subplots()

    for i,s in enumerate(sl) : 
        color = get_color(i,len(sl))

        s.g['speed'] = np.sqrt(s.g['v2'])
        ind = np.where(s.g['rho'].in_units('m_p cm^-3') < 1e6)[0]
        p = pynbody.analysis.profile.Profile(s.g[ind],max = 1, min = .002, type = 'log', nbins=20)
        ax.plot(p['rbins'].in_units('pc'), p['speed']/p['speed_disp'], label = 't=%.0f Myr'%s.properties['time'].in_units('Myr'),color=color)

    ax.set_xlabel('$R$ [pc]')
    ax.set_ylabel(r'$v/\sigma$')
    ax.legend(prop=dict(size=12))
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
                

def make_spaans_plot(tablefile = None): 
    plt.figure()

    if tablefile is None : 
        import glob 
        tablefile = glob.glob('*.Spaanscool')
        if len(tablefile) > 1: raise RuntimeException('Should only have a single cooling table')

    table = np.genfromtxt(tablefile[0])
    # the table specifies temperature as a function of density for
    # different accretion rates and metallicities. The columns are
    # paired by accretion rate -- columns 1-2 = 100 Msol/yr, 3-4 = 60
    # Msol/yr, 5-6 = 30 Msol/yr. The first column in each pair is at
    # solar metallicity, the second at 3x solar. 

    labels = ['$100 \mathrm{~M}_{\odot} / $yr, $\mathrm{Z}_{\odot}$',
              '$100 \mathrm{~M}_{\odot} / $yr, $3\mathrm{Z}_{\odot}$',
              '$60 \mathrm{~M}_{\odot} / $yr, $\mathrm{Z}_{\odot}$',
              '$60 \mathrm{~M}_{\odot} / $yr, $3\mathrm{Z}_{\odot}$',
              '$30 \mathrm{~M}_{\odot} / $yr, $\mathrm{Z}_{\odot}$',
              '$30 \mathrm{~M}_{\odot} / $yr, $3\mathrm{Z}_{\odot}$']


    for i in np.arange(6)+1 : 
        plt.plot(table[:,0], table[:,i], label = labels[i-1])
        
    plt.xlabel(r'$\rho$ [amu/cc]')
    plt.ylabel('T [K]')
    plt.loglog()
    plt.ylim(10,2e3)
    plt.legend(loc='upper right',prop=dict(size=12))

def phaseplot_spaans(s, tablefile=None) : 
    if tablefile is None : 
        import glob 
        tablefile = glob.glob('*.Spaanscool')
        if len(tablefile) > 1: raise RuntimeException('Should only have a single cooling table')

    table = np.log10(np.genfromtxt(tablefile[0]))

    pynbody.plot.rho_T(s.g)
#    res = pynbody.plot.generic.gauss_kde(np.log10(s.g['rho']),np.log10(s.g['temp']),mass=s.g['mass'],make_plot=False)
 #   plt.imshow(np.log10(res[0]),extent=(res[1].min(),res[1].max(),res[2].min(),res[2].max()),origin='lower',vmin=4,vmax=7)
    plt.plot(table[:,0],table[:,1],table[:,0],table[:,2],linewidth=2)

def make_jeansmass_figure(s) :
    from scipy.stats import gaussian_kde as kde
    if 'p' not in s.g.keys() : 
        s.g['p'] = s.g['pres']
        s.g['p'].set_units_like('Pa')


    fig, ax = plt.subplots()#2,1, figsize=(8,12))

        
    x = np.linspace(0,12,200)
    k = kde(np.log10(s.g['mjeans'].in_units('Msol')/s.g['mass'].in_units('Msol')))
    k2 = kde(np.log10(s.g['ljeans'].in_units('kpc')/s.g['smooth'].in_units('kpc')))
    ax.plot(x,k(x),'k')
    ax.set_xlabel(r'$\mathrm{log}_{10}(M_{jeans} / M_{part})$')
    #axs[1].plot(x,k2(x),'k')
    #axs[1].set_xlabel(r'$\mathrm{log}(M_{jeans} / M_{part})$')
    
def make_sfr_figure(s) : 
    f,axs = plt.subplots(3,1)

    sl_s = pynbody.tipsy.StarLog('../gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc/gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc.starlog')

    p = pynbody.load('../gas_merger0.1_thr10_Rx8_nometalcool/6/gas_merger0.1_thr10_Rx8_nometalcool.00613')
    sl_p = pynbody.tipsy.StarLog('../gas_merger0.1_thr10_Rx8_nometalcool/gas_merger0.1_thr10_Rx8_nometalcool.starlog')

    gp = pynbody.load('../gas_merger0.1_thr10/5/gas_merger0.1_thr10.00500')
    
    # the first section -- before splitting
    new = pynbody.filt.HighPass('timeform', 0)
    h_gp, bins_gp = np.histogram(gp.s[new]['timeform'],weights=gp.s[new]['massform'],bins=100)
    scale = 1e9*(bins_gp[1]-bins_gp[0])
    
    axs[0].plot(.5*(bins_gp[:-1]+bins_gp[1:]), h_gp/scale*2.3e5)
    
    # the second section -- after the first split

    h_p, bins_p = np.histogram(sl_p['tform'], 
                               weights = sl_p['massform'],
                               bins = 100)
    scale = 1e9*(bins_p[1]-bins_p[0])
    axs[1].plot(.5*(bins_p[:-1]+bins_p[1:]), h_p/scale*2.e5)
    # the third section -- current s
    print min(sl_s['tform'])
    h_s, bins_s = np.histogram(sl_s['tform'], 
                               weights = sl_s['massform'],
                               bins = 100)
    scale = 1e9*(bins_s[1]-bins_s[0])
    axs[2].plot(.5*(bins_s[:-1]+bins_s[1:]), h_s/scale*2.3e5)
    

def plot_sfr(s) : 
    sfh,bins = np.histogram(s.s['tform'].in_units('Gyr'),weights=s.s['mass'].in_units('Msol'),
                            range=[4,5.1],bins=100)
    bins = .5*(bins[:-1]+bins[1:])
    width = bins[1] - bins[0]
    sfh /= width
    sfh /= 1e9

    f,ax=plt.subplots()
    ax.plot(bins,sfh,'k')
    ax.set_xlabel('$t$ [Gyr]')
    ax.set_ylabel('SFR [M$_{\odot}~\mathrm{yr}^{-1}$]')


def plot_density_profiles(sl) : 
    f,ax=plt.subplots(2,1,figsize=(8,10))
    sph = pynbody.filt.Sphere('100 pc')
    disk = pynbody.filt.Disc('500 pc', '200 pc')
    
    for i,s in enumerate(sl):
        ps = pynbody.analysis.profile.Profile(s.s[disk],min=.001,max=.1,type='log',nbins=50)
        pg = pynbody.analysis.profile.Profile(s.g[disk],min=.001,max=.1,type='log',nbins=50)
        
        ps['rbins'].convert_units('pc')
        pg['rbins'].convert_units('pc')
        ps['density'].convert_units('Msol pc^-2')
        pg['density'].convert_units('Msol pc^-2')
        

        color = get_color(i,len(sl))

        ax[0].plot(ps['rbins'],ps['density'], color = color, 
                   label = '%.0f Myr'%s.properties['time'].in_units('Myr'))
        ax[0].plot(pg['rbins'],pg['density'], color = color, linestyle='--')
        ax[1].plot(ps['rbins'],ps['z_rms'].in_units('pc'), color = color)
        ax[1].plot(pg['rbins'],pg['z_rms'].in_units('pc'), color = color, linestyle = '--')
        

    for a in ax: 
        a.loglog()
        a.set_xlabel('$R$ [pc]')
        
    ax[0].legend(loc = 'upper right', frameon=False, prop = dict(size=12))
    ax[0].set_ylabel('$\Sigma_{\star,\mathrm{g}}$ [M$_{\odot}~\mathrm{pc}^{-2}$]')
    ax[1].set_ylabel('$z_{rms}$ [pc]')
    

def make_movie(t1,t2) : 
    import glob

    fs = glob.glob('*/*.0????')
    fs.sort(key=lambda x: x[-5:])
    # nearest outputs to t1 and t2
    f1 = smbh.nearest_output(t1)
    f2 = smbh.nearest_output(t2)

    for i, f in enumerate(fs) : 
        if f.split('/')[-1] == f1.split('/')[-1] : 
            ind1 = i
        elif f.split('/')[-1] == f2.split('/')[-1] : 
            ind2 = i
            
    print ind1, ind2

    
    for i, f in enumerate(fs[ind1:ind2]) :
        s = pynbody.load(f)
        pynbody.analysis.halo.center(s,mode='ssc')
        s.rotate_x(90)

        im = pynbody.plot.image(s.g,qty='rho',units='Msol kpc^-2',width=.1)
        smbh.overplot_bh(s,plt.gca())
        plt.xlim(-.05,.05)
        plt.ylim(-.05,.05)
#        plt.imshow(np.log10(im))
        plt.savefig('movie/edgeon%s.png'%str(i),format='png',bbox_inches='tight')
        
    

def plot_smbh_forces(sl) : 
    f,ax=plt.subplots()
    from pynbody.grav_omp import direct

    t = pynbody.array.SimArray(np.zeros(len(sl)),'Myr')
    ps = np.empty((len(sl),2))
    fs = np.empty((len(sl),2,3))
    fg = np.empty((len(sl),2,3))
    
    for i,s in enumerate(sl): 
        ps[i], fs[i] = direct(s.s,s[smbh.bh_index(s)]['pos'].view(np.ndarray), eps=.001)
        ps[i],fg[i] = direct(s.g,s[smbh.bh_index(s)]['pos'].view(np.ndarray), eps=.001)

        t[i] = s.properties['time'].in_units('Myr')

    return t, fs, fg

def central_mass(sl) : 
    f,ax = plt.subplots()
    times = np.zeros(len(sl))
    gmasses = np.zeros(len(sl))
    smasses = np.zeros(len(sl))
    gmasses_d = np.zeros(len(sl))
    smasses_d = np.zeros(len(sl))
    sph = pynbody.filt.Sphere('200 pc')
    disc = pynbody.filt.Disc('200 pc','10 pc')
    for i,s in enumerate(sl):
        times[i] = s.properties['time'].in_units('Myr')
        gmasses[i] = s.g[sph]['mass'].in_units('Msol').sum()
        smasses[i] = s.s[sph]['mass'].in_units('Msol').sum()
        gmasses_d[i] = s.g[disc]['mass'].in_units('Msol').sum()
        smasses_d[i] = s.s[disc]['mass'].in_units('Msol').sum()
    plt.plot(times,gmasses,'b-',label='gas / sphere')
  #  plt.plot(times,smasses,'g-',label='stars / sphere')
    plt.plot(times,gmasses_d,'b--',label='gas / disk')
  #  plt.plot(times,smasses_d,'g--',label='stars / disk')
    plt.xlabel('time [Myr]')
    plt.ylabel('Mass enclosed [M$_{\odot}$]')
    plt.legend()
    

def savefig(fname): 

    plt.savefig('smbh_p1_figs/%s.pdf'%fname,format='pdf',bbox_inches='tight')
    plt.savefig('smbh_p1_figs/%s.eps'%fname,format='eps',bbox_inches='tight')


def get_color(i,n,cmap=plt.cm.gist_ncar) : 
    return cmap(int(i*256./n))

def generate_images(s,width,vmin=6,vmax=12):
    from pickle import dump

    fd,axd=plt.subplots()
    fo_im,R,G,B = utils.make_rgb_stellar_image(s,width,filename='%s_stars_fo.png'%s.filename,vmin=vmin,vmax=vmax)
    fo_gas = pynbody.plot.image(s.g,width=width,units='Msol kpc^-2')
    s.rotate_x(90)
    
    eo_im,R,G,B = utils.make_rgb_stellar_image(s,width,filename='%s_stars_eo.png'%s.filename,vmin=vmin,vmax=vmax)
    eo_gas = pynbody.plot.image(s.g,width=width,units='Msol kpc^-2')
    dump({'fo':fo_gas, 'eo':eo_gas},open('%s_gas_image'%s.filename,'w'))
    s.rotate_x(-90)

def make_composite_filmstrip(sl,width) : 
    import utils
    from pickle import dump,load

    f, axs = plt.subplots(len(sl)/2,4,figsize=(20,len(sl)/2*5.0))
    
    axs=axs.flatten()

    for i, s in enumerate(sl) : 
        try: 
            print i, s.filename
            fo_im = plt.imread('%s_stars_fo.png'%s.filename)
            eo_im = plt.imread('%s_stars_eo.png'%s.filename)
            gas_im = load(open('%s_gas_image'%s.filename))
            fo_gas = gas_im['fo']
            eo_gas = gas_im['eo']
            
        except (RuntimeError, IOError) : 
            generate_images(s)

        ax = axs[i*2]
             
        ax.imshow(fo_im,origin='lower')
        ax.contour(fo_gas,levels=[8.5,9,10,11,11.5],colors='white',linewidths=.5)

        ax.annotate('$t = %0.0f$ Myr'%(s.properties['time'].in_units('Myr')), 
                     (0.1,0.9), color='white', fontweight='bold', 
                     xycoords = 'axes fraction', fontsize=12)
        
        if i == 0: 
            ax.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
            ax.annotate("500 pc", xy=(0.38,0.065), color ="white",fontsize='smaller', 
                        xycoords = 'axes fraction')


        axs[i*2+1].imshow(eo_im,origin='lower')
        axs[i*2+1].contour(eo_gas,levels=[8.5,9,10,11,11.5],colors='white',linewidths=.5)
        
    for ax in axs: 
        utils.clear_labels(ax,True)

def make_edgeon_faceon_map(s) : 
    f, axs = plt.subplots(1,2,figsize=(17.7,6.125))
    
    axs = axs.flatten()
    
    pynbody.plot.image(s.g,width='500 pc', cmap=plt.cm.Greys_r, qty='rho', units='Msol pc^-2', subplot=axs[0], show_cbar=False,vmin=1.5,vmax=4.5)
    #smbh.overplot_bh(s,axs[0])

    s.rotate_x(90)
    
    pynbody.plot.image(s.g,width='500 pc', cmap=plt.cm.Greys_r, qty='rho', units='Msol pc^-2', subplot=axs[1], show_cbar=True,vmin=1.5,vmax=4.5)
    #smbh.overplot_bh(s,axs[1])
        
#    for ax in axs: 
#        ax.set_xlim(-.25,.25)
#        ax.set_ylim(-.25,.25)
        
    plt.draw()


def make_general_orbit_plot() :
    dat = np.load('../gas_merger0.1_thr10_Rx8_highSFthresh/bh_orbit.npz')
    dat2 = np.load('./bh_orbit.npz')

    ind = np.where(dat['t'] < dat2['t'][0])[0]

    plt.figure(figsize=(20,6))

    plt.plot(dat['t'][ind],dat['r'][ind]*1e3,'k')
    plt.plot(dat2['t'],dat2['r']*1e3,'k')
    plt.plot((dat2['t'][0],dat2['t'][0]),(1e-1,1e4),'r--',linewidth=3,zorder=-500)
    plt.xlabel('$t$ [Myr]')
    plt.ylabel('separation [pc]')
    plt.xlim(dat['t'][0],dat2['t'][-1])
    
    plt.semilogy()


def plot_no_labels(s,width,vmin,vmax,name) : 
    # gas_merger/2/
    # gas_merger/4/
    # gas_merger/5/
    # gas_merger0.1_thr10_Rx8_nometalcool/5/501
    # 
    import utils
    im = pynbody.plot.image(s.g,width=width,av_z=True)
    f,ax=plt.subplots()
    ax.imshow(im,vmin=vmin,vmax=vmax,cmap=plt.cm.Greys_r)
    utils.clear_labels(ax)
    plt.savefig(name+'.pdf',bbox_inches='tight')
