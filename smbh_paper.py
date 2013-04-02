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
    
    rs = np.sqrt((orbit['pos'][:,:,0:2]**2).sum(axis=2))

    ax = fig.add_subplot(2,1,1)

    plt.plot(orbit['t'], rs*1000.)

    plt.xlabel('$t$ [Myr]')
    plt.ylabel('$R$ [pc]')
    plt.semilogy()
    plt.xlim(orbit['t'].min(), orbit['t'].max())

    ax = fig.add_subplot(2,1,2)
    
    plt.plot(orbit['t'], orbit['pos'][:,0,2]*1000.)
    plt.plot(orbit['t'], orbit['pos'][:,1,2]*1000.)

    plt.xlabel('$t$ [Myr]')
    plt.ylabel('$z$ [pc]')
    plt.xlim(orbit['t'].min(), orbit['t'].max())
    
    
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


def load_snapshot_sequence(dir='./', ntiles = 20, t1 = None, t2 = None):
    out1 = pynbody.load(smbh.nearest_output(0))
    out2 = pynbody.load(smbh.nearest_output(1e6))

    if t1 is None:
        t1 = out1.properties['time'].in_units('Myr')
    if t2 is None: 
        t2 = out2.properties['time'].in_units('Myr')

    times = np.linspace(t1,t2,ntiles)

    return [pynbody.load(x) for x in map(smbh.nearest_output, times)]
    

def center_snapshot(s) : 
    if np.diff(s[smbh.bh_index(s)]['r'])[0] > 0.2 : 
            pynbody.analysis.halo.center(s,mode='ind',ind=smbh.bh_index(s),vel=False)
    else:
        pynbody.analysis.halo.center(s.g,mode='hyb',vel=False)

    pynbody.analysis.halo.vel_center(s.g,cen_size=.5)
 

def make_filmstrip_figure(slist): 

    f, axs = plt.subplots(5,4,figsize=(8,10))

    cmap = plt.cm.Blues_r
    
    for i,s in enumerate(slist) : 
        ax = axs.flatten()[i*2]
        
        if np.diff(s[smbh.bh_index(s)]['r'])[0] > 0.2 : 
            pynbody.analysis.halo.center(s,mode='ind',ind=smbh.bh_index(s))
            width = .7
        else:
            pynbody.analysis.halo.center(s.g,mode='hyb')
            width = .25

        pynbody.plot.image(s.g,width=width,units='Msol kpc^-2', subplot=ax, show_cbar=False, cmap=cmap, threaded=20)
        ax.annotate('$t = %0.0f$ Myr'%(s.properties['time'].in_units('Myr')-slist[0].properties['time'].in_units('Myr')), 
                     (0.1,0.85), color='white', fontweight='bold', 
                     xycoords = 'axes fraction', fontsize=12)
        smbh.overplot_bh(s,ax)
        ax.set_xlim(-width/2.0,width/2.0)
        ax.set_ylim(-width/2.0,width/2.0)

        ax = axs.flatten()[i*2+1]
        s.rotate_x(90)
        pynbody.plot.image(s.g,width=width,units='Msol kpc^-2', subplot=ax, show_cbar=False, cmap=cmap)
        smbh.overplot_bh(s,ax)
        ax.set_xlim(-width/2.0,width/2.0)
        ax.set_ylim(-width/2.0,width/2.0)
        s.rotate_x(-90)

    map(utils.clear_labels,axs.flatten())
    plt.subplots_adjust(hspace=.1,wspace=.1)
        

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
            
            pynbody.plot.image(s.g,width=widths[j],units='Msol kpc^-2', subplot=ax[j], show_cbar=False, cmap=cmap, threaded=20)
        
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

    fig, axs = plt.subplots(2,1, figsize=(8,12))

    labels = ['nomet','met',]
    colors = ['b','r']

    for i, s in enumerate(slist) : 
        pg = pynbody.analysis.profile.Profile(s.g[d], nbins=50, type = 'log',min=.01,max=1)
        ps = pynbody.analysis.profile.Profile(s.s[d], nbins=50, type = 'log',min=.01,max=1)
    

        axs[0].plot(pg['rbins'].in_units('pc'),pg['speed']/pg['speed_disp'], label="t = %.0f"%s.properties['time'].in_units('Myr'))
        axs[1].plot(ps['rbins'].in_units('pc'),ps['speed']/ps['speed_disp'])


    for ax in axs: 
        ax.semilogx()
        ax.set_xlabel(r'R [pc]')
        ax.set_ylabel(r'$v/\sigma$')
    axs[0].legend()
    axs[0].set_title('gas')
    axs[1].set_title('stars')
    
def central_vsigma_inclined(s,ax=None) : 

    d = pynbody.filt.Sphere(1)
    if ax is None:
        f,ax = plt.subplots()

    for angle in [0,20,45,60,89]  : 
        s.rotate_x(angle)
        s['absvz'] = np.abs(s['vz'])
        p = pynbody.analysis.profile.InclinedProfile(s.g,angle,nbins=20,max=.5)
        ax.plot(p['rbins'],p['absvz']/p['absvz_disp'],label='%s'%angle)
        s.rotate_x(-angle)
        del(s['absvz'])

    ax.set_xlabel('$R$ [kpc]')
    ax.set_ylabel('$V/\sigma$')
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
    
    pass

def make_sfr_figure() : 
    f,axs = plt.subplots(3,1)

    s = pynbody.load('../gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc/45/gas_merger0.1_thr10_Rx8_nometalcool_1pc_rc.04500')
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
        
    

def savefig(fname): 

    plt.savefig('smbh_p1_figs/%s.pdf'%fname,format='pdf',bbox_inches='tight')
    plt.savefig('smbh_p1_figs/%s.eps'%fname,format='eps',bbox_inches='tight')

