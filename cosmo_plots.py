import pynbody
import matplotlib.pylab as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from os import system

def make_single_output_maps(s,h) : 

    s.physical_units()

    h1 = h[1]
        
    fig = plt.figure(figsize=(17,10))

    Rvir = h1.properties['Rvir']
    

    # 5 Rvir panels
    
    ax = fig.add_subplot(231)
    pynbody.analysis.angmom.faceon(h1)
    pynbody.plot.image(s.g,units='Msol kpc^-2', width=5*Rvir,subplot=ax)
    title = 'z = %.2f' % (1/s.properties['a']-1.0)
    ax.set_title(title)
    s.rotate_x(90)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    
    ax = fig.add_subplot(234)
    pynbody.plot.image(s.g,units='Msol kpc^-2', width=5*Rvir,subplot=ax)
    s.rotate_x(-90)
    plt.xlabel('kpc')
    plt.ylabel('kpc')
    
    # Rvir panels
    
    ax = fig.add_subplot(232)
    pynbody.plot.image(s.g,units='Msol kpc^-2', width=Rvir,subplot=ax)
    s.rotate_x(90)
    plt.xlabel('kpc')
    plt.ylabel('kpc')

    ax = fig.add_subplot(235)
    pynbody.plot.image(s.g,units='Msol kpc^-2', width=Rvir,subplot=ax)
    s.rotate_x(-90)
    plt.xlabel('kpc')
    plt.ylabel('kpc')

    # Rvir/5 panels
    
    ax = fig.add_subplot(233)
    pynbody.plot.image(s.g,units='Msol kpc^-2', width=Rvir/5,subplot=ax)
    s.rotate_x(90)
    plt.xlabel('kpc')
    plt.ylabel('kpc')

    ax = fig.add_subplot(236)
    pynbody.plot.image(s.g,units='Msol kpc^-2', width=Rvir/5,subplot=ax)
    s.rotate_x(-90)
    plt.xlabel('kpc')
    plt.ylabel('kpc')

    plt.savefig(s.filename+'_gasmaps.pdf', format='pdf')

def make_single_output_profiles(s,h):
    s.physical_units()
    h1 = s[np.where(s['grp'] == 1)[0]]

    pynbody.analysis.angmom.faceon(h1)

    Rvir = h[1].properties['Rvir']
    

    fig = plt.figure(figsize=(15,15))
    
    cold = pynbody.filt.LowPass('temp',1e5)
    hot = pynbody.filt.HighPass('temp',1e5)

    pc = pynbody.analysis.profile.Profile(h1.g[cold],max=Rvir,ndim=3, type='equaln')
    ph = pynbody.analysis.profile.Profile(h1.g[hot],max=Rvir,ndim=3, type = 'equaln')
    pd = pynbody.analysis.profile.Profile(h1.d,max=Rvir,ndim=3, type = 'equaln')

    ax = fig.add_subplot(221)
    plt.plot(pc['rbins']/Rvir,pc['density'],'b-', label='$T < 10^5$ K')
    plt.plot(ph['rbins']/Rvir,ph['density'],'r-', label='$T > 10^5$ K')
    plt.plot(pd['rbins']/Rvir,pd['density'],'k-', label='DM')
    plt.xlabel('$R/R_{vir}$')
    plt.ylabel('$\Sigma$ [M$_{\odot}$ kpc$^{-3}$]')
    plt.loglog()
    plt.legend(prop=FontProperties(size='small'))
    title = 'z = %.2f' % (1/s.properties['a']-1.0)
    ax.set_title(title)

    ax = fig.add_subplot(222)
    plt.plot(pc['rbins']/Rvir,pc['j_theta']*180/np.pi,'b-')
    plt.plot(ph['rbins']/Rvir,ph['j_theta']*180/np.pi,'r-')
    plt.plot(pd['rbins']/Rvir,pd['j_theta']*180/np.pi,'k-')
    plt.semilogx()
    plt.xlabel(r'$R/R_{vir}$')
    plt.ylabel(r'$\theta$')
    plt.savefig(s.filename+'_gas_profiles.pdf', format='pdf')
    

    ax = fig.add_subplot(223)
    plt.plot(pc['rbins']/Rvir,pc['jtot']*pc['mass'],'b-')
    plt.plot(ph['rbins']/Rvir,ph['jtot']*ph['mass'],'r-')
    plt.plot(pd['rbins']/Rvir,pd['jtot']*pd['mass'],'k-')
    plt.semilogx()
    plt.xlabel(r'$R/R_{vir}$')
    plt.ylabel(r'$J_{tot}$')
    plt.savefig(s.filename+'_gas_profiles.pdf', format='pdf')

    ax = fig.add_subplot(224)
    plt.plot(pc['rbins']/Rvir,pc['jtot'],'b-')
    plt.plot(ph['rbins']/Rvir,ph['jtot'],'r-')
    plt.plot(pd['rbins']/Rvir,pd['jtot'],'k-')
    plt.semilogx()
    plt.xlabel(r'$R/R_{vir}$')
    plt.ylabel(r'$j_{tot}$')
    plt.savefig(s.filename+'_gas_profiles.pdf', format='pdf')
    


    del(fig)
    
def make_gas_stars_overlay(s,width=20,yng = pynbody.filt.LowPass('age',1.0),center=False) : 

    fig = plt.figure(figsize=(15,10))

    if center: 
        s.physical_units()
        pynbody.analysis.angmom.faceon(s)

    cold = pynbody.filt.LowPass('temp', 1e5)
    gas_dens = pynbody.plot.image(s.g[cold],width=width,units='Msol kpc^-2',noplot=True)
    xsys = np.linspace(-width/2,width/2,len(gas_dens))
    plt.clf()

    ax = fig.add_subplot(1,2,1)
    yng_dens = pynbody.plot.image(s.s[yng],  width=width,units='Msol kpc^-2',cmap=plt.cm.Blues_r, show_cbar = False, subplot = ax, vmin=5.5,vmax=10)
    plt.contour(xsys,xsys,gas_dens,colors='w',levels=np.linspace(7.5,10,5))

    ax = fig.add_subplot(1,2,2)
    old_dens = pynbody.plot.image(s.s[~yng], width=width,units='Msol kpc^-2',cmap=plt.cm.Reds_r, show_cbar= False, subplot = ax, vmin=6, vmax=10)
    plt.contour(xsys,xsys,gas_dens,colors='w',levels=np.linspace(7.5,10,5))
    

def make_gas_map(s, center=True, subplot = False, savefig = False, annotate = False, **kwargs):
    
    s.physical_units()
    if center:
        pynbody.analysis.angmom.sideon(s.g,mode='ssc',disk_size='1 kpc')
    
    if not subplot: 
        subplot = plt.figure(figsize=(13,10)).add_subplot(111)
        
    pynbody.plot.image(s.g,cmap=plt.cm.Blues_r,units='Msol kpc^-2',subplot=subplot, **kwargs)
    
    if annotate:
        subplot.annotate('z = %.2f'%(1.0/s.properties['a']-1), 
                         (-25,25), color = "white", weight='bold')
    
    if savefig: 
        plt.savefig(s.filename+'_gas_map.png', format='png')

def make_fo_eo_map(s, width = 40, center = True, subplot = False, savefig = False,  **kwargs) : 
    s.physical_units()
    if center:
        pynbody.analysis.angmom.sideon(s.g, mode = kwargs.get('mode', 'hyb'), disk_size=kwargs.get('disk_size', '5 kpc'))
    
    subplot = plt.figure(figsize=(7,14)).add_subplot(211)
    pynbody.plot.image(s.g,width=width,cmap=plt.cm.Blues_r,units='Msol kpc^-2',subplot=subplot,show_cbar=False,**kwargs)
    subplot.annotate('z = %.2f'%(1.0/s.properties['a']-1), 
                     (-25,25), color = "white", weight='bold')
    
    subplot.set_xlabel('')
    subplot.set_xticklabels('')

    subplot = plt.subplot(212)
    s.rotate_x(90)
    pynbody.plot.image(s.g,width=width,cmap=plt.cm.Blues_r,units='Msol kpc^-2',subplot=subplot, show_cbar=False,**kwargs)
    
    plt.subplots_adjust(hspace=0.07)

    if hasattr(s,'base') : filename = s.base.filename+'_gas_eofo_map.png'
    else: filename = s.filename+'_gas_eofo_map.png'

    if savefig: 
        plt.savefig(filename, format='png')


def make_gas_map_wrap(i,ax,flist,**kwargs) : 
    s = pynbody.load(flist[i])
    make_gas_map(s.halos()[1],subplot=ax,**kwargs)

def make_func_wrap(i,ax,flist,func,**kwargs) : 
    s = pynbody.load(flist[i])
    func(s.halos()[1],subplot=ax,**kwargs)

def make_gas_profile(s, logy=True, prof_name = 'density', center=True, subplot = False, **kwargs) : 
    from pynbody.analysis.profile import Profile

    s.physical_units()
    if center: 
        pynbody.analysis.angmom.faceon(s.g,mode='ssc',disk_size='1 kpc')
        

    if not subplot: 
        subplot = plt.figure(figsize=(10,10)).add_subplot()
    
    cold = pynbody.filt.LowPass('temp', 1e5)
    hot = ~cold

    p  = Profile(s.g,       min=0.01,max=100,type='log',ndim=3)
    pc = Profile(s.g[cold], min=0.01,max=100,type='log',ndim=3)
    ph = Profile(s.g[hot],  min=0.01,max=100,type='log',ndim=3)
    
    subplot.plot(p['rbins'],p[prof_name],label='all')
    subplot.plot(pc['rbins'],pc[prof_name],label=r'$T < 10^5$')
    subplot.plot(ph['rbins'],ph[prof_name],label=r'$T > 10^5$')
    subplot.set_title('z = %.2f'%s.properties['z'])
    subplot.set_xlabel(r'$R / \mathrm{kpc}')
    subplot.set_ylabel(r'$'+p[prof_name].units.latex()+'$')
    if logy: subplot.semilogy()
            
def make_gas_fractional_profile(s, logy=False, prof_name = 'density', center=True, subplot = False, **kwargs) : 
    from pynbody.analysis.profile import Profile

    s.physical_units()
    if center: 
        pynbody.analysis.angmom.faceon(s.g,mode='ssc',disk_size='1 kpc')
        

    if not subplot: 
        subplot = plt.figure(figsize=(10,10)).add_subplot()

    cold = pynbody.filt.LowPass('temp', 1e5)
    hot = ~cold

    p  = Profile(s.g,       min=0.01,max=100,type='log',ndim=3)
    pc = Profile(s.g[cold], min=0.01,max=100,type='log',ndim=3)
    ph = Profile(s.g[hot],  min=0.01,max=100,type='log',ndim=3)
    
    subplot.plot(pc['rbins'],
                 pc[prof_name]*pc['mass']/(p[prof_name]*p['mass']),
                 label=r'$T < 10^5$')
    subplot.plot(ph['rbins'],ph[prof_name]*ph['mass']/(p[prof_name]*p['mass']),label=r'$T > 10^5$')
    subplot.set_title('z = %.2f'%s.properties['z'])
    subplot.set_xlabel(r'$R / \mathrm{kpc}')
    if logy: subplot.semilogy()


def make_multiple_snapshot_images(slist,x2=100,vsmin=3.5,vsmax=10,vgmin=6.2,vgmax=10):
    from pynbody.analysis.cosmology import age

    fg = plt.figure(figsize=(10.24,8.192))
    
    ax = plt.axes((0,0,1,1))

    for s in slist:
        dir = s.filename.split('/')[0]
        im = pynbody.sph.threaded_render_image(s.s,kernel=pynbody.sph.Kernel2D(),
                                               x2=x2,nx=1024,ny=819,num_threads=20)

        logim = np.log10(im)
        logim[logim == float('-inf')] = logim[logim>float('-inf')].min()

        plt.imshow(logim,cmap=plt.cm.Greys_r)#, vmin=vsmin,vmax=vsmax)

        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(0)

        plt.savefig(dir+"/images/%.2fGyr_star.png"%age(s),format='png')

        im = pynbody.sph.threaded_render_image(s.g,kernel=pynbody.sph.Kernel2D(),
                                               x2=x2,nx=1024,ny=819,num_threads=20)

        plt.imshow(np.log10(im),cmap=plt.cm.Greys_r)#,vmin=vgmin,vmax=vgmax)

        plt.savefig(dir+"/images/%.2fGyr_gas.png"%age(s),format='png')

        
        # make the composite image

        system("mogrify -fill gold -tint 50 -transparent black -gamma .7 +contrast +contrast +contrast %s/images/%.2fGyr_star.png"%(dir,age(s)))
        system("mogrify -fill blue -tint 50 %s/images/%.2fGyr_gas.png"%(dir,age(s)))

        system("convert %s/images/%.2fGyr_star.png %s/images/%.2fGyr_gas.png -compose blend -define compose:args=100 -composite %s/composites/composite%.2fGyr.png"%(dir,age(s),dir,age(s),dir,age(s)))
        
       # system("mogrify -median 2 composites/composite%.2fGyr.png"%age(s))
        
        
