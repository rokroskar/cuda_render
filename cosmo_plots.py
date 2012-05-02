import pynbody
import matplotlib.pylab as plt
import numpy as np
from matplotlib.font_manager import FontProperties

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
    
