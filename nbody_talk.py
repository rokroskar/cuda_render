
import pynbody
import matplotlib.pylab as plt

def make_gas_figure(s) :

    pynbody.analysis.angmom.faceon(s)
    s.rotate_x(90)
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(131)
    pynbody.plot.image(s.g,width=20,qty='rho',vmin=-1,vmax=3,colorbar=False, threaded=True, subplot=ax)
    plt.annotate('Density', (-8,8), color='white', fontsize=20, fontweight='bold') 
    ax = plt.subplot(132)
    pynbody.plot.image(s.g,width=20,qty='temp',vmin=4,vmax=7,colorbar=False,threaded=True, subplot=ax)
    plt.annotate('Temperature', (-8,8), color='white', fontsize=20, fontweight='bold') 
    ax = plt.subplot(133)
    pynbody.plot.image(s.g,width=20,qty='feh',vmin=-2,log=False,colorbar=False,threaded=True, subplot=ax)
    plt.annotate('[Fe/H]', (-8,8), color='white', fontsize=20, fontweight='bold') 

    plt.subplots_adjust(wspace=.3,hspace=0.15)


def make_stars_figure(s):
    pynbody.analysis.angmom.faceon(s)
    s.rotate_x(90)
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(131)
    pynbody.plot.image(s.s,width=20,qty='rho',vmin=-1,vmax=5,colorbar=False, threaded=True, subplot=ax, av_z=True)
    plt.annotate('Density', (-8,2), color='white',fontsize=20,fontweight='bold')
    ax.set_ylim(-4,4)
    ax.yaxis.set_ticklabels('')
    ax.set_ylabel('')
    
    ax = plt.subplot(132)
    pynbody.plot.image(s.s,width=20,qty='age',vmin=1,vmax=5,log=False,colorbar=False,
                       threaded=True, subplot=ax, av_z=True, units='Gyr')
    plt.annotate('Age', (-8,2), color='white',fontsize=20,fontweight='bold')
    ax.set_ylim(-4,4)
    ax.yaxis.set_ticklabels('')
    ax.set_ylabel('')
    
    ax = plt.subplot(133)
    pynbody.plot.image(s.s,width=20,qty='feh',vmin=-1,vmax=.1,log=False,colorbar=False,
                       threaded=True, subplot=ax, av_z=True)
    plt.annotate('[Fe/H]', (-8,2), color='white',fontsize=20,fontweight='bold')
    ax.set_ylim(-4,4)
    ax.yaxis.set_ticklabels('')
    ax.set_ylabel('')
    
    plt.subplots_adjust(wspace=.08,hspace=0.15)
