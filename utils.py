#
#
# Generally useful stuff
#
#

import matplotlib.pylab as plt
import numpy as np


def make_tile_figure(nrow,ncol,func,*args,**kwargs) : 

    fig = plt.figure(figsize=(5*ncol,5*nrow))

    for i in range(nrow) :
        for j in range(ncol) : 
            
            ax = fig.add_subplot(nrow,ncol,i*ncol+j+1)
            
            func(i*ncol+j,ax,*args,**kwargs)

            if (i < nrow-1) | (j > 0): clear_labels(ax)            
                
def clear_labels(ax):
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_ylabel('')
    ax.set_xlabel('')


def shrink_sphere(sim, r=None, shrink_factor = 0.7, min_particles = 100, verbose = False) : 
    if r is None :
        # use rough estimate for a maximum radius
        # results will be insensitive to the exact value chosen
        r = (sim["r"].max()-sim["r"].min())/2
    com=None

    rarr = np.array(sim['r'])
    pos = np.array(sim['pos'])
    mass = np.array(sim['mass'])

    ind = np.where(rarr < r)[0]
    
    while len(ind)>min_particles or com is None :

        mtot = mass[ind].sum()
        com = np.sum(mass[ind]*pos[ind].transpose(), axis=1)/mtot

        r*=shrink_factor
        ind = np.where(np.sqrt(((pos-com)**2).sum(axis=1)) < r)[0]
        if verbose:
            print com,r,len(ind)
    return com

def get_r200(s,p) : 
    import pynbody
    
    ind = np.where(p['rbins'] > 100)

    den = (p['mass_enc']/(4./3.*np.pi*p['rbins']**3))[ind]
    den /= pynbody.analysis.cosmology.rho_crit(s,unit=den.units)

    return np.interp(200.0,den[::-1],p['rbins'][ind][::-1])

def make_spanned_colorbar(f,axs, label) : 
    # set the colorbar
    bb1 = axs[0,-1].get_position()
    bb2 = axs[1,-1].get_position()
    cbax = f.add_axes([bb1.x1+.01,bb2.y0,0.02,bb1.y1-bb2.y0])
    cb1 = f.colorbar(axs[1,-1].get_images()[0],cax=cbax)
    cb1.set_label(r'%s'%label,fontsize='smaller', fontweight='bold')
