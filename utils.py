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
