#
#
# Generally useful stuff
#
#

import matplotlib.pylab as plt
import numpy as np
import pynbody

def make_tile_figure(nrow,ncol,func,*args,**kwargs) : 

    fig = plt.figure(figsize=(5*ncol,5*nrow))

    for i in range(nrow) :
        for j in range(ncol) : 
            
            ax = fig.add_subplot(nrow,ncol,i*ncol+j+1)
            
            func(i*ncol+j,ax,*args,**kwargs)

            if (i < nrow-1) | (j > 0): clear_labels(ax)            
                
def clear_labels(ax,ticklines=False):
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_ylabel('')
    ax.set_xlabel('')
    if ticklines:
        for line in ax.get_xticklines() + ax.get_yticklines():
            line.set_markersize(0)


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
    
    ind = np.where(p['rbins'].in_units('kpc') > 100)

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

def make_rgb_image(s,width,xsize=500,ysize=500,filename='test.png') : 
    from PIL import Image
    from matplotlib.colors import Normalize

    rgbArray = np.zeros((xsize,ysize,3),'uint8')

    tem = pynbody.plot.image(s,qty='temp',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, denoise=True)
    rho = pynbody.plot.image(s,qty='rho',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, denoise=True)
    met = pynbody.plot.image(s,qty='metals',width=width,resolution=xsize,noplot=True,threaded=10,log=False,approximate_fast=False, denoise=True)
    
    rgbArray[...,0] = Normalize()(tem)*256#Normalize(vmin=3.5,vmax=6.5,clip=True)(tem)*256
    rgbArray[...,1] = Normalize()(rho)*256
    rgbArray[...,2] = Normalize()(np.log10(met/0.02))*256#Normalize(vmin=-3,vmax=0,clip=True)(np.log10(met/0.02))*256

    img = Image.fromarray(rgbArray)

    img.save(filename)
    
    return tem,rho,met


def make_rgb_stellar_image(s,width,vmin=6,vmax=12,xsize=500,ysize=500,filename='test.png') : 
    from PIL import Image
    from matplotlib.colors import Normalize
    from pickle import dump

    rgbArray = np.zeros((xsize,ysize,3),'uint8')

    R = pynbody.plot.image(s.s,qty='k_lum_den',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, av_z=True)
    G = pynbody.plot.image(s.s,qty='b_lum_den',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, av_z=True)
    B = pynbody.plot.image(s.s,qty='u_lum_den',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False,av_z=True)
    
    rgbArray[...,0] = Normalize(vmin=vmin,vmax=vmax,clip=True)(R)*256#Normalize(vmin=3.5,vmax=6.5,clip=True)(tem)*256
    rgbArray[...,1] = Normalize(vmin=vmin,vmax=vmax,clip=True)(G)*256
    rgbArray[...,2] = Normalize(vmin=vmin,vmax=vmax-.5,clip=True)(B)*256#Normalize(vmin=-3,vmax=0,clip=True)(np.log10(met/0.02))*256

    img = Image.fromarray(rgbArray)

    img.save(filename)
    dump({'R':R,'G':G,'B':B,'rgb':rgbArray},open(filename+'.imagedump','w'))

    return rgbArray, R, G, B
