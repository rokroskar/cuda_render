import numpy as np
import pynbody
import matplotlib.pylab as plt
from os import system
from camera import Camera


def make_rotating_movie(sim,nframes,x2=100, fileprefix = 'rotating', noticks = True, **kwargs) : 
    
    fg = plt.figure(figsize=(10.24,7.68))
    ax = plt.axes((0,0,1,1))

    for i in range(nframes) : 
        sim.rotate_y(365.0/nframes)
        im = pynbody.sph.threaded_render_image(sim.g,kernel=pynbody.sph.Kernel2D(), 
                                               x2=x2,nx=1024,ny=768, num_threads=10)
        if i == 0: 
            if not kwargs.has_key('vmin'): kwargs['vmin'] = np.log10(im.min())
            if not kwargs.has_key('vmax'): kwargs['vmax'] = np.log10(im.max())

        plt.imshow(np.log10(im), vmin=kwargs['vmin'], vmax=kwargs['vmax'], cmap=plt.cm.Blues_r)
        if noticks:
            for line in ax.get_xticklines() + ax.get_yticklines():
                line.set_markersize(0)

        plt.savefig(fileprefix+'%05d.png'%i,format='png')

    command = "mencoder 'mf://" + fileprefix+"*.png' -mf fps=25 -o " + fileprefix + ".avi -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=10000:mbd=2:keyint=132:v4mv:vqmin=3:lumi_mask=0.07:dark_mask=0.2:mpeg_quant:scplx_mask=0.1:tcplx_mask=0.1:naq"

    system(command)



def combine_frames(pref1, pref2, outpref) : 
    
    import glob

    flist1 = glob.glob(pref1+'*.png')
    flist1.sort()
    flist2 = glob.glob(pref2+'*.png')
    flist2.sort()
    
    assert len(flist1) == len(flist2)

    for i in range(len(flist1)) : 
        command = "convert " + flist1[i] + " " + flist2[i] + " -alpha on -compose dissolve -define compose:args=60 -gravity South -composite " + outpref + "%05d.png"%i
        system(command)

    command = "mencoder 'mf://" + outpref+"*.png' -mf fps=25 -o " + outpref + ".avi -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=10000:mbd=2:keyint=132:v4mv:vqmin=3:lumi_mask=0.07:dark_mask=0.2:mpeg_quant:scplx_mask=0.1:tcplx_mask=0.1:naq"

    system(command)


def render_spiral_camera_path(cam,nframes,rmax,**kwargs) : 
    from scipy.interpolate import spline
    
    t = np.linspace(0,8*np.pi,nframes)

    # make a spiral in x/y

    xs = 1./np.sqrt(t)*rmax*np.cos(t)
    ys = 1./np.sqrt(t)*rmax*np.sin(t)
    zs = rmax-t/t.max()*rmax

    points = np.vstack([xs,ys,zs]).T

    cam.nx = 200
    cam.ny = 200
    
    plt.figure(figsize=(10,10))

    for i,point in enumerate(points) : 
        cam.camera = point
        cam.update_transform()
        im = np.log10(pynbody.sph.threaded_render_image(cam.sim.g,num_threads=10,**cam.camera_dict))
        if i == 0 : 
            if not kwargs.has_key('vmin'):  kwargs['vmin'] = im.min()
            if not kwargs.has_key('vmax'):  kwargs['vmax'] = im.max()
        
        plt.imshow(im,**kwargs)
        plt.savefig('spiral_%05d.png'%i, format='png')

