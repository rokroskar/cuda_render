"""
The pynbody sph render code implemented in python and wrapped with numba
Runs ~ as fast or sometimes faster than the pynbody module written in C.

A 2D kernel is not implemented, so if you pass two_d = 1 to the render_image
it will process all particles from -inf < z < inf but the density will not 
be correct -- nevertheless, useful for timing purposes.

With the testdata output from the pynbody nosetests:

Rendering a slice:

In [45]: %timeit numba_sph_render.start_image_render(s.d,400,400,-.5,.5,timing=False)
10 loops, best of 3: 67.3 ms per loop

In [46]: %timeit pynbody.sph.render_image(s.d,x1=-.5,x2=.5,nx=400,ny=400,
threaded=False,approximate_fast=False,force_quiet=True)
10 loops, best of 3: 84.7 ms per loop

In [47]: %timeit numba_sph_render.start_image_render(s.d,1600,1600,-.5,.5,timing=False)
1 loops, best of 3: 858 ms per loop

In [48]: %timeit pynbody.sph.render_image(s.d,x1=-.5,x2=.5,nx=1600,ny=1600,
threaded=False,approximate_fast=False,force_quiet=True)
1 loops, best of 3: 814 ms per loop


Integrating along the entire box:

In [54]: %timeit numba_sph_render.start_image_render(s.d,400,400,-.5,.5,two_d=1)
1 loops, best of 3: 443 ms per loop

In [55]: %timeit pynbody.sph.render_image(s.d,x1=-.5,x2=.5,nx=400,ny=400,
threaded=False,approximate_fast=False,force_quiet=True,kernel=pynbody.sph.Kernel2D())
1 loops, best of 3: 856 ms per loop

"""

import numba
from numbapro import vectorize
from numba import autojit, jit, double, int32, void
import numpy as np

import time

@vectorize(['double(double,double)'])
def kernel_func(d, h) : 
    if d < 1 : 
        f = 1.-(3./2)*d**2 + (3./4.)*d**3
    elif d <= 2.0 :
        f = 0.25*(2.-d)**3
    else :
        f = 0
   
    return f/(np.pi*h**3)

@jit('int32(double,double,double)',nopython=True)
def physical_to_pixel(xpos,xmin,dx) : 
    return int32((xpos-xmin)/dx)

@jit('double(int32,double,double)', nopython=True)
def pixel_to_physical(xpix,x_start,dx) : 
    return dx*xpix+x_start

@autojit
def render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,ymin,ymax,two_d) : 
    MAX_D_OVER_H = 2.0

    image = np.zeros((nx,ny))

    Npart = len(xs)

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x_start = xmin+dx/2
    y_start = ymin+dy/2
    zplane = 0.0
    zpixel = zplane

    # set up the kernel values
    kernel_samples = np.arange(0,2.01,0.01,dtype=np.float)
    kernel_vals = kernel_func(kernel_samples,1.0)

    Npart_rendered = 0
    for i in xrange(Npart) : 
        x,y,z,h,qt = [double(xs[i]),double(ys[i]),double(zs[i]),double(hs[i]),double(qts[i]*mass[i]/rhos[i])]

        # is the particle in the frame?
        if ((x > xmin-2*h) and (x < xmax+2*h) and 
            (y > ymin-2*h) and (y < ymax+2*h) and
            (np.abs(z-zplane)*(1-two_d) < 2*h)) : 
            
            Npart_rendered += 1

            if h < dx*0.55 : h = dx*0.55        
            
            h3 = 1./(h*h*h)
            hsq = h*h
            h1 = 1./(.01*h)

            if (MAX_D_OVER_H*h/dx < 1 ) and (MAX_D_OVER_H*h/dy < 1) : 
                # pixel coordinates 
                xpos = int32(physical_to_pixel(x,xmin,dx))
                ypos = int32(physical_to_pixel(y,ymin,dy))
                # physical coordinates of pixel
                xpixel = pixel_to_physical(xpos,x_start,dx)
                ypixel = pixel_to_physical(ypos,y_start,dy)

                dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                dsq = dxpix*dxpix + dypix*dypix + dzpix*dzpix 

                if (xpos > 0) and (xpos < nx) and (ypos > 0) and (ypos < ny) and (dsq/hsq < 4) : 
                    image[ypos,xpos] += qt*kernel_vals[int32(np.sqrt(dsq)*h1)]*h3

            else :
                # bottom left of pixels the particle will contribute to
                x_pix_start = physical_to_pixel(x-MAX_D_OVER_H*h,xmin,dx)
                y_pix_start = physical_to_pixel(y-MAX_D_OVER_H*h,ymin,dy)
                # upper right of pixels the particle will contribute to
                x_pix_stop  = physical_to_pixel(x+MAX_D_OVER_H*h,xmin,dx)
                y_pix_stop  = physical_to_pixel(y+MAX_D_OVER_H*h,ymin,dy)
            
                if(x_pix_start < 0):  x_pix_start = 0
                if(x_pix_stop  > nx): x_pix_stop  = int32(nx-1)
                if(y_pix_start < 0):  y_pix_start = 0
                if(y_pix_stop  > ny): y_pix_stop  = int32(ny-1)
                
                for xpos in xrange(x_pix_start, x_pix_stop) :
                    # physical coordinates of pixel
                    xpixel = pixel_to_physical(xpos,x_start,dx)
                    
                    for ypos in xrange(y_pix_start, y_pix_stop) : 
                        # physical coordinates of pixel
                        ypixel = pixel_to_physical(ypos,y_start,dy)
                        dxpix, dypix, dzpix = [x-xpixel,y-ypixel,z-zpixel]
                        dsq = dxpix*dxpix+dypix*dypix+dzpix*dzpix
                        
                        if dsq/hsq < 4 : 
                            image[ypos,xpos] +=qt*kernel_vals[int32(np.sqrt(dsq)*h1)]*h3
                                                              
    return image, Npart_rendered


def start_image_render(s,nx,ny,xmin,xmax,two_d=0,timing=False) : 
    xs,ys,zs,hs,qts,mass,rhos = [s[arr] for arr in ['x','y','z','smooth','rho','mass','rho']]

    start = time.clock()
    im,N = render_image(xs,ys,zs,hs,qts,mass,rhos,nx,ny,xmin,xmax,xmin,xmax,two_d)
    if timing: print '<<< Rendering %d particles took %f s'%(N,time.clock()-start)
